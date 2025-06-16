import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any

import h5py
import pandas as pd
import torch
from tqdm import tqdm

from synthweave.utils.datasets import get_dataset
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.pipe import AdaFace, ReDimNet

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Process and encode multimodal data")

    # Dataset options
    parser.add_argument(
        "--dataset",
        choices=["DeepSpeak_v1_1", "SWAN_DF"],
        required=True,
        help="Which dataset to load",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the dataset directory",
    )
    # SWAN-DF specific
    parser.add_argument(
        "--resolutions",
        nargs="*",
        default=None,
        help="Subset of fake resolutions to keep",
        choices=[None, "160", "256", "320"],
    )

    # Save options
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./encoded_data",
        help="Directory to save encoded data",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from existing files"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite existing files without prompting",
    )

    return parser.parse_args()


def load_encoders(device):
    """Load encoders if encoding is enabled."""

    logger.info("Loading encoders...")
    try:
        aud_encoder = ReDimNet(freeze=True)
        aud_encoder.to(device)
        aud_encoder.eval()
        img_encoder = AdaFace(path="../../../models", freeze=True)
        img_encoder.to(device)
        img_encoder.eval()
        return img_encoder, aud_encoder
    except Exception as e:
        logger.error(f"Failed to load encoders: {e}")
        raise


def get_existing_data(h5_path: Path, csv_path: Path) -> Tuple[Set[str], Set[str], int]:
    """
    Get existing sample IDs from H5 file and index file.
    Returns (h5_sample_ids, indexed_sample_ids, max_sample_id)
    """
    h5_sample_ids = set()
    max_sample_id = -1

    # Get sample IDs from H5 file
    if h5_path.exists():
        try:
            with h5py.File(h5_path, "r") as h5f:
                h5_sample_ids = set(h5f.keys())
                # Extract max sample ID for continuing numbering
                if h5_sample_ids:
                    sample_nums = [int(sid.split("_")[1]) for sid in h5_sample_ids]
                    max_sample_id = max(sample_nums)
        except Exception as e:
            logger.error(f"Error reading H5 file: {e}")
            h5_sample_ids = set()

    # Get sample IDs from index file
    indexed_sample_ids = set()
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            indexed_sample_ids = set(df["sample_id"].unique())
        except Exception as e:
            logger.error(f"Error reading index file: {e}")

    return h5_sample_ids, indexed_sample_ids, max_sample_id


def create_or_validate_directories(save_dir: str, dataset_name: str) -> Path:
    """Create and validate output directories."""
    root_dir = Path(save_dir) / dataset_name
    root_dir.mkdir(parents=True, exist_ok=True)
    return root_dir


def handle_existing_files(paths: List[Path], resume: bool, force: bool) -> bool:
    """Handle existing files based on resume and force flags."""
    existing_files = [p for p in paths if p.exists()]

    if not existing_files:
        return True

    if resume:
        logger.info(
            f"Resuming with existing files: {', '.join(str(p) for p in existing_files)}"
        )
        return True

    if force:
        for p in existing_files:
            logger.info(f"Removing existing file: {p}")
            p.unlink()
        return True

    # Ask for confirmation
    confirm = input(
        f"⚠️ Files exist: {', '.join(str(p) for p in existing_files)}. Overwrite? (y/n): "
    )
    if confirm.lower() != "y":
        logger.info("Operation aborted by user.")
        return False

    for p in existing_files:
        p.unlink()
    return True


def recover_missing_index_entries(
    h5_path: Path, csv_path: Path, h5_ids: Set[str], indexed_ids: Set[str]
) -> bool:
    """Recover missing entries in the index file."""
    missing_ids = h5_ids - indexed_ids
    if not missing_ids:
        return True

    logger.info(f"Recovering {len(missing_ids)} missing index entries...")

    try:
        with h5py.File(h5_path, "r") as h5f, open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["sample_id", "window_idx"])
            if f.tell() == 0:
                writer.writeheader()

            for sid in sorted(missing_ids):
                try:
                    if "video" not in h5f[sid]:
                        logger.warning(
                            f"Sample {sid} has no video data, skipping recovery"
                        )
                        continue

                    n = h5f[sid]["video"].shape[0]
                    for i in range(n):
                        writer.writerow({"sample_id": sid, "window_idx": i})
                except Exception as e:
                    logger.error(f"Failed to recover {sid}: {e}")
        return True
    except Exception as e:
        logger.error(f"Failed to recover index entries: {e}")
        return False


def encode_sample(
    sample: Dict[str, Any],
    sample_id: str,
    h5f: h5py.File,
    csv_writer: csv.DictWriter,
    ds,
    img_encoder,
    aud_encoder,
    device,
) -> bool:
    """Encode a single sample and save to H5 file."""
    try:
        # Load data
        vid_windows = sample["video"]
        vid_windows = vid_windows.to(device)
        aud_windows = sample["audio"]
        aud_windows = aud_windows.to(device)

        # Encode
        with torch.no_grad():
            vid_windows = img_encoder(vid_windows)
            aud_windows = aud_encoder(aud_windows)
        vid_windows = vid_windows.detach().cpu()
        aud_windows = aud_windows.detach().cpu()

        # Get metadata
        meta = sample["metadata"]
        meta["label"] = ds.encoders["label"].inverse_transform([meta["label"]])[0]
        meta["av"] = ds.encoders["av"].inverse_transform([meta["av"]])[0]

        # Convert NumPy types to native Python types for JSON serialization
        serializable_meta = {}
        for key, value in meta.items():
            if hasattr(value, "item") and callable(value.item):  # NumPy scalar types
                serializable_meta[key] = value.item()
            else:
                serializable_meta[key] = value

        # Create group and datasets
        grp = h5f.create_group(sample_id)
        grp.create_dataset(
            "video",
            data=vid_windows.numpy() if hasattr(vid_windows, "numpy") else vid_windows,
            compression="gzip",
        )
        grp.create_dataset(
            "audio",
            data=aud_windows.numpy() if hasattr(aud_windows, "numpy") else aud_windows,
            compression="gzip",
        )
        grp.attrs["metadata"] = json.dumps(serializable_meta)

        # Update index
        for i in range(len(vid_windows)):
            csv_writer.writerow({"sample_id": sample_id, "window_idx": i})

        # Cleanup on large samples
        if vid_windows.shape[0] > 30:
            del vid_windows, aud_windows, sample
            torch.cuda.empty_cache()

        return True
    except Exception as e:
        logger.error(f"Error processing sample {sample_id}: {e}")
        # Try to remove the partially created group if it exists
        if sample_id in h5f:
            try:
                del h5f[sample_id]
            except:
                pass
        return False


def encode_split(
    ds,
    split_name: str,
    root_dir: Path,
    img_encoder,
    aud_encoder,
    resume: bool = False,
    force: bool = False,
    start_idx: int = 0,
    device="cpu",
) -> bool:
    """Encode a single dataset split."""
    logger.info(f"Encoding {split_name} split...")

    h5_path = root_dir / f"{split_name}.h5"
    csv_path = root_dir / f"{split_name}_flat_index.csv"
    json_path = root_dir / f"{split_name}_flat_index.json"

    # Check if files should be handled
    if not handle_existing_files([h5_path, csv_path, json_path], resume, force):
        return False

    # Get existing data if resuming
    h5_ids, indexed_ids, max_sample_id = set(), set(), -1
    existing_count = 0

    if resume:
        h5_ids, indexed_ids, max_sample_id = get_existing_data(h5_path, csv_path)
        existing_count = len(h5_ids)
        logger.info(
            f"Found {existing_count} samples in H5 file, {len(indexed_ids)} in index"
        )

        # Recover missing index entries
        if h5_ids and not recover_missing_index_entries(
            h5_path, csv_path, h5_ids, indexed_ids
        ):
            logger.error("Failed to recover index, aborting")
            return False

    # Determine starting sample counter and index
    sample_counter = max_sample_id + 1 if max_sample_id >= 0 else start_idx
    start_local_idx = existing_count

    # Open files for writing
    h5_mode = "a" if resume else "w"
    csv_mode = "a" if resume and csv_path.exists() else "w"

    success = True
    encoded_count = 0
    total_samples = len(ds)

    # === Main processing ===
    with h5py.File(h5_path, h5_mode) as h5f, open(
        csv_path, csv_mode, newline=""
    ) as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=["sample_id", "window_idx"])
        if csv_mode == "w" or csv_file.tell() == 0:
            csv_writer.writeheader()

        # Create a progress bar for the remaining samples only
        remaining_count = total_samples - existing_count
        logger.info(
            f"Starting from sample index {start_local_idx}/{total_samples} "
            + f"(skipping {existing_count} already encoded samples)"
        )

        # Optimize by skipping already encoded samples
        pbar = tqdm(total=remaining_count, desc=split_name)

        try:
            # Skip to the first not encoded sample
            if start_local_idx > 0:
                logger.info(f"Skipping to sample {start_local_idx}...")

            # Only iterate over remaining samples
            for local_idx in range(start_local_idx, total_samples):
                sample_id = f"sample_{sample_counter:06d}"

                # if (split_name == 'train' and local_idx in [2214]) or (split_name == 'test' and local_idx in [2541, 2546]):
                #     # error samples
                #     sample_counter += 1
                #     pbar.update(1)
                #     continue

                # Skip if sample already exists (shouldn't happen but safety check)
                if sample_id in h5_ids:
                    sample_counter += 1
                    pbar.update(1)
                    continue

                # Get the sample
                sample = ds[local_idx]

                # Process the sample
                if encode_sample(
                    sample,
                    sample_id,
                    h5f,
                    csv_writer,
                    ds,
                    img_encoder,
                    aud_encoder,
                    device,
                ):
                    encoded_count += 1
                    pbar.set_postfix(
                        {
                            "encoded": encoded_count,
                            "total": existing_count + encoded_count,
                        }
                    )

                # Increment counter and periodically flush
                sample_counter += 1
                pbar.update(1)

                if encoded_count % 10 == 0:
                    h5f.flush()
                    csv_file.flush()

        except KeyboardInterrupt:
            logger.warning("Encoding interrupted by user")
            success = False
        except Exception as e:
            logger.error(f"Error during encoding: {e}")
            success = False
        finally:
            h5f.flush()
            csv_file.flush()
            pbar.close()

    # Create JSON index from CSV
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            with open(json_path, "w") as f:
                json.dump(df.to_dict(orient="records"), f, indent=4)
            logger.info(f"Created JSON index: {json_path}")
        except Exception as e:
            logger.error(f"Failed to create JSON index: {e}")
            success = False

    logger.info(
        f"Encoded {encoded_count} new samples for {split_name} split (total: {existing_count + encoded_count})"
    )
    return success


def main(args: argparse.Namespace):
    # Create output directory
    if args.dataset == "SWAN_DF" and args.resolutions is not None:
        assert None not in args.resolutions, "Resolutions cannot be None"
        ds_name = "SWAN_DF_" + "_".join(args.resolutions)
    else:
        ds_name = args.dataset
    root_dir = create_or_validate_directories(args.save_dir, ds_name)

    # Dump configuration
    config = {
        "data_dir": args.data_dir,
        "preprocessed": True,
        "video_encoder": "AdaFace",
        "audio_encoder": "ReDimNet",
    }

    with open(root_dir / "encoding_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Load encoders if needed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_encoder, aud_encoder = load_encoders(device)

    # Prepare datasets
    ds_kwargs = {
        "data_dir": args.data_dir,
        "preprocessed": True,
        "sample_mode": "sequence",
    }

    if args.dataset == "SWAN_DF":
        ds_kwargs["av_codes"] = ["00", "11"]

    logger.info("Loading datasets...")
    datasets = {}
    try:
        for split in ["train", "dev", "test"]:
            datasets[split] = get_dataset(args.dataset, split=split, **ds_kwargs)
    except Exception as e:
        logger.error(f"Failed to load datasets: {e}")
        return

    # Process each split
    all_success = True
    for split_name, ds in datasets.items():
        success = encode_split(
            ds,
            split_name,
            root_dir,
            img_encoder,
            aud_encoder,
            resume=args.resume,
            force=args.force,
            device=device,
        )
        if not success:
            all_success = False
            logger.warning(
                f"Encoding for {split_name} split did not complete successfully"
            )

    if all_success:
        logger.info("All Encoding completed successfully!")
    else:
        logger.warning("Encoding completed with errors. Check logs for details.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
