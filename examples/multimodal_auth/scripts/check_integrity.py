import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Set, Any

import h5py
import pandas as pd
from tqdm import tqdm

from synthweave.utils.datasets import get_dataset
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.pipe import ImagePreprocessor, AudioPreprocessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Check integrity of processed dataset")

    # Dataset options
    parser.add_argument(
        "--window_len", type=int, default=4, help="Window length used in processing"
    )
    parser.add_argument(
        "--window_step", type=int, default=1, help="Step size used in processing"
    )

    # Path options
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./processed_data",
        help="Directory with processed data",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        help="Which split to check ('train', 'dev', 'test', or 'all')",
    )
    parser.add_argument(
        "--fix", action="store_true", help="Attempt to fix inconsistencies"
    )

    return parser.parse_args()


def load_dataset(split_name: str, window_len: int, window_step: int):
    """Load the original dataset for comparison."""
    try:
        vid_proc = ImagePreprocessor(
            window_len=window_len,
            step=window_step,
            head_pose_dir="../../../models/head_pose",
        )
        aud_proc = AudioPreprocessor(window_len=window_len, step=window_step)

        ds_kwargs = {
            "video_processor": vid_proc,
            "audio_processor": aud_proc,
            "mode": "minimal",
        }

        ds = get_dataset("DeepSpeak_v1_1", split=split_name, **ds_kwargs)
        return ds
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return None


def check_h5_integrity(h5_path: Path) -> bool:
    """Check if the HDF5 file can be opened and has valid structure."""
    if not h5_path.exists():
        logger.error(f"HDF5 file does not exist: {h5_path}")
        return False

    try:
        with h5py.File(h5_path, "r") as h5f:
            sample_count = len(h5f.keys())
            logger.info(f"HDF5 file contains {sample_count} samples")

            # Check sample structure
            invalid_samples = []
            for sample_id in tqdm(h5f.keys(), desc="Checking H5 structure"):
                sample = h5f[sample_id]
                if "video" not in sample or "audio" not in sample:
                    invalid_samples.append(sample_id)
                    continue

                # Check metadata
                try:
                    metadata = json.loads(sample.attrs.get("metadata", "{}"))
                    if not isinstance(metadata, dict):
                        invalid_samples.append(sample_id)
                except:
                    invalid_samples.append(sample_id)

            if invalid_samples:
                logger.warning(
                    f"Found {len(invalid_samples)} samples with invalid structure"
                )
                logger.debug(f"Invalid samples: {invalid_samples[:10]}...")
                return False

            return True
    except Exception as e:
        logger.error(f"Error checking HDF5 file: {e}")
        return False


def check_flat_index_integrity(csv_path: Path, h5_path: Path) -> Dict[str, Any]:
    """Check if the flat index matches the HDF5 file."""
    if not csv_path.exists():
        logger.error(f"Flat index does not exist: {csv_path}")
        return {"valid": False}

    try:
        # Load flat index
        df = pd.read_csv(csv_path)
        index_sample_ids = set(df["sample_id"].unique())
        index_window_counts = df.groupby("sample_id").size().to_dict()

        # Load HDF5 sample IDs
        with h5py.File(h5_path, "r") as h5f:
            h5_sample_ids = set(h5f.keys())

            # Check actual window counts
            h5_window_counts = {}
            sample_errors = []

            for sample_id in h5_sample_ids:
                try:
                    if "video" in h5f[sample_id]:
                        h5_window_counts[sample_id] = h5f[sample_id]["video"].shape[0]
                    else:
                        sample_errors.append(f"{sample_id} (no video)")
                except Exception as e:
                    sample_errors.append(f"{sample_id} ({str(e)})")

            if sample_errors:
                logger.warning(
                    f"Found {len(sample_errors)} problematic samples: {sample_errors[:5]}..."
                )

        # Check for missing samples in index
        missing_in_index = h5_sample_ids - index_sample_ids
        missing_in_h5 = index_sample_ids - h5_sample_ids

        # Check for window count mismatches
        window_count_mismatches = {}
        for sample_id in h5_sample_ids.intersection(index_sample_ids):
            if sample_id in h5_window_counts and sample_id in index_window_counts:
                if h5_window_counts[sample_id] != index_window_counts[sample_id]:
                    window_count_mismatches[sample_id] = {
                        "h5": h5_window_counts[sample_id],
                        "index": index_window_counts[sample_id],
                    }

        result = {
            "valid": len(missing_in_index) == 0
            and len(missing_in_h5) == 0
            and len(window_count_mismatches) == 0,
            "missing_in_index": missing_in_index,
            "missing_in_h5": missing_in_h5,
            "window_count_mismatches": window_count_mismatches,
            "h5_sample_ids": h5_sample_ids,
            "index_sample_ids": index_sample_ids,
            "h5_window_counts": h5_window_counts,
            "index_window_counts": index_window_counts,
        }

        if not result["valid"]:
            logger.warning(f"Flat index has inconsistencies with HDF5:")
            logger.warning(f"- Samples in H5 but not in index: {len(missing_in_index)}")
            logger.warning(f"- Samples in index but not in H5: {len(missing_in_h5)}")
            logger.warning(f"- Window count mismatches: {len(window_count_mismatches)}")

            # List a few missing samples for debugging
            if missing_in_index:
                logger.debug(
                    f"First few missing in index: {list(missing_in_index)[:5]}"
                )
            if missing_in_h5:
                logger.debug(f"First few missing in H5: {list(missing_in_h5)[:5]}")
        else:
            logger.info("Flat index matches HDF5 file perfectly")

        return result
    except Exception as e:
        logger.error(f"Error checking flat index: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return {"valid": False, "error": str(e)}


def check_dataset_integrity(ds, h5_path: Path, index_info: Dict[str, Any]) -> bool:
    """Check if the processed data matches the original dataset."""
    if ds is None:
        return False

    try:
        with h5py.File(h5_path, "r") as h5f:
            h5_sample_count = len(h5f.keys())
            ds_sample_count = len(ds)

            logger.info(
                f"Dataset has {ds_sample_count} samples, HDF5 has {h5_sample_count} samples"
            )

            # We can only do a size comparison since sample IDs may not match directly
            if h5_sample_count < ds_sample_count:
                logger.warning(
                    f"HDF5 has fewer samples than dataset ({h5_sample_count} vs {ds_sample_count})"
                )
                return False
            elif h5_sample_count > ds_sample_count:
                logger.warning(
                    f"HDF5 has more samples than dataset ({h5_sample_count} vs {ds_sample_count})"
                )
                return False
            else:
                logger.info("HDF5 and dataset have the same number of samples")
                return True

            # Check window counts
            # expected_window_counts = []
            # for sample_idx, sample in enumerate(tqdm(ds, desc="Checking window counts")):
            #     vid_windows = sample["video"]
            #     expected_window_counts.append(len(vid_windows))

            # h5_window_counts = list(index_info["h5_window_counts"].values())

            # # total_expected_windows = sum(expected_window_counts)
            # total_h5_windows = sum(h5_window_counts)

            # logger.info(f"Dataset has {total_expected_windows} total windows, HDF5 has {total_h5_windows} total windows")

            # if total_h5_windows < total_expected_windows:
            #     logger.warning(f"HDF5 has fewer windows than expected ({total_h5_windows} vs {total_expected_windows})")
            #     return False

            # return True
    except Exception as e:
        logger.error(f"Error checking dataset integrity: {e}")
        return False


def fix_flat_index(csv_path: Path, h5_path: Path, index_info: Dict[str, Any]) -> bool:
    """Fix inconsistencies in the flat index."""
    try:
        # Extract missing sample IDs
        missing_in_index = index_info.get("missing_in_index", set())
        window_count_mismatches = index_info.get("window_count_mismatches", {})

        if not missing_in_index and not window_count_mismatches:
            logger.info("No flat index fixes needed")
            return True

        # Add more detailed logging
        logger.info(f"Fixing {len(missing_in_index)} missing samples in index")
        for sample_id in list(missing_in_index)[:5]:  # Log a few examples
            logger.debug(f"Sample to fix: {sample_id}")

        # Load existing data
        df = (
            pd.read_csv(csv_path)
            if csv_path.exists()
            else pd.DataFrame(columns=["sample_id", "window_idx"])
        )

        # Fix missing samples - directly access the data
        new_rows = []
        with h5py.File(h5_path, "r") as h5f:
            # Add missing samples
            for sample_id in sorted(missing_in_index):
                try:
                    if sample_id in h5f and "video" in h5f[sample_id]:
                        n_windows = h5f[sample_id]["video"].shape[0]
                        logger.debug(
                            f"Adding {n_windows} windows for sample {sample_id}"
                        )
                        for i in range(n_windows):
                            new_rows.append({"sample_id": sample_id, "window_idx": i})
                    else:
                        logger.warning(
                            f"Sample {sample_id} exists but has no video data"
                        )
                except Exception as e:
                    logger.error(f"Error processing sample {sample_id}: {e}")

            # Fix window count mismatches
            for sample_id, counts in window_count_mismatches.items():
                # Remove existing entries
                df = df[df["sample_id"] != sample_id]

                # Add correct entries
                if sample_id in h5f and "video" in h5f[sample_id]:
                    n_windows = h5f[sample_id]["video"].shape[0]
                    for i in range(n_windows):
                        new_rows.append({"sample_id": sample_id, "window_idx": i})

        # Add new rows
        if new_rows:
            logger.info(f"Adding {len(new_rows)} entries to the index")
            new_df = pd.DataFrame(new_rows)
            df = pd.concat([df, new_df], ignore_index=True)

        # Remove samples that don't exist in H5
        h5_sample_ids = index_info.get("h5_sample_ids", set())
        if h5_sample_ids:
            original_len = len(df)
            df = df[df["sample_id"].isin(h5_sample_ids)]
            removed = original_len - len(df)
            if removed > 0:
                logger.info(f"Removed {removed} entries for samples not in H5")

        # Save the fixed index
        backup_path = csv_path.with_suffix(".csv.bak")
        if csv_path.exists():
            shutil.copy2(csv_path, backup_path)
            logger.info(f"Backed up original index to {backup_path}")

        # Ensure we have data to write
        if len(df) == 0:
            logger.warning("No data to write to index - this is unusual")
            return False

        # Write to a temporary file first
        temp_csv = csv_path.with_suffix(".csv.tmp")
        df.to_csv(temp_csv, index=False)

        # Check the temporary file was written correctly
        if not temp_csv.exists() or temp_csv.stat().st_size == 0:
            logger.error("Failed to write temporary index file")
            return False

        # Replace the original file with the temporary file
        if temp_csv.exists():
            temp_csv.replace(csv_path)
            logger.info(f"Fixed flat index saved to {csv_path}")

        # Recreate JSON index
        json_path = csv_path.with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(df.to_dict(orient="records"), f, indent=4)
        logger.info(f"Updated JSON index: {json_path}")

        # Verify the fix
        if missing_in_index or window_count_mismatches:
            logger.info("Verifying the fix...")
            df_check = pd.read_csv(csv_path)
            fixed_sample_ids = set(df_check["sample_id"].unique())
            missing_after_fix = h5_sample_ids - fixed_sample_ids

            if missing_after_fix:
                logger.warning(
                    f"After fix, still missing {len(missing_after_fix)} samples in index"
                )
                for sid in list(missing_after_fix)[:5]:
                    logger.debug(f"Still missing: {sid}")
                return False

        return True
    except Exception as e:
        logger.error(f"Error fixing flat index: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return False


def check_split(
    split_name: str,
    data_dir: Path,
    window_len: int,
    window_step: int,
    fix: bool = False,
) -> bool:
    """Check integrity of a single dataset split."""
    logger.info(f"\n===== Checking {split_name} split =====")

    h5_path = data_dir / f"{split_name}.h5"
    csv_path = data_dir / f"{split_name}_flat_index.csv"

    # Check HDF5 integrity
    h5_valid = check_h5_integrity(h5_path)
    if not h5_valid:
        logger.error(f"HDF5 file has integrity issues: {h5_path}")
        return False

    # Check flat index integrity
    index_info = check_flat_index_integrity(csv_path, h5_path)
    index_valid = index_info["valid"]

    # Only try to fix flat index if --fix flag is set
    if not index_valid and fix:
        logger.info("Attempting to fix flat index...")
        fix_result = fix_flat_index(csv_path, h5_path, index_info)
        if fix_result:
            logger.info("Flat index fixed successfully")
            # Re-check index integrity
            index_info = check_flat_index_integrity(csv_path, h5_path)
            index_valid = index_info["valid"]
    elif not index_valid:
        # Just print what would be fixed
        missing_in_index = len(index_info.get("missing_in_index", []))
        missing_in_h5 = len(index_info.get("missing_in_h5", []))
        window_mismatches = len(index_info.get("window_count_mismatches", {}))

        logger.info(f"Index could be fixed with --fix flag to address:")
        logger.info(f"- {missing_in_index} samples missing in index")
        logger.info(f"- {missing_in_h5} samples in index but not in H5")
        logger.info(f"- {window_mismatches} samples with window count mismatches")

    # Check against original dataset
    ds = load_dataset(split_name, window_len, window_step)
    ds_valid = check_dataset_integrity(ds, h5_path, index_info)

    overall_valid = h5_valid and index_valid and ds_valid

    if overall_valid:
        logger.info(f"{split_name} split integrity check: PASSED")
    else:
        logger.warning(f"{split_name} split integrity check: FAILED")

    return overall_valid


def main(args: argparse.Namespace):
    data_dir = Path(args.data_dir) / "DeepSpeak_v1_1"

    if not data_dir.exists():
        logger.error(f"Data directory does not exist: {data_dir}")
        return

    # Check if config file exists
    config_path = data_dir / "processing_config.json"
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
                logger.info(f"Loaded config: {config}")

                # Override args with config if not explicitly provided
                if args.window_len == 4 and "window_len" in config:  # 4 is the default
                    args.window_len = config["window_len"]
                if (
                    args.window_step == 1 and "window_step" in config
                ):  # 1 is the default
                    args.window_step = config["window_step"]
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")

    splits_to_check = ["train", "dev", "test"] if args.split == "all" else [args.split]

    all_valid = True
    for split in splits_to_check:
        split_valid = check_split(
            split, data_dir, args.window_len, args.window_step, args.fix
        )
        all_valid = all_valid and split_valid

    if all_valid:
        logger.info("\n===== All integrity checks PASSED =====")
    else:
        logger.warning("\n===== Some integrity checks FAILED =====")


if __name__ == "__main__":
    args = parse_args()
    main(args)
