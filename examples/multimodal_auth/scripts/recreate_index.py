import argparse
import json
import logging
import sys
from pathlib import Path
import h5py
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Recreate flat index from HDF5 file")
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Directory with processed data"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        help="Which split to fix ('train', 'dev', 'test', or 'all')",
    )
    parser.add_argument(
        "--backup", action="store_true", help="Create backup of existing index files"
    )
    return parser.parse_args()


def recreate_index(h5_path, csv_path, json_path, create_backup=False):
    """Completely recreate the flat index from the HDF5 file."""
    if not h5_path.exists():
        logger.error(f"HDF5 file does not exist: {h5_path}")
        return False

    try:
        # Create backup if requested
        if create_backup and csv_path.exists():
            backup_path = csv_path.with_suffix(".csv.bak")
            csv_path.rename(backup_path)
            logger.info(f"Created backup of CSV index: {backup_path}")

        if create_backup and json_path.exists():
            backup_path = json_path.with_suffix(".json.bak")
            json_path.rename(backup_path)
            logger.info(f"Created backup of JSON index: {backup_path}")

        # Read all samples from HDF5
        rows = []
        with h5py.File(h5_path, "r") as h5f:
            total_samples = len(h5f.keys())
            logger.info(f"Found {total_samples} samples in HDF5 file")

            # Generate rows for each sample and window
            for sample_id in h5f.keys():
                if "video" not in h5f[sample_id]:
                    logger.warning(f"Sample {sample_id} has no video data, skipping")
                    continue

                n_windows = h5f[sample_id]["video"].shape[0]
                for window_idx in range(n_windows):
                    rows.append({"sample_id": sample_id, "window_idx": window_idx})

        # Create dataframe and save as CSV
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)

        # Create JSON from dataframe
        with open(json_path, "w") as f:
            json.dump(df.to_dict(orient="records"), f, indent=4)

        logger.info(f"Created new CSV index with {len(df)} entries: {csv_path}")
        logger.info(f"Created new JSON index: {json_path}")
        return True

    except Exception as e:
        logger.error(f"Error recreating index: {e}")
        return False


def main(args):
    data_dir = Path(args.data_dir) / "DeepSpeak_v1_1"

    if not data_dir.exists():
        logger.error(f"Data directory does not exist: {data_dir}")
        return

    splits = ["train", "dev", "test"] if args.split == "all" else [args.split]

    for split in splits:
        logger.info(f"Processing {split} split...")

        h5_path = data_dir / f"{split}.h5"
        csv_path = data_dir / f"{split}_flat_index.csv"
        json_path = data_dir / f"{split}_flat_index.json"

        success = recreate_index(h5_path, csv_path, json_path, args.backup)

        if success:
            logger.info(f"Successfully recreated index for {split} split")
        else:
            logger.error(f"Failed to recreate index for {split} split")


if __name__ == "__main__":
    args = parse_args()
    main(args)
