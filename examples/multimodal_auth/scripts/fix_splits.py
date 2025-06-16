import argparse, json, random, h5py, pandas as pd
from pathlib import Path
from tqdm import tqdm


def scan_subjects(h5_path: Path):
    """Return list of (sample_id, id_target, n_windows)."""
    rows = []
    with h5py.File(h5_path, "r") as h5f:
        for sid in h5f:
            meta = json.loads(h5f[sid].attrs["metadata"])
            n_win = h5f[sid]["video"].shape[0]
            rows.append((sid, meta["id_target"], n_win))
    return rows


def decide_partition(subjects, ratio=0.9, seed=42):
    """Return dict  id_target -> 'train' | 'dev'  with ~ratio in train."""
    random.seed(seed)
    subs = sorted(subjects)
    random.shuffle(subs)
    cut = int(len(subs) * ratio)
    return {s: ("train" if i < cut else "dev") for i, s in enumerate(subs)}

counters = {"train": 0, "dev": 0}

def copy_groups(src_path: Path, dst_files: dict, mapping: dict):
    """Copy each sample group to its new split."""
    with h5py.File(src_path, "r") as src:
        for sid in tqdm(src, desc=f"copying from {src_path.name}"):
            tgt = json.loads(src[sid].attrs["metadata"])["id_target"]
            split = mapping.get(tgt)
            if split:  # only train/dev are rebuilt
                new_sid = f"sample_{counters[split]:06d}"
                counters[split] += 1
                src.copy(src[sid], dst_files[split], name=new_sid)


def build_index(h5_path: Path, csv_path: Path, json_path: Path):
    rows = []
    with h5py.File(h5_path, "r") as h5f:
        for sid in h5f:
            n = h5f[sid]["video"].shape[0]
            rows += [(sid, i) for i in range(n)]
    df = pd.DataFrame(rows, columns=["sample_id", "window_idx"])
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=Path,
        required=True,
        help="folder that contains train.h5, dev.h5, test.h5",
    )
    ap.add_argument(
        "--ratio", type=float, default=0.9, help="fraction of subjects that go to train"
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = args.root.resolve()
    new_root = root.parent / (root.name + "_fixed")
    new_root.mkdir(parents=True, exist_ok=True)

    # gather metadata
    train_rows = scan_subjects(root / "train.h5")
    dev_rows = scan_subjects(root / "dev.h5")
    df = pd.DataFrame(
        train_rows + dev_rows, columns=["sample_id", "id_target", "n_win"]
    )

    mapping = decide_partition(
        df["id_target"].unique(), ratio=args.ratio, seed=args.seed
    )

    # open new files
    dst = {s: h5py.File(new_root / f"{s}.h5", "w") for s in ("train", "dev")}

    # copy groups
    copy_groups(root / "train.h5", dst, mapping)
    copy_groups(root / "dev.h5", dst, mapping)

    # flush & close
    for f in dst.values():
        f.flush()
        f.close()

    # rebuild flat indices
    build_index(
        new_root / "train.h5",
        new_root / "train_flat_index.csv",
        new_root / "train_flat_index.json",
    )
    build_index(
        new_root / "dev.h5",
        new_root / "dev_flat_index.csv",
        new_root / "dev_flat_index.json",
    )

    # copy test files
    for ext in (".h5", "_flat_index.csv", "_flat_index.json"):
        src = root / f"test{ext}"
        dst = new_root / f"test{ext}"
        if not dst.exists():
            dst.write_bytes(src.read_bytes())

    print(f"✅  Fixed splits written to {new_root}")


if __name__ == "__main__":
    main()
