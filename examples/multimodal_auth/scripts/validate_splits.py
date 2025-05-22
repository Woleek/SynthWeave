import argparse, json, sys, h5py, collections, itertools
from pathlib import Path
from typing import Dict, Set, Tuple
from tqdm import tqdm


def scan_split(h5_path: Path) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Returns
        subj2samples : id_target  -> #sample groups
        subj2windows : id_target  -> total #windows
    """
    subj2samples = collections.Counter()
    subj2windows = collections.Counter()

    with h5py.File(h5_path, "r") as h5f:
        for sid in tqdm(h5f, desc=h5_path.name):
            meta = json.loads(h5f[sid].attrs["metadata"])
            subj = str(meta["id_target"]) # ensure hashable
            nwin = h5f[sid]["video"].shape[0]

            subj2samples[subj] += 1
            subj2windows[subj] += nwin

    return subj2samples, subj2windows


def pretty_stats(name: str, subj2samples: Dict[str, int],
                 subj2windows: Dict[str, int]) -> str:
    n_subj   = len(subj2samples)
    n_samp   = sum(subj2samples.values())
    n_win    = sum(subj2windows.values())
    samp_avg = n_samp / n_subj if n_subj else 0
    win_avg  = n_win  / n_subj if n_subj else 0
    return (f"{name:5} | subjects: {n_subj:4d}  | "
            f"samples: {n_samp:5d} (avg {samp_avg:.1f})  | "
            f"windows: {n_win:6d} (avg {win_avg:.1f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True,
        help="folder that contains train.h5, dev.h5, test.h5")
    args = ap.parse_args()

    root = args.root.resolve()
    splits = ["train", "dev", "test"]
    split_files = {s: root / f"{s}.h5" for s in splits}
    for s,p in split_files.items():
        if not p.exists():
            sys.exit(f"❌  Missing file: {p}")

    stats_samples = {}
    stats_windows = {}
    subj_sets : Dict[str, Set[str]] = {}

    print("\n▶ Scanning splits …")
    for split in splits:
        ss, sw = scan_split(split_files[split])
        stats_samples[split] = ss
        stats_windows[split] = sw
        subj_sets[split]     = set(ss.keys())

    # print stats
    print("\n=== Split statistics ===")
    for split in splits:
        print(pretty_stats(split, stats_samples[split], stats_windows[split]))

    # check for overlaps
    print("\n=== Overlap check ===")
    ok = True
    for a,b in itertools.combinations(splits, 2):
        overlap = subj_sets[a] & subj_sets[b]
        if overlap:
            ok = False
            print(f"❌  {a} ↔ {b}: {len(overlap)} overlapping id_target(s):")
            print("     ", ", ".join(sorted(overlap)[:10]),
                  "…" if len(overlap) > 10 else "")
        else:
            print(f"✔️  {a} ↔ {b}: no overlap")

    if ok:
        print("\n✅  All splits are subject-disjoint\n")
        sys.exit(0)
    else:
        print("\n⚠️  Overlaps found!  Fix the splits and run again.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
