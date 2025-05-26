import argparse, json, sys, h5py, collections, itertools
from pathlib import Path
from typing import Dict, List, Set, Tuple
from tqdm import tqdm


def scan_split(h5_path: Path) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int], Dict[str, int]]:
    """
    Returns
        subj2samples : id_target  -> #sample groups
        subj2windows : id_target  -> total #windows
        label_counts : label      -> total samples for this label
        av_counts:   : av         -> total samples for this av
    """
    subj2samples = collections.Counter()
    subj2windows = collections.Counter()
    label_counts = collections.Counter()
    av_counts    = collections.Counter()

    with h5py.File(h5_path, "r") as h5f:
        for sid in tqdm(h5f, desc=h5_path.name):
            meta = json.loads(h5f[sid].attrs["metadata"])
            subj = str(meta["id_target"]) # ensure hashable
            nwin = h5f[sid]["video"].shape[0]
            label = meta["label"]
            av = meta["av"]

            subj2samples[subj] += 1
            subj2windows[subj] += nwin
            label_counts[label] += 1
            av_counts[av] += 1

    return subj2samples, subj2windows, label_counts, av_counts


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
    
def format_label_row(name: str, label_counts: Dict[str, int], keys: List[str]) -> str:
    total = sum(label_counts.values())
    row = f"{name:5} |"
    for k in keys:
        v = label_counts.get(k, 0)
        p = (v / total) * 100 if total else 0
        row += f" {k:>2}: {v:5d} ({p:5.1f}%) |"
    return row
    
def pretty_label_stats(name: str, label_counts: Dict[str, int], av_counts: Dict[str, int]) -> str:
    print(format_label_row(name, label_counts, ["0", "1"]))
    print(format_label_row(name, av_counts, ["00", "01", "10", "11"]))


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
    label_stats = {}
    av_stats = {}
    subj_sets : Dict[str, Set[str]] = {}

    print("\n▶ Scanning splits …")
    for split in splits:
        ss, sw, lc, avc = scan_split(split_files[split])
        stats_samples[split] = ss
        stats_windows[split] = sw
        label_stats[split] = lc
        av_stats[split] = avc
        subj_sets[split]     = set(ss.keys())

    # print stats
    print("\n=== Split statistics ===")
    for split in splits:
        print(pretty_stats(split, stats_samples[split], stats_windows[split]))
        
    # print label distribution
    print("\n=== Label & AV distribution ===")
    for split in splits:
        pretty_label_stats(split, label_stats[split], av_stats[split])
        print(" " * 5, "-" * 85)

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
