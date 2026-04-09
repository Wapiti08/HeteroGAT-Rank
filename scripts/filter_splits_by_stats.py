from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def read_list(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def write_list(lines: List[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def summarize(name: str, df: pd.DataFrame, files: List[str]) -> None:
    sub = df[df["path"].isin(files)]
    n = len(sub)
    if n == 0:
        print(f"{name}: n=0")
        return
    y = sub["y"].dropna()
    n_labeled = len(y)
    pos = int((y == 1).sum()) if n_labeled else 0
    neg = int((y == 0).sum()) if n_labeled else 0
    pos_ratio = pos / max(1, (pos + neg)) if n_labeled else None
    only_load = float((sub["non_load_ratio"] == 0).mean())
    print(f"{name}: n={n} labeled={n_labeled} pos={pos} neg={neg} pos_ratio={pos_ratio} only_load_ratio={only_load:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", type=str, required=True, help="Path to stats_per_graph.csv")
    ap.add_argument("--train-list", type=str, required=True)
    ap.add_argument("--test-list", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--mode", type=str, default="non_load_ratio", choices=["non_load_ratio", "edges_total"])
    ap.add_argument("--threshold", type=float, default=0.0, help="For mode=non_load_ratio: keep > threshold")
    ap.add_argument("--min-edges", type=int, default=1, help="For mode=edges_total: keep > min-edges")
    args = ap.parse_args()

    stats_path = Path(args.stats)
    train_path = Path(args.train_list)
    test_path = Path(args.test_list)
    outdir = Path(args.out_dir)

    df = pd.read_csv(stats_path)
    # Normalize path column: store absolute/relative string as used in split lists.
    # We will also accept matching by basename.
    if "path" not in df.columns:
        # Reconstruct path from id if needed (expects files to be in same dir as stats input)
        raise ValueError("stats csv must include a 'path' column (use scripts/stats_canonical_graphs.py output)")

    # ensure string
    df["path"] = df["path"].astype(str)

    train_files = read_list(train_path)
    test_files = read_list(test_path)

    # Try to match by full path; if stats has relative paths but lists have absolute (or vice versa),
    # also build a basename map.
    df["basename"] = df["path"].map(lambda s: Path(s).name)
    by_basename = dict(zip(df["basename"], df["path"]))

    def normalize(files: List[str]) -> List[str]:
        out = []
        for f in files:
            if f in set(df["path"]):
                out.append(f)
            else:
                b = Path(f).name
                if b in by_basename:
                    out.append(by_basename[b])
        return out

    train_norm = normalize(train_files)
    test_norm = normalize(test_files)

    summarize("before/train", df, train_norm)
    summarize("before/test", df, test_norm)

    if args.mode == "non_load_ratio":
        keep_train = df[(df["path"].isin(train_norm)) & (df["non_load_ratio"] > args.threshold)]["path"].tolist()
        keep_test = df[(df["path"].isin(test_norm)) & (df["non_load_ratio"] > args.threshold)]["path"].tolist()
    else:
        keep_train = df[(df["path"].isin(train_norm)) & (df["num_edges_total"] > args.min_edges)]["path"].tolist()
        keep_test = df[(df["path"].isin(test_norm)) & (df["num_edges_total"] > args.min_edges)]["path"].tolist()

    summarize("after/train", df, keep_train)
    summarize("after/test", df, keep_test)

    write_list(keep_train, outdir / "train.txt")
    write_list(keep_test, outdir / "test.txt")
    print("Wrote:", (outdir / "train.txt").as_posix())
    print("Wrote:", (outdir / "test.txt").as_posix())


if __name__ == "__main__":
    main()

