from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple


def list_graphs(dir_path: Path) -> List[Path]:
    return sorted(dir_path.glob("*.graph.pt"))


def write_list(paths: List[Path], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(p.as_posix() for p in paths) + ("\n" if paths else ""))


def split(paths: List[Path], test_ratio: float, seed: int) -> Tuple[List[Path], List[Path]]:
    rng = random.Random(seed)
    paths = list(paths)
    rng.shuffle(paths)
    n_test = max(1, int(len(paths) * test_ratio))
    test = paths[:n_test]
    train = paths[n_test:]
    return train, test


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qut", type=str, default="artifacts/qut_a")
    ap.add_argument("--osp", type=str, default="artifacts/osp_a")
    ap.add_argument("--out", type=str, default="splits")
    ap.add_argument("--test-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    outdir = Path(args.out)
    qut = list_graphs(Path(args.qut))
    osp = list_graphs(Path(args.osp))
    if not qut:
        raise FileNotFoundError(f"No *.graph.pt under {args.qut}")
    if not osp:
        raise FileNotFoundError(f"No *.graph.pt under {args.osp}")

    qut_train, qut_test = split(qut, args.test_ratio, args.seed)
    osp_train, osp_test = split(osp, args.test_ratio, args.seed)

    write_list(qut_train, outdir / "qut_train.txt")
    write_list(qut_test, outdir / "qut_test.txt")
    write_list(osp_train, outdir / "osp_train.txt")
    write_list(osp_test, outdir / "osp_test.txt")

    # Cross-domain
    write_list(qut_train, outdir / "cross_train_qut.txt")
    write_list(osp_test, outdir / "cross_test_osp.txt")

    print("Wrote splits to", outdir.as_posix())


if __name__ == "__main__":
    main()

