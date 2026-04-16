from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set, Tuple

import torch
from torch_geometric.data import HeteroData

# Ensure repo root importable when running as script.
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from ranking_explain.rarity import RarityStats, save_rarity_stats  # noqa: E402


EType = Tuple[str, str, str]


def load_graph(path: Path) -> HeteroData:
    obj = torch.load(path, map_location="cpu")
    return HeteroData.from_dict(obj["data_dict"])


def get_label(path: Path) -> int | None:
    obj = torch.load(path, map_location="cpu")
    y = obj.get("y", None)
    if y is None:
        return None
    try:
        return int(y)
    except Exception:
        try:
            return int(y[0])
        except Exception:
            return None


def dst_key_for(data: HeteroData, dst_type: str, dst_local_id: int) -> str:
    keys = getattr(data[dst_type], "node_key", None)
    if keys is None or dst_local_id < 0 or dst_local_id >= len(keys):
        return f"{dst_type}#{dst_local_id}"
    return str(keys[dst_local_id])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", type=str, nargs="+", required=True, help="Graph dirs containing *.graph.pt")
    ap.add_argument("--out", type=str, default="artifacts/stats/benign_rarity_stats.json")
    ap.add_argument("--label", type=int, default=0, help="Label treated as benign (default: 0)")
    ap.add_argument("--limit", type=int, default=0, help="If >0, cap number of benign graphs")
    args = ap.parse_args()

    graph_paths: list[Path] = []
    for d in args.graphs:
        p = Path(d)
        if p.is_file() and p.name.endswith(".graph.pt"):
            graph_paths.append(p)
        elif p.is_dir():
            graph_paths.extend(sorted(p.glob("*.graph.pt")))
    if not graph_paths:
        raise SystemExit("No *.graph.pt found under --graphs")

    # df over graphs: count each (etype,dst_key) once per graph.
    df: Dict[Tuple[EType, str], int] = defaultdict(int)
    num = 0
    for gp in graph_paths:
        y = get_label(gp)
        if y is None or int(y) != int(args.label):
            continue
        data = load_graph(gp)

        seen: Set[Tuple[EType, str]] = set()
        for etype in data.edge_types:
            dst_type = etype[2]
            edge_index = data[etype].edge_index
            if edge_index.numel() == 0:
                continue
            dst = edge_index[1].tolist()
            for dst_local in dst:
                k = (etype, dst_key_for(data, dst_type, int(dst_local)))
                seen.add(k)
        for k in seen:
            df[k] += 1
        num += 1
        if args.limit > 0 and num >= int(args.limit):
            break

    stats = RarityStats(num_graphs=num, df=dict(df))
    save_rarity_stats(args.out, stats)
    print(f"out={args.out} benign_graphs={num} unique_pairs={len(stats.df)}")


if __name__ == "__main__":
    main()

