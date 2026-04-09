from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Tuple

import torch
from torch_geometric.data import HeteroData


REL_COLS = ["LOAD", "DEPEND", "INVOKE", "CONNECT", "READ", "WRITE", "EXEC", "DNS_QUERY", "RESOLVE", "DELETE"]


def iter_graph_files(dirs: List[str]) -> List[Path]:
    paths: List[Path] = []
    for d in dirs:
        p = Path(d)
        if p.is_file() and p.suffix == ".pt":
            paths.append(p)
        else:
            paths.extend(sorted(p.glob("*.graph.pt")))
    return paths


def rel_counts(data: HeteroData) -> Counter:
    c = Counter()
    for (src, rel, dst) in data.edge_types:
        e = data[(src, rel, dst)].edge_index
        c[rel] += int(e.size(1))
    return c


def total_nodes(data: HeteroData) -> int:
    return sum(int(data[nt].num_nodes) for nt in data.node_types)


def total_edges(data: HeteroData) -> int:
    return sum(int(data[et].edge_index.size(1)) for et in data.edge_types)


def load_graph(path: Path) -> Tuple[HeteroData, Dict]:
    obj = torch.load(path, map_location="cpu")
    data = HeteroData.from_dict(obj["data_dict"])
    meta = {
        "path": path,
        "package": obj.get("package", path.stem),
        "y": obj.get("y", None),
        "source": "qut" if "qut" in path.as_posix() else "osptrack" if "osp" in path.as_posix() else "unknown",
    }
    return data, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", type=str, nargs="+", required=True, help="One or more dirs containing *.graph.pt")
    ap.add_argument("--out", type=str, default="artifacts/stats")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    paths = iter_graph_files(args.graphs)
    if not paths:
        raise FileNotFoundError(f"No *.graph.pt found under: {args.graphs}")

    rows: List[Dict] = []
    summary = defaultdict(list)
    rel_presence = defaultdict(int)

    for p in paths:
        data, meta = load_graph(p)
        rc = rel_counts(data)
        e_total = total_edges(data)
        e_load = int(rc.get("LOAD", 0))
        non_load_ratio = (e_total - e_load) / max(1, e_total)

        row = {
            "id": p.stem.replace(".graph", ""),
            "source": meta["source"],
            "y": meta["y"] if meta["y"] is not None else "",
            "path": p.as_posix(),
            "num_nodes_total": total_nodes(data),
            "num_edges_total": e_total,
            "num_node_types": len(data.node_types),
            "num_edge_types": len(data.edge_types),
            "non_load_ratio": non_load_ratio,
        }

        for rel in REL_COLS:
            row[f"E_{rel}"] = int(rc.get(rel, 0))
            row[f"has_{rel}"] = 1 if rc.get(rel, 0) > 0 else 0
            if rc.get(rel, 0) > 0:
                rel_presence[(meta["source"], rel)] += 1

        rows.append(row)
        summary[(meta["source"], "edges_total")].append(e_total)
        summary[(meta["source"], "non_load_ratio")].append(non_load_ratio)
        summary[(meta["source"], "only_load")].append(1 if non_load_ratio == 0 else 0)

    # Write per-graph CSV
    per_path = outdir / "stats_per_graph.csv"
    fieldnames = list(rows[0].keys())
    with per_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # Write summary markdown
    md = []
    sources = sorted(set(r["source"] for r in rows))
    md.append("## Canonical graph stats summary\n")
    for src in sources:
        n = sum(1 for r in rows if r["source"] == src)
        edges = summary[(src, "edges_total")]
        nlr = summary[(src, "non_load_ratio")]
        only = summary[(src, "only_load")]
        md.append(f"### {src}\n")
        md.append(f"- graphs: **{n}**\n")
        md.append(
            f"- edges_total: mean={sum(edges)/max(1,len(edges)):.2f}, median={median(edges):.2f}, max={max(edges)}\n"
        )
        md.append(
            f"- non_load_ratio: mean={sum(nlr)/max(1,len(nlr)):.3f}, median={median(nlr):.3f}\n"
        )
        md.append(f"- only_load_ratio: **{sum(only)/max(1,len(only)):.3f}**\n")
        md.append("- relation coverage (fraction of graphs with ≥1 edge):\n")
        for rel in REL_COLS:
            cov = rel_presence.get((src, rel), 0) / max(1, n)
            md.append(f"  - {rel}: {cov:.3f}\n")
        md.append("\n")

    (outdir / "stats_summary.md").write_text("".join(md))
    print(f"Wrote: {per_path}")
    print(f"Wrote: {outdir / 'stats_summary.md'}")


if __name__ == "__main__":
    main()

