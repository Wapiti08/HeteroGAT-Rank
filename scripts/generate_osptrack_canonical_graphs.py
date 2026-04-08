from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pandas as pd

# Ensure repo root is importable when running as a script.
REPO_ROOT = Path(__file__).resolve().parent.parent
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from config.osptrack_canonical import extract_events, pkg_key  # noqa: E402


def iter_csv_rows(path: Path, *, chunksize: int = 50_000, limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    seen = 0
    for chunk in pd.read_csv(path, chunksize=chunksize):
        for _, row in chunk.iterrows():
            yield row.to_dict()
            seen += 1
            if limit is not None and seen >= limit:
                return


def events_to_jsonable(events) -> Dict[str, Any]:
    edges: List[Dict[str, Any]] = []
    nodes: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for e in events:
        nodes[(e.src.type, e.src.key)] = {"type": e.src.type, "key": e.src.key, "attrs": e.src.attrs}
        nodes[(e.dst.type, e.dst.key)] = {"type": e.dst.type, "key": e.dst.key, "attrs": e.dst.attrs}
        edges.append(
            {
                "src_type": e.src.type,
                "src_key": e.src.key,
                "etype": e.etype,
                "dst_type": e.dst.type,
                "dst_key": e.dst.key,
                "attrs": e.attrs,
            }
        )
    return {"nodes": list(nodes.values()), "edges": edges}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="data/label_data.csv")
    ap.add_argument("--out", type=str, default="artifacts/osptrack_canonical")
    ap.add_argument("--limit-rows", type=int, default=1000)
    ap.add_argument("--chunksize", type=int, default=50_000)
    args = ap.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # Optional graph build
    have_graph = False
    build_heterodata_from_events = None
    try:
        from graph import build_heterodata_from_events as _build  # noqa: WPS433

        build_heterodata_from_events = _build
        have_graph = True
    except Exception:
        have_graph = False

    for row in iter_csv_rows(csv_path, chunksize=args.chunksize, limit=args.limit_rows):
        events = extract_events(row)
        key = pkg_key(row).replace("/", "_")
        obj = events_to_jsonable(events)
        obj["label"] = int(row.get("Label", 0)) if str(row.get("Label", "")).strip() != "" else None
        obj["sub_label"] = row.get("Sub_Label")
        (outdir / f"{key}.events.json").write_text(json.dumps(obj, ensure_ascii=False))

        if have_graph and build_heterodata_from_events is not None:
            try:
                y = int(row.get("Label", 0))
                data = build_heterodata_from_events(events, y=y)
                safe = {"data_dict": data.to_dict(), "package": key, "num_events": len(events), "y": y}
                import torch  # type: ignore

                torch.save(safe, outdir / f"{key}.graph.pt")
            except ModuleNotFoundError:
                pass


if __name__ == "__main__":
    main()

