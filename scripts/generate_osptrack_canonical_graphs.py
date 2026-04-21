from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
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


def iter_pkl_rows(path: Path, *, limit: Optional[int] = None, random_sample: bool = False, seed: int = 42) -> Iterator[Dict[str, Any]]:
    """Iterate rows from `label_data.pkl`.

    Note: `pd.read_pickle` loads the full DataFrame into memory. This is fine for
    small-scale tests (e.g., limit=200/2000). For full-scale runs, consider
    creating a smaller subset pickle first.
    """
    df = pd.read_pickle(path)
    if limit is not None:
        if random_sample and limit < len(df):
            df = df.sample(n=limit, random_state=seed)
        else:
            df = df.head(limit)
    for _, row in df.iterrows():
        yield row.to_dict()


def iter_pkl_rows_stratified(
    path: Path,
    *,
    per_class: int,
    seed: int = 42,
    label_col: str = "Label",
) -> Iterator[Dict[str, Any]]:
    """Stratified sampling from pkl to ensure both labels are present."""
    df = pd.read_pickle(path)
    if label_col not in df.columns:
        raise KeyError(f"Missing column {label_col} in {path}")

    # Normalize label column to int {0,1} if possible.
    labels = df[label_col]
    df0 = df[labels == 0]
    df1 = df[labels == 1]
    if len(df0) == 0 or len(df1) == 0:
        raise ValueError(f"Cannot stratify: label distribution is degenerate (0={len(df0)}, 1={len(df1)})")

    n0 = min(per_class, len(df0))
    n1 = min(per_class, len(df1))
    s0 = df0.sample(n=n0, random_state=seed)
    s1 = df1.sample(n=n1, random_state=seed)
    out = pd.concat([s0, s1]).sample(frac=1.0, random_state=seed)  # shuffle

    for _, row in out.iterrows():
        yield row.to_dict()


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

def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _atomic_torch_save(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    import torch  # type: ignore

    torch.save(obj, tmp)
    tmp.replace(path)


def _process_one_row(row: Dict[str, Any], outdir: str, have_graph: bool) -> Tuple[str, int]:
    # Import inside worker for multiprocessing compatibility.
    from config.osptrack_canonical import extract_events as _extract_events, pkg_key as _pkg_key

    build_heterodata_from_events = None
    if have_graph:
        try:
            from graph import build_heterodata_from_events as _build  # noqa: WPS433

            build_heterodata_from_events = _build
        except Exception:
            build_heterodata_from_events = None

    outdir_p = Path(outdir)
    events = _extract_events(row)
    key = _pkg_key(row).replace("/", "_")
    obj = events_to_jsonable(events)
    obj["label"] = int(row.get("Label", 0)) if str(row.get("Label", "")).strip() != "" else None
    obj["sub_label"] = row.get("Sub_Label")
    _atomic_write_text(outdir_p / f"{key}.events.json", json.dumps(obj, ensure_ascii=False))

    if build_heterodata_from_events is not None:
        y = int(row.get("Label", 0))
        data = build_heterodata_from_events(events, y=y)
        safe = {"data_dict": data.to_dict(), "package": key, "num_events": len(events), "y": y}
        _atomic_torch_save(outdir_p / f"{key}.graph.pt", safe)
    return key, len(events)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", type=str, default="data/OSPTrack/label_data.pkl")
    ap.add_argument("--csv", type=str, default="data/OSPTrack/label_data.csv")
    ap.add_argument("--prefer-pkl", action="store_true", default=True)
    ap.add_argument("--no-prefer-pkl", action="store_false", dest="prefer_pkl")
    ap.add_argument("--out", type=str, default="artifacts/osptrack_canonical")
    ap.add_argument("--limit-rows", type=int, default=1000, help="Max rows to process (0 = all rows)")
    ap.add_argument("--chunksize", type=int, default=50_000)
    ap.add_argument("--random-sample", action="store_true", default=False)
    ap.add_argument("--stratify-label", action="store_true", default=False)
    ap.add_argument("--per-class", type=int, default=0, help="If --stratify-label, sample this many per label")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=0, help="Parallel workers (0=auto, 1=disable)")
    args = ap.parse_args()

    pkl_path = Path(args.pkl)
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

    use_pkl = args.prefer_pkl and pkl_path.exists()
    limit_rows = None if args.limit_rows <= 0 else int(args.limit_rows)
    if use_pkl:
        if args.stratify_label:
            if args.per_class <= 0:
                raise ValueError("--per-class must be > 0 when --stratify-label is set")
            row_iter = iter_pkl_rows_stratified(pkl_path, per_class=args.per_class, seed=args.seed)
        else:
            row_iter = iter_pkl_rows(pkl_path, limit=limit_rows, random_sample=args.random_sample, seed=args.seed)
    else:
        if args.stratify_label:
            raise ValueError("--stratify-label requires --prefer-pkl (CSV streaming stratification is unsupported)")
        row_iter = iter_csv_rows(csv_path, chunksize=args.chunksize, limit=limit_rows)

    workers = args.workers
    if workers <= 0:
        # conservative default to avoid memory spikes with pandas row dicts
        workers = max(1, (os.cpu_count() or 2) // 2)

    if workers <= 1:
        for row in row_iter:
            _process_one_row(row, outdir.as_posix(), have_graph)
    else:
        rows = list(row_iter)
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_process_one_row, row, outdir.as_posix(), have_graph) for row in rows]
            for fut in as_completed(futs):
                _ = fut.result()


if __name__ == "__main__":
    main()

