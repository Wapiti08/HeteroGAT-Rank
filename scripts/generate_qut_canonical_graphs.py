from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from typing import Any, Dict, List, Optional, Tuple

# Ensure repo root is importable when running as a script.
REPO_ROOT = Path(__file__).resolve().parent.parent
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

import pandas as pd

from config.qut_canonical import QUTKind, extract_events_from_row


QUT_BASE = Path("data/QUT-DV25_Datasets/QUT-DV25_Processed_Datasets")
QUT_FILES: Dict[QUTKind, Path] = {
    "install": QUT_BASE / "QUT-DV25_Install_Traces/QUT-DV25_Install_Traces.csv",
    "syscall": QUT_BASE / "QUT-DV25_SysCall_Traces/QUT-DV25_SysCall_Traces.csv",
    "filetop": QUT_BASE / "QUT-DV25_Filetop_Traces/QUT-DV25_Filetop_Traces.csv",
    "opensnoop": QUT_BASE / "QUT-DV25_Opensnoop_Traces/QUT-DV25_Opensnoop_Traces.csv",
    "tcp": QUT_BASE / "QUT-DV25_TCP_Traces/QUT-DV25_TCP_Traces.csv",
    "pattern": QUT_BASE / "QUT-DV25_Pattern_Traces/QUT-DV25_Pattern_Traces.csv",
}


def _load_table(path: Path, *, usecols: Optional[List[str]] = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path.as_posix())
    return pd.read_csv(path, usecols=usecols)


def _events_to_jsonable(events) -> Dict[str, Any]:
    # events are dataclasses from config.osptrack_canonical
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


def build_joined_events_for_package(
    pkg_name: str,
    tables: Dict[QUTKind, pd.DataFrame],
    kinds: List[QUTKind],
) -> List:
    events: List = []
    for kind in kinds:
        df = tables[kind]
        sub = df[df["Package_Name"] == pkg_name]
        if sub.empty:
            continue
        row = sub.iloc[0].to_dict()
        events.extend(extract_events_from_row(row, kind))
    return events


def get_qut_label(pkg_name: str, tables: Dict[QUTKind, pd.DataFrame], kinds: List[QUTKind]) -> Optional[int]:
    """Fetch the binary label from QUT `Level` column (0/1).

    `Level` is consistent across all processed tables, so we can take it from the
    first available table.
    """
    for kind in kinds:
        df = tables[kind]
        if "Level" not in df.columns:
            continue
        sub = df[df["Package_Name"] == pkg_name]
        if sub.empty:
            continue
        v = sub.iloc[0].get("Level")
        try:
            return int(v)
        except Exception:
            return None
    return None

def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _atomic_torch_save(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    import torch  # type: ignore

    torch.save(obj, tmp)
    tmp.replace(path)


def _process_one_package(payload: Dict[str, Any]) -> Tuple[str, int]:
    # Imports inside worker.
    from config.qut_canonical import extract_events_from_row as _extract

    build_heterodata_from_events = None
    if payload.get("have_graph", False):
        try:
            from graph import build_heterodata_from_events as _build  # noqa: WPS433

            build_heterodata_from_events = _build
        except Exception:
            build_heterodata_from_events = None

    pkg = str(payload["pkg"])
    kinds: List[QUTKind] = payload["kinds"]
    rows_by_kind: Dict[QUTKind, Dict[str, Any]] = payload["rows_by_kind"]
    y = payload.get("y", None)
    outdir = Path(payload["outdir"])

    events: List = []
    for kind in kinds:
        row = rows_by_kind.get(kind)
        if not row:
            continue
        events.extend(_extract(row, kind))

    obj = _events_to_jsonable(events)
    if y is not None:
        obj["y"] = int(y)
    _atomic_write_text(outdir / f"{pkg}.events.json", json.dumps(obj, ensure_ascii=False))

    if build_heterodata_from_events is not None:
        data = build_heterodata_from_events(events, y=y)
        safe = {"data_dict": data.to_dict(), "package": pkg, "kinds": kinds, "num_events": len(events), "y": y}
        _atomic_torch_save(outdir / f"{pkg}.graph.pt", safe)
    return pkg, len(events)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="artifacts/qut_canonical")
    ap.add_argument("--package-name", type=str, default="")
    ap.add_argument("--limit-packages", type=int, default=50)
    ap.add_argument(
        "--kinds",
        type=str,
        default="install,syscall,filetop,opensnoop,tcp,pattern",
        help="Comma-separated subset of: install,syscall,filetop,opensnoop,tcp,pattern",
    )
    ap.add_argument("--workers", type=int, default=0, help="Parallel workers (0=auto, 1=disable)")
    args = ap.parse_args()

    kinds: List[QUTKind] = [k.strip() for k in args.kinds.split(",") if k.strip()]  # type: ignore[assignment]
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load only once; these are moderate-sized CSVs.
    tables: Dict[QUTKind, pd.DataFrame] = {k: _load_table(QUT_FILES[k]) for k in kinds}

    if args.package_name:
        pkg_names = [args.package_name]
    else:
        # Use the install table as the package universe if available, otherwise first table.
        base_kind = "install" if "install" in tables else kinds[0]
        pkg_names = tables[base_kind]["Package_Name"].dropna().astype(str).unique().tolist()[: args.limit_packages]

    # Try optional graph build; always save JSON.
    have_graph = False
    build_heterodata_from_events = None
    try:
        from graph import build_heterodata_from_events as _build  # noqa: WPS433

        build_heterodata_from_events = _build
        have_graph = True
    except Exception:
        have_graph = False

    payloads: List[Dict[str, Any]] = []

    for pkg in pkg_names:
        # Prepare compact per-package payload (avoid passing whole DataFrames to workers).
        rows_by_kind: Dict[QUTKind, Dict[str, Any]] = {}
        for kind in kinds:
            df = tables[kind]
            sub = df[df["Package_Name"] == pkg]
            if sub.empty:
                continue
            rows_by_kind[kind] = sub.iloc[0].to_dict()
        y = get_qut_label(pkg, tables, kinds)
        payloads.append(
            {
                "pkg": pkg,
                "kinds": kinds,
                "rows_by_kind": rows_by_kind,
                "y": y,
                "outdir": outdir.as_posix(),
                "have_graph": have_graph,
            }
        )

    workers = args.workers
    if workers <= 0:
        workers = max(1, (os.cpu_count() or 2) // 2)

    if workers <= 1:
        for pl in payloads:
            _process_one_package(pl)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_process_one_package, pl) for pl in payloads]
            for fut in as_completed(futs):
                _ = fut.result()


if __name__ == "__main__":
    main()

