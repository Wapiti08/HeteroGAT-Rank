from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


BUCKET_ORDER = ["Load-only", "Low-observability", "Medium-observability", "High-observability"]


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _safe_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def _precision_recall_at_k(y_true: np.ndarray, y_score: np.ndarray, ks: Iterable[int]) -> dict[str, float]:
    out: dict[str, float] = {}
    if y_true.size == 0:
        for k in ks:
            out[f"P@{int(k)}"] = float("nan")
            out[f"R@{int(k)}"] = float("nan")
        return out

    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    pos_total = int((y_true == 1).sum())
    for k in ks:
        kk = min(int(k), len(y_sorted))
        if kk <= 0 or pos_total <= 0:
            out[f"P@{int(k)}"] = float("nan")
            out[f"R@{int(k)}"] = float("nan")
            continue
        tp = int((y_sorted[:kk] == 1).sum())
        out[f"P@{int(k)}"] = tp / float(kk)
        out[f"R@{int(k)}"] = tp / float(pos_total)
    return out


def _fmt(x: float) -> str:
    if np.isnan(float(x)):
        return "nan"
    return f"{float(x):.6g}"


def _method_names(rows: list[dict]) -> list[str]:
    seen: list[str] = []
    for row in rows:
        for name in row.get("scores", {}).keys():
            if name not in seen:
                seen.append(name)
    return seen


def _quantile_bucket_rows(rows: list[dict], *, field: str) -> dict[str, list[dict]]:
    load_only: list[dict] = []
    observable: list[dict] = []
    for row in rows:
        value = float(row.get(field, 0.0))
        if float(row.get("behavior_edges", 0.0)) <= 0.0:
            load_only.append(row)
        else:
            observable.append(row)

    grouped: dict[str, list[dict]] = defaultdict(list)
    if load_only:
        grouped["Load-only"].extend(load_only)
    if not observable:
        return grouped

    values = np.asarray([float(r.get(field, 0.0)) for r in observable], dtype=float)
    q1, q2 = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
    for row, value in zip(observable, values):
        if value <= q1:
            grouped["Low-observability"].append(row)
        elif value <= q2:
            grouped["Medium-observability"].append(row)
        else:
            grouped["High-observability"].append(row)
    return grouped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True, help="JSON produced by eval_explainer_baselines.py")
    ap.add_argument("--method", type=str, default="HetHunt + filtering + rarity")
    ap.add_argument("--ks", type=str, default="", help="Override K list; default reads config.ks from JSON")
    ap.add_argument(
        "--bucket-mode",
        choices=["stored", "quantile"],
        default="stored",
        help="stored uses eval-time fixed buckets; quantile re-buckets observable graphs into tertiles.",
    )
    ap.add_argument(
        "--bucket-field",
        type=str,
        default="behavior_edges",
        help="Per-graph numeric field used with --bucket-mode quantile.",
    )
    ap.add_argument("--out", type=str, default="", help="Optional TSV output path")
    args = ap.parse_args()

    obj = json.loads(Path(args.input).read_text())
    rows = list(obj.get("per_graph", []))
    if not rows:
        raise SystemExit("No per_graph rows found. Re-run eval_explainer_baselines.py with the updated script.")

    methods = _method_names(rows)
    if args.method not in methods:
        raise SystemExit(f"Unknown method {args.method!r}. Available: {', '.join(methods)}")

    if str(args.ks).strip():
        ks = [int(x) for x in str(args.ks).split(",") if x.strip()]
    else:
        ks = [int(x) for x in obj.get("config", {}).get("ks", [10, 50, 100])]

    if args.bucket_mode == "quantile":
        grouped = _quantile_bucket_rows(rows, field=str(args.bucket_field))
    else:
        grouped: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            grouped[str(row.get("observability_bucket", "Unknown"))].append(row)

    header = [
        "bucket",
        "n",
        "pos",
        "neg",
        "min_bucket_field",
        "max_bucket_field",
        "avg_behavior_edges",
        "avg_family_richness",
        "avg_edge_type_richness",
        "auroc",
        "auprc",
    ]
    for k in ks:
        header.extend([f"P@{k}", f"R@{k}"])

    out_lines = ["\t".join(header)]
    ordered_buckets = [b for b in BUCKET_ORDER if b in grouped] + sorted(b for b in grouped if b not in BUCKET_ORDER)
    for bucket in ordered_buckets:
        bucket_rows = grouped[bucket]
        y = np.asarray([int(r["y"]) for r in bucket_rows], dtype=int)
        score = np.asarray([float(r.get("scores", {}).get(args.method, float("nan"))) for r in bucket_rows], dtype=float)
        finite = np.isfinite(score)
        y_eval = y[finite]
        score_eval = score[finite]

        behavior_edges = np.asarray([float(r.get("behavior_edges", 0.0)) for r in bucket_rows], dtype=float)
        family_richness = np.asarray([float(r.get("behavior_family_richness", 0.0)) for r in bucket_rows], dtype=float)
        edge_type_richness = np.asarray([float(r.get("edge_type_richness", 0.0)) for r in bucket_rows], dtype=float)
        bucket_field = np.asarray([float(r.get(args.bucket_field, 0.0)) for r in bucket_rows], dtype=float)

        pr = _precision_recall_at_k(y_eval, score_eval, ks)
        line = [
            bucket,
            str(len(bucket_rows)),
            str(int((y == 1).sum())),
            str(int((y == 0).sum())),
            _fmt(float(bucket_field.min()) if bucket_field.size else float("nan")),
            _fmt(float(bucket_field.max()) if bucket_field.size else float("nan")),
            _fmt(float(behavior_edges.mean()) if behavior_edges.size else float("nan")),
            _fmt(float(family_richness.mean()) if family_richness.size else float("nan")),
            _fmt(float(edge_type_richness.mean()) if edge_type_richness.size else float("nan")),
            _fmt(_safe_auc(y_eval, score_eval)),
            _fmt(_safe_auprc(y_eval, score_eval)),
        ]
        for k in ks:
            line.extend([_fmt(pr[f"P@{k}"]), _fmt(pr[f"R@{k}"])])
        out_lines.append("\t".join(line))

    text = "\n".join(out_lines) + "\n"
    print(text, end="")
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text)
        print(f"wrote_tsv={out_path.as_posix()}")


if __name__ == "__main__":
    main()
