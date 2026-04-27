from __future__ import annotations

import argparse
import csv
import itertools
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Set

import numpy as np
import torch

# Ensure repo root importable when running as script.
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from scripts import eval_hunt_rarity as ehr  # noqa: E402


@dataclass(frozen=True)
class AblationRow:
    rarity_lambda: float
    rarity_idf_cap: float
    rarity_etypes: str
    rarity_normalize: str

    n_graphs: int
    auroc_base: float
    auprc_base: float
    auroc_rarity: float
    auprc_rarity: float
    p_at_k: dict[int, float]
    r_at_k: dict[int, float]
    seconds: float


def _read_list(path: str | None) -> list[str]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p.as_posix())
    return [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]


def _iter_graph_paths(graph_args: Sequence[str]) -> list[Path]:
    out: list[Path] = []
    for a in graph_args:
        p = Path(a)
        if p.is_file() and p.name.endswith(".graph.pt"):
            out.append(p)
        elif p.is_dir():
            out.extend(sorted(p.glob("*.graph.pt")))
    return out


def _as_int_list_csv(s: str) -> list[int]:
    return [int(x) for x in str(s).split(",") if x.strip()]


def _as_float_list_csv(s: str) -> list[float]:
    return [float(x) for x in str(s).split(",") if x.strip()]


def _parse_etypes_grid(s: str) -> list[Set[str]]:
    """
    Format:
      - configs separated by ';'
      - within each config, etypes separated by ','
      - an empty config means "all etypes" (i.e., empty set => no filtering)

    Example:
      "PROC|CONNECT|NET,PROC|EXEC|CMD;PROC|READ|FILE"
    """
    raw = str(s).strip()
    if not raw:
        return [set()]
    configs = [c.strip() for c in raw.split(";")]
    out: list[Set[str]] = []
    for c in configs:
        if not c:
            out.append(set())
            continue
        out.append({x.strip() for x in c.split(",") if x.strip()})
    return out or [set()]


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y_true, y_score))


def _safe_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y_true, y_score))


def _precision_recall_at_k(y_true: np.ndarray, y_score: np.ndarray, ks: Iterable[int]) -> tuple[dict[int, float], dict[int, float]]:
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    pos_total = int((y_true == 1).sum())
    p_at: dict[int, float] = {}
    r_at: dict[int, float] = {}
    for k in sorted(set(int(x) for x in ks if int(x) > 0)):
        kk = min(k, len(y_sorted))
        if kk <= 0 or pos_total <= 0:
            p_at[k] = float("nan")
            r_at[k] = float("nan")
            continue
        tp = int((y_sorted[:kk] == 1).sum())
        p_at[k] = tp / float(kk)
        r_at[k] = tp / float(pos_total)
    return p_at, r_at


def _set_seeds(seed: int) -> None:
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _filter_with_config(ranked: list[ehr.RankedEdge], cfg: ehr.HuntConfig) -> list[ehr.RankedEdge]:
    return ehr._filter_ranked_edges(
        ranked,
        filter_net=cfg.filter_net,
        filter_net_ip=cfg.filter_net_ip,
        filter_domains=cfg.filter_domains,
        filter_system_files=cfg.filter_system_files,
        system_file_prefixes=cfg.system_file_prefixes,
        filter_tmp_tempfile=cfg.filter_tmp_tempfile,
        filter_cmd_noise=cfg.filter_cmd_noise,
        cmd_noise_keywords=cfg.cmd_noise_keywords,
        dedup_dst=cfg.dedup_dst,
        max_per_etype=cfg.max_per_etype,
    )


@torch.no_grad()
def _score_graphs_once(
    *,
    graph_paths: Sequence[Path],
    backbone,
    explainer,
    device: torch.device,
    k: int,
    cfg: ehr.HuntConfig,
    rarity_normalize: str,
) -> tuple[list[int], list[list[ehr.RankedEdge]]]:
    """Compute and cache ranked edges for each graph once (model/explainer are constant)."""
    # We reuse ehr's filtering & topk to ensure identical behavior.
    ranked_all: list[list[ehr.RankedEdge]] = []
    y_true: list[int] = []

    for gp in graph_paths:
        y = ehr._get_y(gp)
        if y is None or int(y) not in (0, 1):
            continue
        data = ehr._load_graph(gp).to(device)

        ranked = ehr.topk_edges(
            backbone=backbone,
            explainer=explainer,
            hetero_batch=data,
            k=1_000_000,
        )
        ranked_all.append(ranked)
        y_true.append(int(y))

    _ = rarity_normalize
    return y_true, ranked_all


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", type=str, nargs="+", default=[], help="Graph dirs or files; ignored if --test-list is set")
    ap.add_argument("--test-list", type=str, default="", help="Optional: path to *.graph.pt list for evaluation")
    ap.add_argument("--backbone-ckpt", type=str, required=True)
    ap.add_argument("--explainer-ckpt", type=str, required=True)
    ap.add_argument("--device", type=str, default="cpu", help="cpu|cuda")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--k", type=int, default=20, help="Top-K suspicious edges used to score each graph")
    ap.add_argument("--ks", type=str, default="10,50,100", help="Graph-level P@K/R@K on rarity suspicious scores")

    # Mirror hunt filters (keep defaults identical to eval_hunt_rarity.py)
    ap.add_argument("--filter-net", action="store_true", default=False)
    ap.add_argument("--filter-net-ip", action="store_true", default=False)
    ap.add_argument("--filter-domains", type=str, default="")
    ap.add_argument("--filter-system-files", action="store_true", default=False)
    ap.add_argument("--system-file-prefixes", type=str, default="")
    ap.add_argument("--filter-tmp-tempfile", action="store_true", default=False)
    ap.add_argument("--filter-cmd-noise", action="store_true", default=False)
    ap.add_argument("--cmd-noise-keywords", type=str, default="")
    ap.add_argument("--dedup-dst", action="store_true", default=False)
    ap.add_argument("--max-per-etype", type=int, default=0)

    # Rarity grid
    ap.add_argument("--rarity-stats", type=str, required=True)
    ap.add_argument("--rarity-lambdas", type=str, default="0.0,0.1,0.3,0.5,1.0")
    ap.add_argument("--rarity-idf-caps", type=str, default="0.0,1.0,2.0,3.0,5.0")
    ap.add_argument(
        "--rarity-etypes-grid",
        type=str,
        default="PROC|CONNECT|NET,PROC|EXEC|CMD;PROC|CONNECT|NET;PROC|EXEC|CMD;",
        help="';' separated configs, each config is comma-separated etype strings. Empty config means all etypes.",
    )
    ap.add_argument("--rarity-normalize", type=str, default="qut")

    ap.add_argument("--out-csv", type=str, default="artifacts/rarity_ablation_qut.csv")
    ap.add_argument("--out-json", type=str, default="", help="Optional: write a machine-readable report JSON")
    args = ap.parse_args()

    _set_seeds(args.seed)

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    backbone = ehr._load_backbone(args.backbone_ckpt, device=device)
    explainer = ehr._load_explainer(args.explainer_ckpt, device=device)

    test_files = _read_list(args.test_list)
    graph_paths = [Path(x) for x in test_files] if test_files else _iter_graph_paths(args.graphs)
    if not graph_paths:
        raise SystemExit("No graphs found (provide --graphs or --test-list)")

    cfg = ehr.HuntConfig(
        k=int(args.k),
        filter_net=bool(args.filter_net),
        filter_net_ip=bool(args.filter_net_ip),
        filter_domains=[s.strip() for s in str(args.filter_domains).split(",") if s.strip()],
        filter_system_files=bool(args.filter_system_files),
        system_file_prefixes=[s.strip() for s in str(args.system_file_prefixes).split(",") if s.strip()],
        filter_tmp_tempfile=bool(args.filter_tmp_tempfile),
        filter_cmd_noise=bool(args.filter_cmd_noise),
        cmd_noise_keywords=[s.strip() for s in str(args.cmd_noise_keywords).split(",") if s.strip()],
        dedup_dst=bool(args.dedup_dst),
        max_per_etype=int(args.max_per_etype),
    )

    ks = _as_int_list_csv(args.ks)

    t0 = time.perf_counter()
    y_true_list, ranked_all = _score_graphs_once(
        graph_paths=graph_paths,
        backbone=backbone,
        explainer=explainer,
        device=device,
        k=int(args.k),
        cfg=cfg,
        rarity_normalize=str(args.rarity_normalize),
    )
    y_true = np.asarray(y_true_list, dtype=int)
    if len(y_true) == 0:
        raise SystemExit("No labeled graphs found (expected y in each *.graph.pt)")

    rarity = ehr.load_rarity_stats(str(args.rarity_stats))

    # Baseline suspicious score: low explainer keep scores are more suspicious.
    base_ranked_all = [_filter_with_config(ehr._to_suspicious_edges(r), cfg) for r in ranked_all]
    base_scores = np.asarray([ehr._suspicious_score_from_ranked(r, k=int(args.k)) for r in base_ranked_all], dtype=float)
    auroc_base = _safe_auc(y_true, base_scores)
    auprc_base = _safe_auprc(y_true, base_scores)

    lambdas = _as_float_list_csv(args.rarity_lambdas)
    caps = _as_float_list_csv(args.rarity_idf_caps)
    etypes_grid = _parse_etypes_grid(args.rarity_etypes_grid)

    rows: list[AblationRow] = []
    for lam, cap, etypes in itertools.product(lambdas, caps, etypes_grid):
        t1 = time.perf_counter()
        ranked_adj = [
            _filter_with_config(
                ehr._to_suspicious_edges(
                    r,
                    rarity=rarity,
                    rarity_lambda=float(lam),
                    rarity_idf_cap=float(cap),
                    rarity_etypes=set(etypes),
                    rarity_normalize=str(args.rarity_normalize),
                ),
                cfg,
            )
            for r in ranked_all
        ]
        rarity_scores = np.asarray([ehr._suspicious_score_from_ranked(r, k=int(args.k)) for r in ranked_adj], dtype=float)
        auroc_r = _safe_auc(y_true, rarity_scores)
        auprc_r = _safe_auprc(y_true, rarity_scores)
        p_at, r_at = _precision_recall_at_k(y_true, rarity_scores, ks)
        rows.append(
            AblationRow(
                rarity_lambda=float(lam),
                rarity_idf_cap=float(cap),
                rarity_etypes=",".join(sorted(etypes)),
                rarity_normalize=str(args.rarity_normalize),
                n_graphs=int(len(y_true)),
                auroc_base=float(auroc_base),
                auprc_base=float(auprc_base),
                auroc_rarity=float(auroc_r),
                auprc_rarity=float(auprc_r),
                p_at_k=p_at,
                r_at_k=r_at,
                seconds=float(time.perf_counter() - t1),
            )
        )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        header = [
            "rarity_lambda",
            "rarity_idf_cap",
            "rarity_etypes",
            "rarity_normalize",
            "n_graphs",
            "auroc_base",
            "auprc_base",
            "auroc_rarity",
            "auprc_rarity",
        ]
        for k in ks:
            header.append(f"p_at_{k}")
            header.append(f"r_at_{k}")
        header.append("seconds")
        w.writerow(header)

        for r in rows:
            row = [
                r.rarity_lambda,
                r.rarity_idf_cap,
                r.rarity_etypes,
                r.rarity_normalize,
                r.n_graphs,
                r.auroc_base,
                r.auprc_base,
                r.auroc_rarity,
                r.auprc_rarity,
            ]
            for k in ks:
                row.append(r.p_at_k.get(int(k), float("nan")))
                row.append(r.r_at_k.get(int(k), float("nan")))
            row.append(r.seconds)
            w.writerow(row)

    report = {
        "graphs": [p.as_posix() for p in graph_paths],
        "n_graphs_used": int(len(y_true)),
        "device": str(device),
        "k": int(args.k),
        "ks": ks,
        "backbone_ckpt": str(args.backbone_ckpt),
        "explainer_ckpt": str(args.explainer_ckpt),
        "rarity_stats": str(args.rarity_stats),
        "rarity_lambdas": lambdas,
        "rarity_idf_caps": caps,
        "rarity_etypes_grid": [sorted(list(s)) for s in etypes_grid],
        "rarity_normalize": str(args.rarity_normalize),
        "auroc_base": float(auroc_base),
        "auprc_base": float(auprc_base),
        "out_csv": out_csv.as_posix(),
        "seconds_total": float(time.perf_counter() - t0),
    }
    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2))

    print(f"wrote_csv={out_csv.as_posix()} rows={len(rows)} graphs_used={len(y_true)} seconds_total={report['seconds_total']:.2f}")


if __name__ == "__main__":
    main()

