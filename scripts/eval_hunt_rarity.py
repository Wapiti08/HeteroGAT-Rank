from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from torch_geometric.data import HeteroData

# Ensure repo root importable when running as script.
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from model.rgcn import RGCNGraphClassifier  # noqa: E402
from ranking_explain.hunt import RankedEdge, topk_edges  # noqa: E402
from ranking_explain.pgexplainer import PGExplainer  # noqa: E402
from ranking_explain.rarity import RarityStats, etype_to_str, load_rarity_stats, normalize_dst_key  # noqa: E402
from ranking_explain.run_hunt import _filter_ranked_edges  # noqa: E402

EType = Tuple[str, str, str]


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


def _load_graph(path: Path) -> HeteroData:
    obj = torch.load(path, map_location="cpu")
    return HeteroData.from_dict(obj["data_dict"])


def _get_y(path: Path) -> Optional[int]:
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


def _load_backbone(path: str, *, device: torch.device) -> RGCNGraphClassifier:
    ckpt = torch.load(path, map_location=device)
    kwargs = ckpt.get("model_kwargs", {"hidden_dim": 64, "num_layers": 2, "dropout": 0.2, "num_classes": 2})
    model = RGCNGraphClassifier(**kwargs).to(device)
    schema = ckpt.get("schema", {})
    nnt = int(schema.get("num_node_types", 0))
    nr = int(schema.get("num_relations", 0))
    if nnt > 0 and nr > 0:
        model.materialize(num_node_types=nnt, num_relations=nr, device=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def _load_explainer(path: str, *, device: torch.device) -> PGExplainer:
    ckpt = torch.load(path, map_location=device)
    kwargs = ckpt.get("explainer_kwargs", {"hidden_dim": 64})
    explainer = PGExplainer(**kwargs).to(device)
    explainer.load_state_dict(ckpt["state_dict"])
    explainer.eval()
    return explainer


@dataclass(frozen=True)
class HuntConfig:
    k: int
    filter_net: bool
    filter_net_ip: bool
    filter_domains: list[str]
    filter_system_files: bool
    system_file_prefixes: list[str]
    filter_tmp_tempfile: bool
    filter_cmd_noise: bool
    cmd_noise_keywords: list[str]
    dedup_dst: bool
    max_per_etype: int


def _mean_topk_score(ranked: list[RankedEdge], *, k: int) -> float:
    if not ranked:
        return float("nan")
    kk = min(int(k), len(ranked))
    if kk <= 0:
        return float("nan")
    return float(sum(r.score for r in ranked[:kk]) / float(kk))


def _rarity_bonus(
    r: RankedEdge,
    *,
    rarity: Optional[RarityStats],
    rarity_lambda: float,
    rarity_idf_cap: float,
    rarity_etypes: Set[str],
    rarity_normalize: str,
) -> float:
    if rarity is None or float(rarity_lambda) == 0.0:
        return 0.0
    if rarity_etypes and etype_to_str(r.etype) not in rarity_etypes:
        return 0.0

    scheme = rarity_normalize.strip() or str(getattr(rarity, "normalize", "none"))
    norm_key = normalize_dst_key(scheme=scheme, etype=r.etype, dst_type=r.dst_type, dst_key=str(r.dst_key))
    idf = float(rarity.idf(etype=r.etype, dst_key=norm_key))
    if float(rarity_idf_cap) > 0.0:
        idf = min(idf, float(rarity_idf_cap))
    return float(rarity_lambda) * idf


def _to_suspicious_edges(
    ranked: list[RankedEdge],
    *,
    rarity: Optional[RarityStats] = None,
    rarity_lambda: float = 0.0,
    rarity_idf_cap: float = 0.0,
    rarity_etypes: Optional[Set[str]] = None,
    rarity_normalize: str = "",
) -> list[RankedEdge]:
    """Convert explainer keep scores into suspicious scores.

    PGExplainer edge scores are treated as normality/keep scores. Lower keep
    scores are therefore more suspicious, while benign-IDF rarity increases
    suspiciousness.
    """
    etypes = rarity_etypes or set()
    out: list[RankedEdge] = []
    for r in ranked:
        bonus = _rarity_bonus(
            r,
            rarity=rarity,
            rarity_lambda=rarity_lambda,
            rarity_idf_cap=rarity_idf_cap,
            rarity_etypes=etypes,
            rarity_normalize=rarity_normalize,
        )
        out.append(
            RankedEdge(
                graph_id=r.graph_id,
                score=float(-r.score + bonus),
                etype=r.etype,
                src_type=r.src_type,
                src_key=r.src_key,
                dst_type=r.dst_type,
                dst_key=r.dst_key,
            )
        )
    out.sort(key=lambda x: x.score, reverse=True)
    return out


def _suspicious_score_from_ranked(ranked: list[RankedEdge], *, k: int) -> float:
    return _mean_topk_score(ranked, k=k)


def _as_set_csv(s: str) -> Set[str]:
    return {x.strip() for x in str(s).split(",") if x.strip()}


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _safe_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def _precision_recall_at_k(y_true: np.ndarray, y_score: np.ndarray, ks: Iterable[int]) -> list[tuple[int, float, float]]:
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    pos_total = int((y_true == 1).sum())
    out: list[tuple[int, float, float]] = []
    for k in sorted(set(int(x) for x in ks if int(x) > 0)):
        kk = min(k, len(y_sorted))
        if kk <= 0 or pos_total <= 0:
            out.append((k, float("nan"), float("nan")))
            continue
        tp = int((y_sorted[:kk] == 1).sum())
        out.append((k, tp / float(kk), tp / float(pos_total)))
    return out


def _safe_div(num: float, den: float) -> float:
    if den == 0.0:
        return float("nan")
    return float(num / den)


def _binary_metrics_at_threshold(y_true: np.ndarray, y_score: np.ndarray, *, thr: float) -> dict[str, float]:
    """Compute alerting-style metrics at a fixed suspicious-score threshold.

    Convention: higher score => more suspicious => predicted positive.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    # Treat NaN/Inf scores as "no alert" (pred negative).
    finite = np.isfinite(y_score)
    y_pred = np.zeros_like(y_true, dtype=int)
    y_pred[finite] = (y_score[finite] >= float(thr)).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    precision = _safe_div(float(tp), float(tp + fp))
    recall = _safe_div(float(tp), float(tp + fn))
    f1 = float("nan")
    if not np.isnan(precision) and not np.isnan(recall):
        if (precision + recall) > 0.0:
            f1 = float(2.0 * precision * recall / (precision + recall))
        else:
            # Define F1 as 0 when both precision and recall are 0.
            f1 = 0.0
    fpr = _safe_div(float(fp), float(fp + tn))

    return {
        "thr": float(thr),
        "prec": float(precision),
        "rec": float(recall),
        "f1": float(f1),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "fpr": float(fpr),
    }


def _threshold_at_fpr(y_true: np.ndarray, y_score: np.ndarray, *, target_fpr: float) -> float:
    """Pick a threshold on validation scores such that FPR ≲ target_fpr.

    We use a robust quantile-based selector on NEGATIVE (benign) scores:
    choose thr as the (1 - target_fpr) quantile of y_score[y_true==0].

    This avoids roc_curve() edge cases (inf threshold, ties) on large/noisy datasets.
    """
    target = float(target_fpr)
    target = max(0.0, min(1.0, target))

    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    # Filter NaNs/Infs defensively.
    m = np.isfinite(y_score) & np.isfinite(y_true.astype(float))
    y_true = y_true[m]
    y_score = y_score[m]

    neg = y_score[y_true == 0]
    if neg.size == 0:
        return float("inf")

    q = 1.0 - target
    # For target_fpr==0, q==1.0 -> max(neg). That's the most conservative finite threshold.
    thr = float(np.quantile(neg, q))

    # Make threshold slightly above quantile to bias toward <= target FPR under ties.
    return float(np.nextafter(thr, float("inf")))


def _filter_with_config(ranked: list[RankedEdge], cfg: HuntConfig) -> list[RankedEdge]:
    return _filter_ranked_edges(
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


def _summary_stats(values: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "p90": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(arr.mean()),
        "median": float(np.quantile(arr, 0.5)),
        "p90": float(np.quantile(arr, 0.9)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", type=str, nargs="+", default=[], help="Graph dirs or files; ignored if --test-list is set")
    ap.add_argument("--test-list", type=str, default="", help="Optional: path to *.graph.pt list for evaluation")
    ap.add_argument("--val-list", type=str, default="", help="Optional: thresholds selected on this list, evaluated on --test-list")
    ap.add_argument("--backbone-ckpt", type=str, required=True)
    ap.add_argument("--explainer-ckpt", type=str, required=True)
    ap.add_argument("--device", type=str, default="cpu", help="cpu|cuda")

    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--ks", type=str, default="10,50,100", help="Graph-level P@K/R@K on suspicious scores")

    # Mirror hunt filters
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

    # Rarity
    ap.add_argument("--rarity-stats", type=str, required=True)
    ap.add_argument("--rarity-lambda", type=float, default=0.5)
    ap.add_argument("--rarity-idf-cap", type=float, default=3.0)
    ap.add_argument("--rarity-etypes", type=str, default="PROC|CONNECT|NET,PROC|EXEC|CMD")
    ap.add_argument("--rarity-normalize", type=str, default="")
    ap.add_argument("--op-fprs", type=str, default="0.01,0.05,0.10", help="3 operating points: target FPRs on val split")
    ap.add_argument("--latency-out", type=str, default="", help="Optional: write latency and metric summary JSON")

    args = ap.parse_args()
    t_total0 = time.perf_counter()

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    backbone = _load_backbone(args.backbone_ckpt, device=device)
    explainer = _load_explainer(args.explainer_ckpt, device=device)

    test_files = _read_list(args.test_list)
    graph_paths = [Path(x) for x in test_files] if test_files else _iter_graph_paths(args.graphs)
    if not graph_paths:
        raise SystemExit("No graphs found (provide --graphs or --test-list)")

    val_files = _read_list(args.val_list)
    val_paths = [Path(x) for x in val_files] if val_files else []

    cfg = HuntConfig(
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

    rarity_etypes = _as_set_csv(args.rarity_etypes)
    rarity = load_rarity_stats(str(args.rarity_stats))

    def score_paths(paths: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        y_true_: list[int] = []
        score_base_: list[float] = []
        score_rarity_: list[float] = []
        node_counts_: list[int] = []
        edge_counts_: list[int] = []

        with torch.no_grad():
            for gp in paths:
                yv = _get_y(gp)
                if yv is None or int(yv) not in (0, 1):
                    continue
                data = _load_graph(gp).to(device)

                # Graph size stats (sum over node/edge types).
                n_nodes = 0
                for nt in data.node_types:
                    try:
                        n_nodes += int(data[nt].num_nodes or 0)
                    except Exception:
                        pass
                n_edges = 0
                for et in data.edge_types:
                    try:
                        ei = data[et].edge_index
                        if ei is not None:
                            n_edges += int(ei.size(1))
                    except Exception:
                        pass

                ranked = topk_edges(backbone=backbone, explainer=explainer, hetero_batch=data, k=1_000_000)
                base_ranked = _filter_with_config(_to_suspicious_edges(ranked), cfg)
                base = _suspicious_score_from_ranked(base_ranked, k=cfg.k)

                ranked_r = _filter_with_config(
                    _to_suspicious_edges(
                        ranked,
                        rarity=rarity,
                        rarity_lambda=float(args.rarity_lambda),
                        rarity_idf_cap=float(args.rarity_idf_cap),
                        rarity_etypes=rarity_etypes,
                        rarity_normalize=str(args.rarity_normalize),
                    ),
                    cfg,
                )
                rar = _suspicious_score_from_ranked(ranked_r, k=cfg.k)

                y_true_.append(int(yv))
                score_base_.append(float(base))
                score_rarity_.append(float(rar))
                node_counts_.append(int(n_nodes))
                edge_counts_.append(int(n_edges))

        y_arr = np.asarray(y_true_, dtype=int)
        sb_arr = np.asarray(score_base_, dtype=float)
        sr_arr = np.asarray(score_rarity_, dtype=float)
        nc_arr = np.asarray(node_counts_, dtype=float)
        ec_arr = np.asarray(edge_counts_, dtype=float)
        return y_arr, sb_arr, sr_arr, nc_arr, ec_arr

    # Timing is best-effort and measured only on test split (for comparability with existing runs).
    graph_latencies: list[float] = []

    # Score test split (always).
    y_true: list[int] = []
    score_base: list[float] = []
    score_rarity: list[float] = []
    node_counts: list[int] = []
    edge_counts: list[int] = []

    with torch.no_grad():
        for gp in graph_paths:
            t_graph0 = time.perf_counter()
            yv = _get_y(gp)
            if yv is None or int(yv) not in (0, 1):
                continue
            data = _load_graph(gp).to(device)

            # Graph size stats (sum over node/edge types).
            n_nodes = 0
            for nt in data.node_types:
                try:
                    n_nodes += int(data[nt].num_nodes or 0)
                except Exception:
                    pass
            n_edges = 0
            for et in data.edge_types:
                try:
                    ei = data[et].edge_index
                    if ei is not None:
                        n_edges += int(ei.size(1))
                except Exception:
                    pass

            ranked = topk_edges(backbone=backbone, explainer=explainer, hetero_batch=data, k=1_000_000)
            base_ranked = _filter_with_config(_to_suspicious_edges(ranked), cfg)
            base = _suspicious_score_from_ranked(base_ranked, k=cfg.k)

            ranked_r = _filter_with_config(
                _to_suspicious_edges(
                    ranked,
                    rarity=rarity,
                    rarity_lambda=float(args.rarity_lambda),
                    rarity_idf_cap=float(args.rarity_idf_cap),
                    rarity_etypes=rarity_etypes,
                    rarity_normalize=str(args.rarity_normalize),
                ),
                cfg,
            )
            rar = _suspicious_score_from_ranked(ranked_r, k=cfg.k)

            y_true.append(int(yv))
            score_base.append(float(base))
            score_rarity.append(float(rar))
            node_counts.append(int(n_nodes))
            edge_counts.append(int(n_edges))
            graph_latencies.append(float(time.perf_counter() - t_graph0))

    if not y_true:
        raise SystemExit("No labeled graphs found (y in {0,1})")

    y = np.asarray(y_true, dtype=int)
    sb = np.asarray(score_base, dtype=float)
    sr = np.asarray(score_rarity, dtype=float)
    nc = np.asarray(node_counts, dtype=float)
    ec = np.asarray(edge_counts, dtype=float)

    print(f"eval_n={len(y)} pos={int((y==1).sum())} neg={int((y==0).sum())}")
    if len(nc) > 0:
        def q(arr, p):
            return float(np.quantile(arr, p))
        print(
            f"graph_size: nodes mean={float(nc.mean()):.1f} median={q(nc,0.5):.1f} p90={q(nc,0.9):.1f} | "
            f"edges mean={float(ec.mean()):.1f} median={q(ec,0.5):.1f} p90={q(ec,0.9):.1f}"
        )

    print("\n== suspicious_score (mean top-k; higher means more suspicious) ==")
    base_auc = _safe_auc(y, sb)
    base_auprc = _safe_auprc(y, sb)

    rar_auc = _safe_auc(y, sr)
    rar_auprc = _safe_auprc(y, sr)

    print(f"base_auroc={base_auc:.4f} base_auprc={base_auprc:.4f}")
    print(f"rarity_auroc={rar_auc:.4f} rarity_auprc={rar_auprc:.4f}")

    # Sanity check: score distributions per class (helps debug sign/label issues).
    def _cls_summary(name: str, scores: np.ndarray) -> None:
        pos = scores[y == 1]
        neg = scores[y == 0]

        def _summ(arr: np.ndarray) -> str:
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                return "empty"
            return (
                f"mean={float(arr.mean()):.4g} "
                f"p10={float(np.quantile(arr,0.1)):.4g} "
                f"p50={float(np.quantile(arr,0.5)):.4g} "
                f"p90={float(np.quantile(arr,0.9)):.4g}"
            )

        print(f"{name} score_by_class: y=1 {_summ(pos)} | y=0 {_summ(neg)}")

    _cls_summary("base", sb)
    _cls_summary("rarity", sr)

    ks = [int(x) for x in str(args.ks).split(",") if x.strip()]
    print("\n== graph-level ranking (by suspicious_score) ==")
    p_at: dict[str, dict[str, float]] = {"base": {}, "rarity": {}}
    r_at: dict[str, dict[str, float]] = {"base": {}, "rarity": {}}
    for k, p, r in _precision_recall_at_k(y, sb, ks):
        p_at["base"][str(k)] = float(p)
        r_at["base"][str(k)] = float(r)
        print(f"base P@{k}={p:.4f} R@{k}={r:.4f}")
    for k, p, r in _precision_recall_at_k(y, sr, ks):
        p_at["rarity"][str(k)] = float(p)
        r_at["rarity"][str(k)] = float(r)
        print(f"rarity P@{k}={p:.4f} R@{k}={r:.4f}")

    # Operating points: select thresholds on val split, evaluate on test split.
    op_fprs = [float(x) for x in str(args.op_fprs).split(",") if x.strip()]
    if op_fprs:
        if val_paths:
            yv, sbv, srv, _ncv, _ecv = score_paths(val_paths)
            val_note = f"val_n={len(yv)}"
        else:
            # Fall back to selecting on test (not ideal, but still useful for quick sanity checks).
            yv, sbv, srv = y, sb, sr
            val_note = "val_list=NONE (thresholds selected on test; interpret cautiously)"

        print("\n== operating points (thresholds selected on val; evaluated on test) ==")
        if not val_paths:
            print(val_note)

        op_report: dict[str, dict[str, dict[str, float]]] = {"base": {}, "rarity": {}}
        for tgt in op_fprs:
            thr_b = _threshold_at_fpr(yv, sbv, target_fpr=tgt)
            thr_r = _threshold_at_fpr(yv, srv, target_fpr=tgt)
            mb = _binary_metrics_at_threshold(y, sb, thr=thr_b)
            mr = _binary_metrics_at_threshold(y, sr, thr=thr_r)
            op_report["base"][str(tgt)] = mb
            op_report["rarity"][str(tgt)] = mr
            print(
                f"base  fpr<= {tgt:.3f}@val  thr={mb['thr']:.6g}  prec={mb['prec']:.4f}  rec={mb['rec']:.4f}  "
                f"f1={mb['f1']:.4f}  tp={int(mb['tp'])}  fp={int(mb['fp'])}  tn={int(mb['tn'])}  fn={int(mb['fn'])}  fpr={mb['fpr']:.4f}"
            )
            print(
                f"rarity fpr<= {tgt:.3f}@val  thr={mr['thr']:.6g}  prec={mr['prec']:.4f}  rec={mr['rec']:.4f}  "
                f"f1={mr['f1']:.4f}  tp={int(mr['tp'])}  fp={int(mr['fp'])}  tn={int(mr['tn'])}  fn={int(mr['fn'])}  fpr={mr['fpr']:.4f}"
            )

    if args.latency_out:
        out_path = Path(args.latency_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "n_graphs_requested": int(len(graph_paths)),
            "n_graphs_used": int(len(y)),
            "device": str(device),
            "k": int(args.k),
            "ks": ks,
            "backbone_ckpt": str(args.backbone_ckpt),
            "explainer_ckpt": str(args.explainer_ckpt),
            "rarity_stats": str(args.rarity_stats),
            "rarity_lambda": float(args.rarity_lambda),
            "rarity_idf_cap": float(args.rarity_idf_cap),
            "rarity_etypes": sorted(rarity_etypes),
            "rarity_normalize": str(args.rarity_normalize),
            "metrics": {
                "base_auroc": float(base_auc),
                "base_auprc": float(base_auprc),
                "rarity_auroc": float(rar_auc),
                "rarity_auprc": float(rar_auprc),
                "p_at_k": p_at,
                "r_at_k": r_at,
                "operating_points": {
                    "op_fprs": op_fprs,
                    "selected_on": "val" if bool(val_paths) else "test",
                    "report": op_report if op_fprs else {},
                },
            },
            "latency_seconds": {
                "total": float(time.perf_counter() - t_total0),
                "per_graph": _summary_stats(graph_latencies),
            },
        }
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"latency_out={out_path.as_posix()}")


if __name__ == "__main__":
    main()

