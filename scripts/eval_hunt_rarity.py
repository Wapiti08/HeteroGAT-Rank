from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.data import HeteroData

# Ensure repo root importable when running as script.
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from model.rgcn import RGCNGraphClassifier  # noqa: E402
from ranking_explain.hunt import RankedEdge, topk_edges  # noqa: E402
from ranking_explain.pgexplainer import PGExplainer  # noqa: E402
from ranking_explain.rarity import etype_to_str, load_rarity_stats, normalize_dst_key  # noqa: E402
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


def _anomaly_score_from_ranked(ranked: list[RankedEdge], *, k: int) -> float:
    if not ranked:
        return float("nan")
    kk = min(int(k), len(ranked))
    if kk <= 0:
        return float("nan")
    return float(sum(r.score for r in ranked[:kk]) / float(kk))


def _apply_rarity(
    ranked: list[RankedEdge],
    *,
    rarity_stats_path: str,
    rarity_lambda: float,
    rarity_idf_cap: float,
    rarity_etypes: Set[str],
    rarity_normalize: str,
) -> list[RankedEdge]:
    rarity = load_rarity_stats(rarity_stats_path)
    scheme = rarity_normalize.strip() or str(getattr(rarity, "normalize", "none"))
    lam = float(rarity_lambda)
    cap = float(rarity_idf_cap)
    out: list[RankedEdge] = []
    for r in ranked:
        allow = (not rarity_etypes) or (etype_to_str(r.etype) in rarity_etypes)
        bonus = 0.0
        if allow and lam != 0.0:
            norm_key = normalize_dst_key(scheme=scheme, etype=r.etype, dst_type=r.dst_type, dst_key=str(r.dst_key))
            idf = float(rarity.idf(etype=r.etype, dst_key=norm_key))
            if cap > 0.0:
                idf = min(idf, cap)
            bonus = lam * idf
        out.append(
            RankedEdge(
                graph_id=r.graph_id,
                score=float(r.score + bonus),
                etype=r.etype,
                src_type=r.src_type,
                src_key=r.src_key,
                dst_type=r.dst_type,
                dst_key=r.dst_key,
            )
        )
    out.sort(key=lambda x: x.score, reverse=True)
    return out


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", type=str, nargs="+", default=[], help="Graph dirs or files; ignored if --test-list is set")
    ap.add_argument("--test-list", type=str, default="", help="Optional: path to *.graph.pt list for evaluation")
    ap.add_argument("--backbone-ckpt", type=str, required=True)
    ap.add_argument("--explainer-ckpt", type=str, required=True)
    ap.add_argument("--device", type=str, default="cpu", help="cpu|cuda")

    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--ks", type=str, default="10,50,100", help="Graph-level P@K/R@K on anomaly scores")

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

    args = ap.parse_args()

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    backbone = _load_backbone(args.backbone_ckpt, device=device)
    explainer = _load_explainer(args.explainer_ckpt, device=device)

    test_files = _read_list(args.test_list)
    graph_paths = [Path(x) for x in test_files] if test_files else _iter_graph_paths(args.graphs)
    if not graph_paths:
        raise SystemExit("No graphs found (provide --graphs or --test-list)")

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

    y_true: list[int] = []
    score_base: list[float] = []
    score_rarity: list[float] = []

    with torch.no_grad():
        for gp in graph_paths:
            y = _get_y(gp)
            if y is None or int(y) not in (0, 1):
                continue

            data = _load_graph(gp).to(device)

            ranked = topk_edges(backbone=backbone, explainer=explainer, hetero_batch=data, k=max(int(cfg.k) * 20, int(cfg.k)))
            ranked = _filter_ranked_edges(
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
            ranked.sort(key=lambda r: r.score, reverse=True)

            base = _anomaly_score_from_ranked(ranked, k=cfg.k)

            ranked_r = _apply_rarity(
                ranked,
                rarity_stats_path=str(args.rarity_stats),
                rarity_lambda=float(args.rarity_lambda),
                rarity_idf_cap=float(args.rarity_idf_cap),
                rarity_etypes=rarity_etypes,
                rarity_normalize=str(args.rarity_normalize),
            )
            rar = _anomaly_score_from_ranked(ranked_r, k=cfg.k)

            y_true.append(int(y))
            score_base.append(float(base))
            score_rarity.append(float(rar))

    if not y_true:
        raise SystemExit("No labeled graphs found (y in {0,1})")

    y = np.asarray(y_true, dtype=int)
    sb = np.asarray(score_base, dtype=float)
    sr = np.asarray(score_rarity, dtype=float)

    print(f"eval_n={len(y)} pos={int((y==1).sum())} neg={int((y==0).sum())}")

    print("\n== anomaly_score (mean top-k) ==")
    # Depending on how the explainer score is calibrated, higher may mean "more normal"
    # (e.g., stable/install-like patterns) rather than "more anomalous". Report both
    # directions so users can pick the correct orientation.
    base_auc = _safe_auc(y, sb)
    base_auc_inv = _safe_auc(y, -sb)
    base_auprc = _safe_auprc(y, sb)
    base_auprc_inv = _safe_auprc(y, -sb)

    rar_auc = _safe_auc(y, sr)
    rar_auc_inv = _safe_auc(y, -sr)
    rar_auprc = _safe_auprc(y, sr)
    rar_auprc_inv = _safe_auprc(y, -sr)

    print(f"base_auroc={base_auc:.4f} base_auprc={base_auprc:.4f} | inverted: auroc={base_auc_inv:.4f} auprc={base_auprc_inv:.4f}")
    print(f"rarity_auroc={rar_auc:.4f} rarity_auprc={rar_auprc:.4f} | inverted: auroc={rar_auc_inv:.4f} auprc={rar_auprc_inv:.4f}")

    ks = [int(x) for x in str(args.ks).split(",") if x.strip()]
    print("\n== graph-level ranking (by anomaly_score) ==")
    for k, p, r in _precision_recall_at_k(y, sb, ks):
        print(f"base P@{k}={p:.4f} R@{k}={r:.4f}")
    for k, p, r in _precision_recall_at_k(y, sr, ks):
        print(f"rarity P@{k}={p:.4f} R@{k}={r:.4f}")

    print("\n== graph-level ranking (by inverted anomaly_score) ==")
    for k, p, r in _precision_recall_at_k(y, -sb, ks):
        print(f"base(inv) P@{k}={p:.4f} R@{k}={r:.4f}")
    for k, p, r in _precision_recall_at_k(y, -sr, ks):
        print(f"rarity(inv) P@{k}={p:.4f} R@{k}={r:.4f}")


if __name__ == "__main__":
    main()

