from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Optional, Sequence, Set

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# Ensure repo root importable when running as script.
REPO_ROOT = Path(__file__).resolve().parent.parent
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from ranking_explain.hunt import RankedEdge, ranked_edges_from_scores, topk_edges  # noqa: E402
from ranking_explain.rarity import RarityStats, etype_to_str, load_rarity_stats  # noqa: E402
from scripts.eval_hunt_rarity import (  # noqa: E402
    HuntConfig,
    _as_set_csv,
    _binary_metrics_at_threshold,
    _filter_with_config,
    _get_y,
    _load_backbone,
    _load_explainer,
    _load_graph,
    _mean_topk_score,
    _precision_recall_at_k,
    _rarity_bonus,
    _read_list,
    _safe_auc,
    _safe_auprc,
    _threshold_at_fpr,
    _to_suspicious_edges,
)


def _stable_seed(path: Path, *, seed: int) -> int:
    raw = f"{int(seed)}::{path.as_posix()}".encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:16], 16) % (2**31)


def _homogeneous_ranked_edges(*, backbone, hetero_batch) -> list[RankedEdge]:
    backbone.eval()
    with torch.no_grad():
        _logits, _x, data = backbone(hetero_batch, return_node_emb=True)
    scores = torch.zeros(int(data.edge_index.size(1)), device=data.edge_index.device)
    return ranked_edges_from_scores(hetero_batch=hetero_batch, data=data, scores=scores, k=1_000_000)


def _random_ranked_edges(*, backbone, hetero_batch, graph_path: Path, seed: int) -> list[RankedEdge]:
    backbone.eval()
    with torch.no_grad():
        _logits, _x, data = backbone(hetero_batch, return_node_emb=True)
    gen = torch.Generator(device=data.edge_index.device)
    gen.manual_seed(_stable_seed(graph_path, seed=seed))
    scores = torch.rand(int(data.edge_index.size(1)), generator=gen, device=data.edge_index.device)
    return ranked_edges_from_scores(hetero_batch=hetero_batch, data=data, scores=scores, k=1_000_000)


def _model_malicious_probability(*, backbone, hetero_batch) -> float:
    backbone.eval()
    with torch.no_grad():
        logits = backbone(hetero_batch)
        prob = torch.softmax(logits.view(1, -1), dim=-1)
    if prob.size(-1) < 2:
        return float("nan")
    return float(prob[0, 1].item())


def _rarity_only_edges(
    ranked: list[RankedEdge],
    *,
    rarity: RarityStats,
    rarity_idf_cap: float,
    rarity_etypes: Set[str],
    rarity_normalize: str,
) -> list[RankedEdge]:
    out: list[RankedEdge] = []
    for r in ranked:
        score = _rarity_bonus(
            r,
            rarity=rarity,
            rarity_lambda=1.0,
            rarity_idf_cap=rarity_idf_cap,
            rarity_etypes=rarity_etypes,
            rarity_normalize=rarity_normalize,
        )
        out.append(
            RankedEdge(
                graph_id=r.graph_id,
                score=float(score),
                etype=r.etype,
                src_type=r.src_type,
                src_key=r.src_key,
                dst_type=r.dst_type,
                dst_key=r.dst_key,
            )
        )
    out.sort(key=lambda x: x.score, reverse=True)
    return out


class _HomogeneousRGCNWrapper(nn.Module):
    def __init__(self, backbone) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        edge_type: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        h = x
        for conv in self.backbone.convs:
            h = conv(h, edge_index, edge_type)
            h = F.relu(h)
            h = F.dropout(h, p=float(self.backbone.dropout), training=False)
        from torch_geometric.nn import global_mean_pool

        g = global_mean_pool(h, batch)
        return self.backbone.classifier(g)


def _gnnexplainer_ranked_edges(
    *,
    backbone,
    hetero_batch,
    epochs: int,
    target_class: int,
) -> list[RankedEdge]:
    from torch_geometric.explain import Explainer
    from torch_geometric.explain.algorithm import GNNExplainer
    from torch_geometric.explain.algorithm.utils import clear_masks

    backbone.eval()
    with torch.no_grad():
        _logits, _x, data = backbone(hetero_batch, return_node_emb=True)

    assert backbone.node_type_emb is not None
    x0 = backbone.node_type_emb(data.node_type).detach()
    batch = getattr(data, "batch", torch.zeros(x0.size(0), dtype=torch.long, device=x0.device))
    wrapper = _HomogeneousRGCNWrapper(backbone).to(x0.device)
    wrapper.eval()

    explainer = Explainer(
        model=wrapper,
        algorithm=GNNExplainer(epochs=int(epochs)),
        explanation_type="phenomenon",
        node_mask_type=None,
        edge_mask_type="object",
        model_config={
            "mode": "multiclass_classification",
            "task_level": "graph",
            "return_type": "raw",
        },
    )
    target = torch.tensor([int(target_class)], dtype=torch.long, device=x0.device)
    try:
        explanation = explainer(x0, data.edge_index, target=target, edge_type=data.edge_type, batch=batch)
    finally:
        clear_masks(wrapper)
    edge_mask = explanation.edge_mask
    if edge_mask is None:
        raise RuntimeError("GNNExplainer did not return an edge_mask")
    return ranked_edges_from_scores(hetero_batch=hetero_batch, data=data, scores=edge_mask, k=1_000_000)


def _score_ranked(ranked: list[RankedEdge], *, cfg: HuntConfig) -> float:
    return _mean_topk_score(_filter_with_config(ranked, cfg), k=cfg.k)


def _score_ranked_raw(ranked: list[RankedEdge], *, k: int) -> float:
    return _mean_topk_score(ranked, k=k)


def _format_float(x: float) -> str:
    if np.isnan(float(x)):
        return "nan"
    return f"{float(x):.6g}"


def _finite_xy(y: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(score)
    return y[mask], score[mask]


def _method_auc(y: np.ndarray, score: np.ndarray) -> float:
    yy, ss = _finite_xy(y, score)
    if yy.size == 0:
        return float("nan")
    return _safe_auc(yy, ss)


def _method_auprc(y: np.ndarray, score: np.ndarray) -> float:
    yy, ss = _finite_xy(y, score)
    if yy.size == 0:
        return float("nan")
    return _safe_auprc(yy, ss)


def _method_precision_recall_at_k(y: np.ndarray, score: np.ndarray, ks: list[int]) -> list[tuple[int, float, float]]:
    yy, ss = _finite_xy(y, score)
    if yy.size == 0:
        return [(int(k), float("nan"), float("nan")) for k in ks]
    return _precision_recall_at_k(yy, ss, ks)


def _iter_graph_paths(graph_args: Sequence[str]) -> list[Path]:
    out: list[Path] = []
    for a in graph_args:
        p = Path(a)
        if p.is_file() and p.name.endswith(".graph.pt"):
            out.append(p)
        elif p.is_dir():
            out.extend(sorted(p.glob("*.graph.pt")))
    return out


def _behavior_stats(hetero_batch) -> dict[str, object]:
    family_edges = {"NET": 0, "CMD": 0, "FILE": 0, "SYSCALL": 0, "PROC": 0, "OTHER": 0}
    behavior_edges = 0
    families_seen: set[str] = set()
    etypes_seen: set[str] = set()
    load_edges = 0

    for etype in hetero_batch.edge_types:
        try:
            n_edges = int(hetero_batch[etype].edge_index.size(1))
        except Exception:
            n_edges = 0
        if n_edges <= 0:
            continue

        src, rel, dst = etype
        etypes_seen.add(f"{src}|{rel}|{dst}")
        if rel == "LOAD":
            load_edges += n_edges
            continue

        behavior_edges += n_edges
        if dst in family_edges:
            family = dst
        elif rel in ("DNS_QUERY", "RESOLVE", "CONNECT"):
            family = "NET"
        elif rel == "EXEC":
            family = "CMD" if dst == "CMD" else "PROC"
        elif rel in ("READ", "WRITE", "DELETE"):
            family = "FILE"
        elif rel == "INVOKE":
            family = "SYSCALL"
        else:
            family = "OTHER"
        family_edges[family] = family_edges.get(family, 0) + n_edges
        families_seen.add(family)

    richness = len(families_seen)
    if behavior_edges <= 0:
        bucket = "Load-only"
    elif behavior_edges < 10 or richness <= 1:
        bucket = "Low-observability"
    elif behavior_edges < 50 or richness <= 2:
        bucket = "Medium-observability"
    else:
        bucket = "High-observability"

    return {
        "observability_bucket": bucket,
        "behavior_edges": int(behavior_edges),
        "load_edges": int(load_edges),
        "behavior_family_richness": int(richness),
        "edge_type_richness": int(len(etypes_seen)),
        "family_edges": {k: int(v) for k, v in family_edges.items()},
    }


def _score_paths(
    *,
    paths: list[Path],
    backbone,
    explainer,
    rarity: RarityStats,
    cfg: HuntConfig,
    rarity_lambda: float,
    rarity_idf_cap: float,
    rarity_etypes: Set[str],
    rarity_normalize: str,
    random_seed: int,
    include_gnnexplainer: bool,
    gnnexplainer_epochs: int,
    gnnexplainer_target: int,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, np.ndarray], list[dict[str, object]]]:
    y_true: list[int] = []
    scores: dict[str, list[float]] = {
        "Random-K": [],
        "Raw model probability": [],
        "Rarity-only": [],
        "HetHunt raw": [],
        "HetHunt + filtering": [],
        "HetHunt + filtering + rarity": [],
    }
    if include_gnnexplainer:
        scores["GNNExplainer"] = []
    per_graph: list[dict[str, object]] = []

    for gp in paths:
        yv = _get_y(gp)
        if yv is None or int(yv) not in (0, 1):
            continue
        data = _load_graph(gp).to(device)
        obs = _behavior_stats(data)

        all_ranked = _homogeneous_ranked_edges(backbone=backbone, hetero_batch=data)
        model_prob = _model_malicious_probability(backbone=backbone, hetero_batch=data)

        rand_ranked = _random_ranked_edges(backbone=backbone, hetero_batch=data, graph_path=gp, seed=random_seed)
        rarity_ranked = _rarity_only_edges(
            all_ranked,
            rarity=rarity,
            rarity_idf_cap=rarity_idf_cap,
            rarity_etypes=rarity_etypes,
            rarity_normalize=rarity_normalize,
        )
        pg_ranked_raw = topk_edges(backbone=backbone, explainer=explainer, hetero_batch=data, k=1_000_000)
        hethunt_raw = _to_suspicious_edges(pg_ranked_raw)
        hethunt_ranked = _to_suspicious_edges(
            pg_ranked_raw,
            rarity=rarity,
            rarity_lambda=float(rarity_lambda),
            rarity_idf_cap=float(rarity_idf_cap),
            rarity_etypes=rarity_etypes,
            rarity_normalize=rarity_normalize,
        )

        y_true.append(int(yv))
        graph_scores = {
            "Random-K": _score_ranked(rand_ranked, cfg=cfg),
            "Raw model probability": model_prob,
            "Rarity-only": _score_ranked(rarity_ranked, cfg=cfg),
            "HetHunt raw": _score_ranked_raw(hethunt_raw, k=cfg.k),
            "HetHunt + filtering": _score_ranked(hethunt_raw, cfg=cfg),
            "HetHunt + filtering + rarity": _score_ranked(hethunt_ranked, cfg=cfg),
        }
        for method, value in graph_scores.items():
            scores[method].append(float(value))

        if include_gnnexplainer:
            try:
                gnn_ranked = _gnnexplainer_ranked_edges(
                    backbone=backbone,
                    hetero_batch=data,
                    epochs=gnnexplainer_epochs,
                    target_class=gnnexplainer_target,
                )
                graph_scores["GNNExplainer"] = _score_ranked(gnn_ranked, cfg=cfg)
                scores["GNNExplainer"].append(float(graph_scores["GNNExplainer"]))
            except Exception as exc:
                print(
                    f"warning: GNNExplainer failed for {gp.as_posix()}: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                graph_scores["GNNExplainer"] = float("nan")
                scores["GNNExplainer"].append(float("nan"))

        per_graph.append(
            {
                "path": gp.as_posix(),
                "y": int(yv),
                "scores": {k: float(v) for k, v in graph_scores.items()},
                **obs,
            }
        )

    return np.asarray(y_true, dtype=int), {k: np.asarray(v, dtype=float) for k, v in scores.items()}, per_graph


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", type=str, nargs="+", default=[], help="Graph dirs or files; ignored if --test-list is set")
    ap.add_argument("--test-list", type=str, default="", help="Optional: path to *.graph.pt list for evaluation")
    ap.add_argument("--val-list", type=str, default="", help="Optional: thresholds selected on this list, evaluated on --test-list")
    ap.add_argument("--backbone-ckpt", type=str, required=True)
    ap.add_argument("--explainer-ckpt", type=str, required=True)
    ap.add_argument("--rarity-stats", type=str, required=True)
    ap.add_argument("--device", type=str, default="cpu", help="cpu|cuda")
    ap.add_argument("--out", type=str, default="", help="Optional TSV output path")

    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--ks", type=str, default="10,50,100", help="Graph-level P@K/R@K on suspicious scores")
    ap.add_argument("--op-fprs", type=str, default="0.01,0.05,0.10")

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

    ap.add_argument("--rarity-lambda", type=float, default=0.5)
    ap.add_argument("--rarity-idf-cap", type=float, default=3.0)
    ap.add_argument("--rarity-etypes", type=str, default="PROC|CONNECT|NET,PROC|EXEC|CMD")
    ap.add_argument("--rarity-normalize", type=str, default="")

    ap.add_argument("--random-seed", type=int, default=0)
    ap.add_argument("--include-gnnexplainer", action="store_true", default=False)
    ap.add_argument("--gnnexplainer-epochs", type=int, default=100)
    ap.add_argument("--gnnexplainer-target", type=int, default=1, help="Class explained by GNNExplainer; 1 = malicious")
    ap.add_argument("--json-out", type=str, default="", help="Optional detailed JSON output")
    args = ap.parse_args()

    t0 = time.perf_counter()
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    backbone = _load_backbone(args.backbone_ckpt, device=device)
    explainer = _load_explainer(args.explainer_ckpt, device=device)
    rarity = load_rarity_stats(args.rarity_stats)
    rarity_etypes = _as_set_csv(args.rarity_etypes)

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

    score_kwargs = {
        "backbone": backbone,
        "explainer": explainer,
        "rarity": rarity,
        "cfg": cfg,
        "rarity_lambda": float(args.rarity_lambda),
        "rarity_idf_cap": float(args.rarity_idf_cap),
        "rarity_etypes": rarity_etypes,
        "rarity_normalize": str(args.rarity_normalize),
        "random_seed": int(args.random_seed),
        "include_gnnexplainer": bool(args.include_gnnexplainer),
        "gnnexplainer_epochs": int(args.gnnexplainer_epochs),
        "gnnexplainer_target": int(args.gnnexplainer_target),
        "device": device,
    }
    y, method_scores, per_graph = _score_paths(paths=graph_paths, **score_kwargs)
    if len(y) == 0:
        raise SystemExit("No labeled graphs found (y in {0,1})")

    val_y: Optional[np.ndarray] = None
    val_scores: Optional[dict[str, np.ndarray]] = None
    if val_paths:
        val_y, val_scores, _val_per_graph = _score_paths(paths=val_paths, **score_kwargs)

    ks = [int(x) for x in str(args.ks).split(",") if x.strip()]
    op_fprs = [float(x) for x in str(args.op_fprs).split(",") if x.strip()]
    header = ["method", "auroc", "auprc"]
    for k in ks:
        header.extend([f"P@{k}", f"R@{k}"])
    for fpr in op_fprs:
        header.extend([f"thr@fpr{fpr:g}", f"prec@fpr{fpr:g}", f"rec@fpr{fpr:g}", f"f1@fpr{fpr:g}"])

    rows: list[list[str]] = []
    report: dict[str, object] = {
        "n_graphs": int(len(y)),
        "pos": int((y == 1).sum()),
        "neg": int((y == 0).sum()),
        "config": {
            "k": int(args.k),
            "ks": ks,
            "op_fprs": op_fprs,
            "rarity_lambda": float(args.rarity_lambda),
            "rarity_idf_cap": float(args.rarity_idf_cap),
            "rarity_etypes": sorted(rarity_etypes),
            "rarity_normalize": str(args.rarity_normalize),
            "include_gnnexplainer": bool(args.include_gnnexplainer),
        },
        "methods": {},
        "per_graph": per_graph,
    }

    for method, score in method_scores.items():
        auroc = _method_auc(y, score)
        auprc = _method_auprc(y, score)
        row = [method, _format_float(auroc), _format_float(auprc)]
        p_at: dict[str, float] = {}
        r_at: dict[str, float] = {}
        for k, p, r in _method_precision_recall_at_k(y, score, ks):
            p_at[str(k)] = float(p)
            r_at[str(k)] = float(r)
            row.extend([_format_float(p), _format_float(r)])

        op_report: dict[str, dict[str, float]] = {}
        if op_fprs:
            threshold_y = val_y if val_y is not None else y
            threshold_scores = val_scores[method] if val_scores is not None else score
            for fpr in op_fprs:
                ty, ts = _finite_xy(threshold_y, threshold_scores)
                yy, ss = _finite_xy(y, score)
                if ty.size == 0 or yy.size == 0:
                    metrics = {"thr": float("nan"), "prec": float("nan"), "rec": float("nan"), "f1": float("nan")}
                else:
                    thr = _threshold_at_fpr(ty, ts, target_fpr=fpr)
                    metrics = _binary_metrics_at_threshold(yy, ss, thr=thr)
                op_report[str(fpr)] = metrics
                row.extend([
                    _format_float(metrics["thr"]),
                    _format_float(metrics["prec"]),
                    _format_float(metrics["rec"]),
                    _format_float(metrics["f1"]),
                ])

        rows.append(row)
        report["methods"][method] = {
            "auroc": float(auroc),
            "auprc": float(auprc),
            "p_at_k": p_at,
            "r_at_k": r_at,
            "operating_points": op_report,
        }

    lines = ["\t".join(header), *["\t".join(row) for row in rows]]
    text = "\n".join(lines) + "\n"
    print(f"eval_n={len(y)} pos={int((y == 1).sum())} neg={int((y == 0).sum())}")
    print(text, end="")
    print(f"elapsed_s={time.perf_counter() - t0:.3f}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text)
        print(f"wrote_tsv={out_path.as_posix()}")

    if args.json_out:
        json_path = Path(args.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        report["elapsed_seconds"] = float(time.perf_counter() - t0)
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"wrote_json={json_path.as_posix()}")


if __name__ == "__main__":
    main()
