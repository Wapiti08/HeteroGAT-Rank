from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
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


def _unique_paths(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = path.as_posix()
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
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


def _as_set_csv(s: str) -> Set[str]:
    return {x.strip() for x in str(s).split(",") if x.strip()}


def _read_csv_or_list(value: str | None, list_path: str | None) -> list[str]:
    items: list[str] = []
    if value:
        items.extend(x.strip() for x in str(value).split(",") if x.strip())
    items.extend(_read_list(list_path))
    return items


def _graph_id(path: Path) -> str:
    name = path.name
    suffix = ".graph.pt"
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return path.stem


def _matches_package(path: Path, package: str) -> bool:
    q = package.strip().lower()
    if not q:
        return False
    graph_id = _graph_id(path).lower()
    if graph_id == q or graph_id.startswith(f"{q}-") or graph_id.startswith(f"{q}@"):
        return True
    return q in graph_id


@dataclass(frozen=True)
class HuntFilters:
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


def _filter_with_config(ranked: list[RankedEdge], filt: HuntFilters) -> list[RankedEdge]:
    return _filter_ranked_edges(
        ranked,
        filter_net=filt.filter_net,
        filter_net_ip=filt.filter_net_ip,
        filter_domains=filt.filter_domains,
        filter_system_files=filt.filter_system_files,
        system_file_prefixes=filt.system_file_prefixes,
        filter_tmp_tempfile=filt.filter_tmp_tempfile,
        filter_cmd_noise=filt.filter_cmd_noise,
        cmd_noise_keywords=filt.cmd_noise_keywords,
        dedup_dst=filt.dedup_dst,
        max_per_etype=filt.max_per_etype,
    )


def _to_suspicious_edges(
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


def _suspicious_score(ranked: list[RankedEdge], *, k: int) -> float:
    if not ranked:
        return float("nan")
    kk = min(int(k), len(ranked))
    if kk <= 0:
        return float("nan")
    return float(sum(r.score for r in ranked[:kk]) / float(kk))


def _port_from_dst_key(dst_key: str) -> Optional[int]:
    s = str(dst_key)
    m = re.search(r"port::(\d+)", s)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _has_suspicious_port(ranked: list[RankedEdge], *, ports: Set[int]) -> bool:
    for r in ranked:
        if r.dst_type != "NET":
            continue
        p = _port_from_dst_key(str(r.dst_key))
        if p is not None and p in ports:
            return True
    return False


def _format_edge_row(r: RankedEdge) -> str:
    # Keep dst shorter in tables.
    dst = str(r.dst_key)
    if len(dst) > 100:
        dst = dst[:97] + "..."
    return f"| {r.score:.4f} | `{etype_to_str(r.etype)}` | `{r.src_type}:{r.src_key}` | `{r.dst_type}:{dst}` |"


def _emit_case_markdown(
    *,
    title: str,
    graph_path: Path,
    y: int,
    base_rank: int,
    rarity_rank: int,
    base_edges: list[RankedEdge],
    rarity_edges: list[RankedEdge],
    k_show: int,
) -> str:
    lines: list[str] = []
    lines.append(f"### {title}")
    lines.append("")
    lines.append(f"- **graph**: `{graph_path.as_posix()}`")
    lines.append(f"- **label (y/Level)**: `{y}`")
    lines.append(f"- **triage rank (lower is more suspicious)**: base `{base_rank}` -> rarity `{rarity_rank}`")
    lines.append("")
    lines.append("#### Top-K suspicious edges (base)")
    lines.append("")
    lines.append("| suspicious score | etype | src | dst |")
    lines.append("|---:|---|---|---|")
    for r in base_edges[:k_show]:
        lines.append(_format_edge_row(r))
    lines.append("")
    lines.append("#### Top-K suspicious edges (rarity-adjusted)")
    lines.append("")
    lines.append("| suspicious score | etype | src | dst |")
    lines.append("|---:|---|---|---|")
    for r in rarity_edges[:k_show]:
        lines.append(_format_edge_row(r))
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-list", type=str, required=True, help="Path to *.graph.pt list")
    ap.add_argument(
        "--extra-graphs",
        type=str,
        default="",
        help="Comma-separated extra *.graph.pt paths to include in the ranking pool",
    )
    ap.add_argument(
        "--extra-graph-list",
        type=str,
        default="",
        help="File with extra *.graph.pt paths to include in the ranking pool",
    )
    ap.add_argument("--backbone-ckpt", type=str, required=True)
    ap.add_argument("--explainer-ckpt", type=str, required=True)
    ap.add_argument("--device", type=str, default="cpu", help="cpu|cuda")

    ap.add_argument("--out", type=str, default="artifacts/case_studies.md")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--k-show", type=int, default=12, help="How many edges to show per case")
    ap.add_argument("--topn", type=int, default=3, help="How many cases to emit")
    ap.add_argument("--require-y", type=int, default=1, help="Only emit cases with this label (default: 1)")
    ap.add_argument(
        "--include-packages",
        type=str,
        default="",
        help="Comma-separated package names to emit as cases before auto-selected cases",
    )
    ap.add_argument(
        "--include-package-list",
        type=str,
        default="",
        help="File with one package name per line to emit as cases before auto-selected cases",
    )
    ap.add_argument(
        "--prefer-suspicious-ports",
        type=str,
        default="22,23,25,53,110,143,389,445,465,587,6667,3306,3389,4444,5555,8080,8443,9001,9050,19305",
    )

    # Filters aligned with run_hunt defaults
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

    graph_paths = [Path(p) for p in _read_list(args.test_list)]
    graph_paths.extend(Path(p) for p in _read_csv_or_list(args.extra_graphs, args.extra_graph_list))
    graph_paths = _unique_paths(graph_paths)
    graph_paths = [p for p in graph_paths if p.exists() and p.name.endswith(".graph.pt")]
    if not graph_paths:
        raise SystemExit("No *.graph.pt paths found in --test-list")

    filt = HuntFilters(
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
    susp_ports = {int(x) for x in str(args.prefer_suspicious_ports).split(",") if x.strip().isdigit()}

    # Compute base + rarity ranks over all labeled graphs.
    rows = []
    for gp in graph_paths:
        y = _get_y(gp)
        if y is None or int(y) not in (0, 1):
            continue
        data = _load_graph(gp).to(device)
        ranked = topk_edges(backbone=backbone, explainer=explainer, hetero_batch=data, k=1_000_000)
        base_ranked = _filter_with_config(
            _to_suspicious_edges(
                ranked,
                rarity_stats_path=str(args.rarity_stats),
                rarity_lambda=0.0,
                rarity_idf_cap=float(args.rarity_idf_cap),
                rarity_etypes=rarity_etypes,
                rarity_normalize=str(args.rarity_normalize),
            ),
            filt,
        )
        base_score = _suspicious_score(base_ranked, k=int(args.k))

        ranked_r = _filter_with_config(
            _to_suspicious_edges(
                ranked,
                rarity_stats_path=str(args.rarity_stats),
                rarity_lambda=float(args.rarity_lambda),
                rarity_idf_cap=float(args.rarity_idf_cap),
                rarity_etypes=rarity_etypes,
                rarity_normalize=str(args.rarity_normalize),
            ),
            filt,
        )
        rar_score = _suspicious_score(ranked_r, k=int(args.k))

        rows.append(
            {
                "path": gp,
                "y": int(y),
                "base_triage": float(base_score),
                "rarity_triage": float(rar_score),
                "base_edges": base_ranked[: int(args.k)],
                "rarity_edges": ranked_r[: int(args.k)],
            }
        )

    if not rows:
        raise SystemExit("No labeled graphs found in test-list")

    # Rank (descending triage score)
    base_order = sorted(range(len(rows)), key=lambda i: rows[i]["base_triage"], reverse=True)
    rar_order = sorted(range(len(rows)), key=lambda i: rows[i]["rarity_triage"], reverse=True)
    base_rank = {idx: r + 1 for r, idx in enumerate(base_order)}
    rar_rank = {idx: r + 1 for r, idx in enumerate(rar_order)}

    # Candidate selection: label match + (optional) suspicious ports preference + biggest rank gain.
    include_packages = _read_csv_or_list(args.include_packages, args.include_package_list)
    chosen: list[int] = []
    missing_packages: list[str] = []
    for pkg in include_packages:
        matches = [
            i
            for i, row in enumerate(rows)
            if int(row["y"]) == int(args.require_y) and _matches_package(row["path"], pkg)
        ]
        if not matches:
            missing_packages.append(pkg)
            continue
        matches.sort(key=lambda i: rar_rank[i])
        for idx in matches:
            if idx not in chosen:
                chosen.append(idx)
                break

    cand = []
    for i, row in enumerate(rows):
        if int(row["y"]) != int(args.require_y):
            continue
        if i in chosen:
            continue
        gain = base_rank[i] - rar_rank[i]  # positive means rarity ranks it higher (smaller)
        has_port = _has_suspicious_port(row["rarity_edges"], ports=susp_ports)
        cand.append((has_port, gain, i))

    cand.sort(key=lambda t: (t[0], t[1]), reverse=True)
    remaining = max(0, int(args.topn) - len(chosen))
    chosen.extend(i for _, _, i in cand[:remaining])

    out_lines: list[str] = []
    out_lines.append("## Case studies (auto-generated)")
    out_lines.append("")
    out_lines.append(
        f"Config: k={int(args.k)} rarity_lambda={float(args.rarity_lambda)} idf_cap={float(args.rarity_idf_cap)} "
        f"rarity_etypes={sorted(rarity_etypes) if rarity_etypes else 'ALL'}"
    )
    if include_packages:
        out_lines.append(f"Included packages: {include_packages}")
    if missing_packages:
        out_lines.append(f"Missing requested packages: {missing_packages}")
    out_lines.append("")

    for j, idx in enumerate(chosen, 1):
        row = rows[idx]
        gp: Path = row["path"]
        out_lines.append(
            _emit_case_markdown(
                title=f"Case {j}: {gp.stem}",
                graph_path=gp,
                y=int(row["y"]),
                base_rank=int(base_rank[idx]),
                rarity_rank=int(rar_rank[idx]),
                base_edges=row["base_edges"],
                rarity_edges=row["rarity_edges"],
                k_show=int(args.k_show),
            )
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines))
    print(f"wrote={out_path.as_posix()} cases={len(chosen)}")


if __name__ == "__main__":
    main()

