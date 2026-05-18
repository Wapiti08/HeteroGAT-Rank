from __future__ import annotations

import argparse
import csv
import json
import random
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Set

import torch

# Ensure repo root importable when running as script.
REPO_ROOT = Path(__file__).resolve().parent.parent
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from ranking_explain.hunt import RankedEdge, topk_edges  # noqa: E402
from ranking_explain.rarity import RarityStats, etype_to_str, load_rarity_stats, normalize_dst_key  # noqa: E402
from ranking_explain.run_hunt import _filter_ranked_edges  # noqa: E402
from scripts.eval_hunt_rarity import _get_y, _load_backbone, _load_explainer, _load_graph  # noqa: E402


EVIDENCE_LEVELS = ("IOC", "TTP", "contextual", "telemetry_gap")
METHODS = ("raw_explainer", "filtering", "filtering_rarity")


DEFAULT_FILTER_DOMAINS = (
    "pypi.org,files.pythonhosted.org,pythonhosted.org,registry.npmjs.org,npmjs.org,"
    "crates.io,static.crates.io,rubygems.org,files.rubygems.org,github.com,raw.githubusercontent.com"
)
DEFAULT_SYSTEM_FILE_PREFIXES = "/dev/,dev/,pipe:[,host:[,/{dev=,{dev="
DEFAULT_CMD_NOISE_KEYWORDS = (
    "sleep,sleep30m,uname,dpkg-query,lsb_release,pipinstall,pip install,rustc,--version,"
    "analyze-python.py,analyze-node.js"
)


@dataclass(frozen=True)
class FilterConfig:
    filter_net: bool = False
    filter_net_ip: bool = False
    filter_domains: tuple[str, ...] = tuple(s for s in DEFAULT_FILTER_DOMAINS.split(",") if s)
    filter_system_files: bool = False
    system_file_prefixes: tuple[str, ...] = tuple(s for s in DEFAULT_SYSTEM_FILE_PREFIXES.split(",") if s)
    filter_tmp_tempfile: bool = False
    filter_cmd_noise: bool = False
    cmd_noise_keywords: tuple[str, ...] = tuple(s for s in DEFAULT_CMD_NOISE_KEYWORDS.split(",") if s)
    dedup_dst: bool = False
    max_per_etype: int = 0


def _read_list(path: str | Path | None) -> list[str]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p.as_posix())
    return [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]


def _as_set_csv(value: str) -> Set[str]:
    return {x.strip() for x in str(value).split(",") if x.strip()}


def _csv_tuple(value: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in str(value).split(",") if x.strip())


def _graph_id(path: Path) -> str:
    suffix = ".graph.pt"
    return path.name[: -len(suffix)] if path.name.endswith(suffix) else path.stem


def _format_node(node_type: str, node_key: str) -> str:
    return f"{node_type}:{node_key}"


def _short_text(value: object, *, limit: int = 240) -> str:
    text = str(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _method_label(method: str) -> str:
    labels = {
        "raw_explainer": "Raw explainer",
        "filtering": "Filtering",
        "filtering_rarity": "Filtering + rarity",
    }
    return labels.get(method, method)


def _suggest_evidence_level(edge: RankedEdge) -> tuple[str, str]:
    etype = etype_to_str(edge.etype)
    dst_key = str(edge.dst_key).lower()

    if edge.dst_type == "NET":
        return "IOC", "Network destination can be pivoted as an indicator."
    if edge.dst_type == "CMD":
        return "TTP", "Command execution reflects attacker behavior."
    if edge.dst_type == "SYSCALL":
        return "TTP", "System-call edge reflects behavioral technique."
    if edge.etype[1] == "EXEC":
        return "TTP", "Process execution edge reflects behavioral technique."
    if edge.dst_type == "FILE":
        if any(token in dst_key for token in ("tmp/", "/tmp/", "payload", "token", "secret", "ssh", "key")):
            return "contextual", "File path offers investigation context."
        return "contextual", "File artifact is supporting context."
    if edge.etype == ("PKG", "LOAD", "PROC"):
        return "telemetry_gap", "Load edge is generic and usually not directly actionable."
    return "contextual", f"{etype} edge provides supporting context."


def _suspicious_edges(
    ranked: list[RankedEdge],
    *,
    rarity: Optional[RarityStats],
    rarity_lambda: float,
    rarity_idf_cap: float,
    rarity_etypes: Set[str],
    rarity_normalize: str,
) -> list[RankedEdge]:
    scheme = rarity_normalize.strip() or str(getattr(rarity, "normalize", "none"))
    out: list[RankedEdge] = []
    for r in ranked:
        bonus = 0.0
        if rarity is not None and rarity_lambda != 0.0 and (not rarity_etypes or etype_to_str(r.etype) in rarity_etypes):
            norm_key = normalize_dst_key(scheme=scheme, etype=r.etype, dst_type=r.dst_type, dst_key=str(r.dst_key))
            idf = float(rarity.idf(etype=r.etype, dst_key=norm_key))
            if rarity_idf_cap > 0.0:
                idf = min(idf, rarity_idf_cap)
            bonus = rarity_lambda * idf
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
    out.sort(key=lambda edge: edge.score, reverse=True)
    return out


def _apply_filters(ranked: list[RankedEdge], cfg: FilterConfig) -> list[RankedEdge]:
    return _filter_ranked_edges(
        ranked,
        filter_net=cfg.filter_net,
        filter_net_ip=cfg.filter_net_ip,
        filter_domains=list(cfg.filter_domains),
        filter_system_files=cfg.filter_system_files,
        system_file_prefixes=list(cfg.system_file_prefixes),
        filter_tmp_tempfile=cfg.filter_tmp_tempfile,
        filter_cmd_noise=cfg.filter_cmd_noise,
        cmd_noise_keywords=list(cfg.cmd_noise_keywords),
        dedup_dst=cfg.dedup_dst,
        max_per_etype=cfg.max_per_etype,
    )


def _filter_config_from_text(filter_args: str) -> FilterConfig:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--filter-net", action="store_true", default=False)
    parser.add_argument("--filter-net-ip", action="store_true", default=False)
    parser.add_argument("--filter-domains", type=str, default=DEFAULT_FILTER_DOMAINS)
    parser.add_argument("--filter-system-files", action="store_true", default=False)
    parser.add_argument("--system-file-prefixes", type=str, default=DEFAULT_SYSTEM_FILE_PREFIXES)
    parser.add_argument("--filter-tmp-tempfile", action="store_true", default=False)
    parser.add_argument("--filter-cmd-noise", action="store_true", default=False)
    parser.add_argument("--cmd-noise-keywords", type=str, default=DEFAULT_CMD_NOISE_KEYWORDS)
    parser.add_argument("--dedup-dst", action="store_true", default=False)
    parser.add_argument("--max-per-etype", type=int, default=0)
    parsed, unknown = parser.parse_known_args(shlex.split(filter_args or ""))
    if unknown:
        raise ValueError(f"Unsupported filter args in meta: {' '.join(unknown)}")
    return FilterConfig(
        filter_net=bool(parsed.filter_net),
        filter_net_ip=bool(parsed.filter_net_ip),
        filter_domains=_csv_tuple(parsed.filter_domains),
        filter_system_files=bool(parsed.filter_system_files),
        system_file_prefixes=_csv_tuple(parsed.system_file_prefixes),
        filter_tmp_tempfile=bool(parsed.filter_tmp_tempfile),
        filter_cmd_noise=bool(parsed.filter_cmd_noise),
        cmd_noise_keywords=_csv_tuple(parsed.cmd_noise_keywords),
        dedup_dst=bool(parsed.dedup_dst),
        max_per_etype=int(parsed.max_per_etype),
    )


def _sample_positive_paths(paths: Sequence[Path], *, n: int, seed: int) -> list[Path]:
    positives: list[Path] = []
    for path in paths:
        y = _get_y(path)
        if y is not None and int(y) == 1:
            positives.append(path)
    positives = sorted(positives, key=lambda p: p.as_posix())
    rng = random.Random(int(seed))
    if len(positives) <= int(n):
        return positives
    return sorted(rng.sample(positives, int(n)), key=lambda p: p.as_posix())


def _load_meta(path: str | Path) -> dict[str, object]:
    meta_path = Path(path)
    if not meta_path.exists():
        raise FileNotFoundError(meta_path.as_posix())
    return json.loads(meta_path.read_text())


def _generate_template(args: argparse.Namespace) -> None:
    meta = _load_meta(args.meta) if args.meta else {}
    dataset = args.dataset or str(meta.get("tag", "")).replace("_hunt_rarity", "") or "dataset"
    test_list = args.test_list or str(meta.get("test_list", ""))
    backbone_ckpt = args.backbone_ckpt or str(meta.get("backbone_ckpt", ""))
    explainer_ckpt = args.explainer_ckpt or str(meta.get("explainer_ckpt", ""))
    rarity_stats_path = args.rarity_stats or str(meta.get("rarity_stats", ""))
    rarity_normalize = args.rarity_normalize or str(meta.get("rarity_normalize", ""))
    filter_args = args.filter_args if args.filter_args else str(meta.get("filter_args", ""))
    filter_cfg = _filter_config_from_text(filter_args)

    if not test_list or not backbone_ckpt or not explainer_ckpt or not rarity_stats_path:
        raise SystemExit("--test-list, --backbone-ckpt, --explainer-ckpt, and --rarity-stats are required unless --meta provides them")

    graph_paths = [Path(x) for x in _read_list(test_list)]
    sampled = _sample_positive_paths(graph_paths, n=int(args.sample_size), seed=int(args.seed))
    if not sampled:
        raise SystemExit("No malicious graphs found to sample")

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    backbone = _load_backbone(backbone_ckpt, device=device)
    explainer = _load_explainer(explainer_ckpt, device=device)
    rarity = load_rarity_stats(rarity_stats_path)
    rarity_etypes = _as_set_csv(args.rarity_etypes)

    rows: list[dict[str, object]] = []
    for graph_path in sampled:
        data = _load_graph(graph_path).to(device)
        pg_ranked = topk_edges(backbone=backbone, explainer=explainer, hetero_batch=data, k=1_000_000)
        raw_ranked = _suspicious_edges(
            pg_ranked,
            rarity=None,
            rarity_lambda=0.0,
            rarity_idf_cap=0.0,
            rarity_etypes=set(),
            rarity_normalize="",
        )
        rarity_ranked = _suspicious_edges(
            pg_ranked,
            rarity=rarity,
            rarity_lambda=float(args.rarity_lambda),
            rarity_idf_cap=float(args.rarity_idf_cap),
            rarity_etypes=rarity_etypes,
            rarity_normalize=rarity_normalize,
        )
        method_edges = {
            "raw_explainer": raw_ranked,
            "filtering": _apply_filters(raw_ranked, filter_cfg),
            "filtering_rarity": _apply_filters(rarity_ranked, filter_cfg),
        }
        for method in METHODS:
            for rank, edge in enumerate(method_edges[method][: int(args.top_k)], start=1):
                suggested, rationale = _suggest_evidence_level(edge)
                rows.append(
                    {
                        "dataset": dataset,
                        "package": _graph_id(graph_path),
                        "graph_path": graph_path.as_posix(),
                        "method": method,
                        "method_label": _method_label(method),
                        "rank": rank,
                        "score": f"{float(edge.score):.8g}",
                        "etype": etype_to_str(edge.etype),
                        "src": _short_text(_format_node(edge.src_type, edge.src_key)),
                        "dst": _short_text(_format_node(edge.dst_type, edge.dst_key)),
                        "suggested_level": suggested,
                        "suggested_rationale": rationale,
                        "evidence_level": "",
                        "annotator_note": "",
                    }
                )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "package",
        "graph_path",
        "method",
        "method_label",
        "rank",
        "score",
        "etype",
        "src",
        "dst",
        "suggested_level",
        "suggested_rationale",
        "evidence_level",
        "annotator_note",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote={out_path.as_posix()} packages={len(sampled)} rows={len(rows)}")


def _normalize_evidence_level(value: str) -> str:
    level = str(value).strip()
    aliases = {
        "ioc": "IOC",
        "ttp": "TTP",
        "context": "contextual",
        "contextual": "contextual",
        "gap": "telemetry_gap",
        "telemetry": "telemetry_gap",
        "telemetry_gap": "telemetry_gap",
    }
    return aliases.get(level.lower(), level)


def _summarize_annotations(args: argparse.Namespace) -> None:
    in_path = Path(args.annotations_in)
    if not in_path.exists():
        raise FileNotFoundError(in_path.as_posix())

    counts: dict[tuple[str, str, str], int] = {}
    totals: dict[tuple[str, str], int] = {}
    actionable: dict[tuple[str, str], int] = {}
    with in_path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            dataset = str(row.get("dataset", "")).strip() or "dataset"
            method = str(row.get("method", "")).strip()
            level = _normalize_evidence_level(str(row.get("evidence_level", "")))
            if not method or not level:
                continue
            if level not in EVIDENCE_LEVELS:
                raise ValueError(f"Unknown evidence_level={level!r}; expected one of {EVIDENCE_LEVELS}")
            key = (dataset, method)
            totals[key] = totals.get(key, 0) + 1
            counts[(dataset, method, level)] = counts.get((dataset, method, level), 0) + 1
            if level in {"IOC", "TTP", "contextual"}:
                actionable[key] = actionable.get(key, 0) + 1

    rows: list[dict[str, object]] = []
    for dataset, method in sorted(totals):
        total = totals[(dataset, method)]
        for level in EVIDENCE_LEVELS:
            count = counts.get((dataset, method, level), 0)
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "method_label": _method_label(method),
                    "evidence_level": level,
                    "count": count,
                    "total_labeled": total,
                    "proportion": f"{count / float(total):.6f}" if total else "nan",
                    "actionable_count": actionable.get((dataset, method), 0),
                    "actionable_proportion": f"{actionable.get((dataset, method), 0) / float(total):.6f}" if total else "nan",
                }
            )

    out_path = Path(args.summary_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "method",
        "method_label",
        "evidence_level",
        "count",
        "total_labeled",
        "proportion",
        "actionable_count",
        "actionable_proportion",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote={out_path.as_posix()} groups={len(totals)} rows={len(rows)}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate and summarize analyst-usefulness evidence-level annotations for top-K pivots."
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    gen = sub.add_parser("generate", help="Generate TSV rows for manual evidence-level annotation")
    gen.add_argument("--meta", type=str, default="", help="Optional ablation meta JSON with paths/filter args")
    gen.add_argument("--dataset", type=str, default="", help="Dataset label used in output rows")
    gen.add_argument("--test-list", type=str, default="")
    gen.add_argument("--backbone-ckpt", type=str, default="")
    gen.add_argument("--explainer-ckpt", type=str, default="")
    gen.add_argument("--rarity-stats", type=str, default="")
    gen.add_argument("--rarity-normalize", type=str, default="")
    gen.add_argument("--device", type=str, default="cpu")
    gen.add_argument("--sample-size", type=int, default=50)
    gen.add_argument("--top-k", type=int, default=5)
    gen.add_argument("--seed", type=int, default=0)
    gen.add_argument("--rarity-lambda", type=float, default=0.3)
    gen.add_argument("--rarity-idf-cap", type=float, default=2.0)
    gen.add_argument("--rarity-etypes", type=str, default="PROC|DNS_QUERY|NET,PROC|RESOLVE|NET,PROC|CONNECT|NET")
    gen.add_argument(
        "--filter-args",
        type=str,
        default="",
        help="Optional run_hunt-style filter flags; overrides filter_args from --meta",
    )
    gen.add_argument("--out", type=str, required=True)

    summ = sub.add_parser("summarize", help="Summarize manually labeled evidence-level TSV")
    summ.add_argument("--annotations-in", type=str, required=True)
    summ.add_argument("--summary-out", type=str, required=True)

    args = ap.parse_args()
    if args.cmd == "generate":
        _generate_template(args)
    elif args.cmd == "summarize":
        _summarize_annotations(args)
    else:
        raise SystemExit(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
