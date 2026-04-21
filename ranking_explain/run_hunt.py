from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch_geometric.data import HeteroData
import random

# Ensure repo root importable when running as script.
REPO_ROOT = Path(__file__).resolve().parent.parent
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from model.rgcn import RGCNGraphClassifier  # noqa: E402
from ranking_explain.pgexplainer import PGExplainer  # noqa: E402
from ranking_explain.hunt import RankedEdge, topk_edges  # noqa: E402
from ranking_explain.rarity import etype_to_str, load_rarity_stats, normalize_dst_key, str_to_etype  # noqa: E402


def load_one_graph(path: Path) -> HeteroData:
    obj = torch.load(path, map_location="cpu")
    return HeteroData.from_dict(obj["data_dict"])

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


def _filter_ranked_edges(
    ranked: list[RankedEdge],
    *,
    filter_net: bool,
    filter_net_ip: bool,
    filter_domains: list[str],
    filter_system_files: bool,
    system_file_prefixes: list[str],
    filter_tmp_tempfile: bool,
    filter_cmd_noise: bool,
    cmd_noise_keywords: list[str],
    dedup_dst: bool,
    max_per_etype: int,
) -> list[RankedEdge]:
    out: list[RankedEdge] = []
    per_etype: dict[tuple[str, str, str], int] = {}
    seen_dst: set[tuple[tuple[str, str, str], str]] = set()

    doms = {d.strip().lower() for d in filter_domains if d.strip()}
    sys_prefixes = [p.strip() for p in system_file_prefixes if p.strip()]
    cmd_noise = [k.strip().lower() for k in cmd_noise_keywords if k.strip()]
    for r in ranked:
        if filter_net and r.dst_type == "NET":
            dk = str(r.dst_key).lower()
            # Drop common installer noise by domain/host/dns key substring match.
            if any(d in dk for d in doms):
                continue
            # Optionally drop raw IP targets (keep dns/host nodes).
            if filter_net_ip and "ip::" in dk:
                continue

        if filter_cmd_noise and r.dst_type == "CMD":
            dk = str(r.dst_key).lower()
            cmd = dk.split("cmd::", 1)[1] if "cmd::" in dk else dk
            if any(k in cmd for k in cmd_noise):
                continue

        if filter_system_files and r.dst_type == "FILE":
            dk = str(r.dst_key)
            # Our canonical FILE keys often look like "...::file::<path>".
            path = dk.split("::file::", 1)[1] if "::file::" in dk else dk
            # Normalize for cases like "dev/tty" vs "/dev/tty".
            path_norm = path.lstrip("/")
            if filter_tmp_tempfile:
                # Filter only the very common tempfile pattern "tmp/tmp*"
                # while keeping other /tmp/* paths (often attacker-used).
                if path_norm.startswith("tmp/tmp"):
                    continue
            # Drop empty/missing paths (parsing artifacts).
            if path_norm == "":
                continue
            if any(path.startswith(pref) or path_norm.startswith(pref.lstrip("/")) for pref in sys_prefixes):
                continue

        if dedup_dst:
            sig = (r.etype, r.dst_key)
            if sig in seen_dst:
                continue
            seen_dst.add(sig)

        n = per_etype.get(r.etype, 0)
        if max_per_etype > 0 and n >= max_per_etype:
            continue
        per_etype[r.etype] = n + 1

        out.append(r)
    return out


def _tmp_suspicious_bonus(
    r: RankedEdge,
    *,
    enable: bool,
    suffixes: list[str],
    keywords: list[str],
) -> float:
    if not enable or r.dst_type != "FILE":
        return 0.0
    dk = str(r.dst_key)
    path = dk.split("::file::", 1)[1] if "::file::" in dk else dk
    path_norm = path.lstrip("/")
    if not path_norm.startswith("tmp/"):
        return 0.0

    p = path_norm.lower()
    sfx = [s.strip().lower() for s in suffixes if s.strip()]
    kws = [k.strip().lower() for k in keywords if k.strip()]
    bonus = 0.0
    if any(p.endswith(x) for x in sfx):
        bonus += 0.05
    if any(k in p for k in kws):
        bonus += 0.05
    return bonus


def _demotion_penalty(r: RankedEdge, *, demote_load: bool) -> float:
    if not demote_load:
        return 0.0
    # This edge is almost always present; keep it but don't let it dominate top-K.
    if r.etype == ("PKG", "LOAD", "PROC"):
        return 0.2
    return 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", type=str, required=True, help="Path to *.graph.pt")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--backbone-ckpt", type=str, default="", help="Backbone checkpoint (.pt)")
    ap.add_argument("--explainer-ckpt", type=str, default="", help="Explainer checkpoint (.pt)")
    ap.add_argument("--device", type=str, default="cpu", help="cpu|cuda")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for reproducible ranking (0 = no seeding)")
    ap.add_argument("--rarity-stats", type=str, default="", help="Path to benign rarity stats json (build_benign_rarity_stats.py)")
    ap.add_argument("--rarity-lambda", type=float, default=0.0, help="Add lambda * idf(etype,dst_key) to edge score")
    ap.add_argument(
        "--rarity-normalize",
        type=str,
        default="",
        help="Override rarity dst_key normalization (none|osp|qut). Default: use stats' normalize field.",
    )
    ap.add_argument(
        "--rarity-idf-cap",
        type=float,
        default=0.0,
        help="If >0, cap IDF to this value before scaling by lambda (helps OSP long-tail).",
    )
    ap.add_argument(
        "--rarity-etypes",
        type=str,
        default="",
        help="Comma-separated etypes to apply rarity to (format SRC|REL|DST). Empty = all etypes.",
    )
    ap.add_argument(
        "--print-anomaly-score",
        action="store_true",
        default=False,
        help="Print graph-level anomaly score (mean of top-K adjusted edge scores)",
    )
    ap.add_argument("--filter-net", action="store_true", default=False, help="Filter common NET noise (domains)")
    ap.add_argument("--filter-net-ip", action="store_true", default=False, help="Filter NET:ip::* destinations (keep dns/host)")
    ap.add_argument(
        "--filter-domains",
        type=str,
        default="pypi.org,files.pythonhosted.org,pythonhosted.org,registry.npmjs.org,npmjs.org,crates.io,static.crates.io,rubygems.org,files.rubygems.org,github.com,raw.githubusercontent.com",
        help="Comma-separated domain substrings to filter when dst_type==NET",
    )
    ap.add_argument(
        "--filter-system-files",
        action="store_true",
        default=False,
        help="Filter common system/temp FILE paths (prefix match)",
    )
    ap.add_argument(
        "--system-file-prefixes",
        type=str,
        default="/dev/,dev/,pipe:[,host:[,/{dev=,{dev=",
        help="Comma-separated FILE path prefixes to filter (e.g. /dev/,pipe:[,/tmp/tmp)",
    )
    ap.add_argument(
        "--filter-tmp-tempfile",
        action="store_true",
        default=False,
        help="Filter FILE paths matching tmp/tmp* (tempfile noise) while keeping other /tmp/* paths",
    )
    ap.add_argument(
        "--filter-cmd-noise",
        action="store_true",
        default=False,
        help="Filter common install/environment-probe CMDs (substring match)",
    )
    ap.add_argument(
        "--cmd-noise-keywords",
        type=str,
        default="sleep,sleep30m,uname,dpkg-query,lsb_release,pipinstall,pip install,rustc,--version,analyze-python.py,analyze-node.js",
        help="Comma-separated CMD substrings to filter when --filter-cmd-noise is set",
    )
    ap.add_argument(
        "--rerank-tmp-suspicious",
        action="store_true",
        default=False,
        help="Rerank /tmp FILE edges upward if they look like scripts/binaries (suffix/keyword heuristics)",
    )
    ap.add_argument(
        "--tmp-suspicious-suffixes",
        type=str,
        default=".sh,.py,.so,.bin,.exe,.elf",
        help="Comma-separated suffixes treated as suspicious under /tmp",
    )
    ap.add_argument(
        "--tmp-suspicious-keywords",
        type=str,
        default="payload,drop,stage,inject,miner,ssh,key,token,secret",
        help="Comma-separated keywords treated as suspicious under /tmp",
    )
    ap.add_argument(
        "--demote-load",
        action="store_true",
        default=False,
        help="Demote ubiquitous PKG->LOAD->PROC edge in ranking (keep but lower priority)",
    )
    ap.add_argument("--dedup-dst", action="store_true", default=False, help="Deduplicate by (etype,dst_key)")
    ap.add_argument(
        "--max-per-etype",
        type=int,
        default=0,
        help="If >0, keep at most this many edges per etype (after filtering)",
    )
    args = ap.parse_args()

    p = Path(args.graph)
    if int(args.seed) != 0:
        s = int(args.seed)
        random.seed(s)
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # Older torch versions may not support this; seeding is still helpful.
            pass
    data = load_one_graph(p)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    data = data.to(device)

    backbone = (
        _load_backbone(args.backbone_ckpt, device=device)
        if args.backbone_ckpt
        else RGCNGraphClassifier(hidden_dim=64, num_layers=2, dropout=0.2, num_classes=2).to(device)
    )
    explainer = _load_explainer(args.explainer_ckpt, device=device) if args.explainer_ckpt else PGExplainer(hidden_dim=64).to(device)

    rarity = load_rarity_stats(args.rarity_stats) if args.rarity_stats else None

    # Pull a larger candidate set, then filter down to k.
    ranked = topk_edges(backbone=backbone, explainer=explainer, hetero_batch=data, k=max(args.k * 20, args.k))
    if rarity is not None and float(args.rarity_lambda) != 0.0:
        lam = float(args.rarity_lambda)
        cap = float(args.rarity_idf_cap)
        scheme = str(args.rarity_normalize).strip() or str(getattr(rarity, "normalize", "none"))
        allowed = {s.strip() for s in str(args.rarity_etypes).split(",") if s.strip()}
        ranked = [
            RankedEdge(
                graph_id=r.graph_id,
                score=float(
                    r.score
                    + (
                        lam
                        * (
                            min(
                                rarity.idf(
                                    etype=r.etype,
                                    dst_key=normalize_dst_key(
                                        scheme=scheme, etype=r.etype, dst_type=r.dst_type, dst_key=str(r.dst_key)
                                    ),
                                ),
                                cap,
                            )
                            if cap > 0.0
                            else rarity.idf(
                                etype=r.etype,
                                dst_key=normalize_dst_key(
                                    scheme=scheme, etype=r.etype, dst_type=r.dst_type, dst_key=str(r.dst_key)
                                ),
                            )
                        )
                        if (not allowed or etype_to_str(r.etype) in allowed)
                        else 0.0
                    )
                ),
                etype=r.etype,
                src_type=r.src_type,
                src_key=r.src_key,
                dst_type=r.dst_type,
                dst_key=r.dst_key,
            )
            for r in ranked
        ]
        ranked.sort(key=lambda x: x.score, reverse=True)
    ranked = _filter_ranked_edges(
        ranked,
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
    if args.rerank_tmp_suspicious:
        sfx = [s.strip() for s in str(args.tmp_suspicious_suffixes).split(",") if s.strip()]
        kws = [s.strip() for s in str(args.tmp_suspicious_keywords).split(",") if s.strip()]
        ranked.sort(
            key=lambda r: (
                r.score
                + _tmp_suspicious_bonus(r, enable=True, suffixes=sfx, keywords=kws)
                - _demotion_penalty(r, demote_load=bool(args.demote_load))
            ),
            reverse=True,
        )
    elif args.demote_load:
        ranked.sort(key=lambda r: r.score - _demotion_penalty(r, demote_load=True), reverse=True)
    ranked = ranked[: args.k]

    print(f"graph={p.name} topk={len(ranked)}")
    if args.print_anomaly_score and ranked:
        # A simple, calibrated-by-benign baseline: mean of top-K adjusted scores.
        anom = sum(r.score for r in ranked) / float(len(ranked))
        print(f"anomaly_score={anom:.6f} (mean_topk_adjusted)")
    for r in ranked:
        print(f"{r.score:.4f}\t{r.etype}\t{r.src_type}:{r.src_key} -> {r.dst_type}:{r.dst_key}")


if __name__ == "__main__":
    main()

