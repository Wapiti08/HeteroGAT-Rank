from __future__ import annotations

import argparse
import itertools
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class RunSpec:
    rarity_etypes: str
    rarity_lambda: float
    rarity_idf_cap: float


_RE_AUC = re.compile(r"^base_auroc=(?P<base_auc>[-+.0-9nan]+)\s+base_auprc=(?P<base_auprc>[-+.0-9nan]+)\s*$")
_RE_RAR = re.compile(r"^rarity_auroc=(?P<rar_auc>[-+.0-9nan]+)\s+rarity_auprc=(?P<rar_auprc>[-+.0-9nan]+)\s*$")
_RE_OP = re.compile(
    r"^(?P<which>base|rarity)\s+fpr<=\s+(?P<tgt>[0-9.]+)@val\s+thr=(?P<thr>[-+.0-9eEinf]+)\s+"
    r"prec=(?P<prec>[-+.0-9nan]+)\s+rec=(?P<rec>[-+.0-9nan]+)\s+f1=(?P<f1>[-+.0-9nan]+)\s+"
    r"tp=(?P<tp>[-+.0-9]+)\s+fp=(?P<fp>[-+.0-9]+)\s+tn=(?P<tn>[-+.0-9]+)\s+fn=(?P<fn>[-+.0-9]+)\s+"
    r"fpr=(?P<fpr>[-+.0-9nan]+)\s*$"
)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _now_ms() -> int:
    return int(time.time() * 1000)


def _run_eval(cmd: list[str], *, timeout_s: Optional[int] = None) -> tuple[int, str]:
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    return int(p.returncode), str(p.stdout)


def _parse_metrics(stdout: str) -> dict[str, Any]:
    base_auc = base_auprc = rar_auc = rar_auprc = None
    ops: dict[str, dict[str, dict[str, float]]] = {"base": {}, "rarity": {}}
    for line in stdout.splitlines():
        m = _RE_AUC.match(line.strip())
        if m:
            base_auc = m.group("base_auc")
            base_auprc = m.group("base_auprc")
            continue
        m = _RE_RAR.match(line.strip())
        if m:
            rar_auc = m.group("rar_auc")
            rar_auprc = m.group("rar_auprc")
            continue
        m = _RE_OP.match(line.strip())
        if m:
            which = m.group("which")
            tgt = m.group("tgt")
            ops[which][tgt] = {
                "thr": float(m.group("thr")) if m.group("thr") not in ("inf", "+inf", "-inf") else float("inf"),
                "prec": float(m.group("prec")) if m.group("prec") != "nan" else float("nan"),
                "rec": float(m.group("rec")) if m.group("rec") != "nan" else float("nan"),
                "f1": float(m.group("f1")) if m.group("f1") != "nan" else float("nan"),
                "tp": float(m.group("tp")),
                "fp": float(m.group("fp")),
                "tn": float(m.group("tn")),
                "fn": float(m.group("fn")),
                "fpr": float(m.group("fpr")) if m.group("fpr") != "nan" else float("nan"),
            }
    return {
        "base_auroc": base_auc,
        "base_auprc": base_auprc,
        "rarity_auroc": rar_auc,
        "rarity_auprc": rar_auprc,
        "operating_points": ops,
    }


def _iter_specs(
    *,
    etypes: Sequence[str],
    lambdas: Sequence[float],
    idf_caps: Sequence[float],
) -> list[RunSpec]:
    out: list[RunSpec] = []
    for e, lam, cap in itertools.product(etypes, lambdas, idf_caps):
        out.append(RunSpec(rarity_etypes=str(e), rarity_lambda=float(lam), rarity_idf_cap=float(cap)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", type=str, default="", help="Run name tag (used in output filenames)")
    ap.add_argument("--out-dir", type=str, required=True, help="Write TSV + logs here")

    # Split lists
    ap.add_argument("--val-list", type=str, required=True)
    ap.add_argument("--test-list", type=str, required=True)

    # Model artifacts
    ap.add_argument("--backbone-ckpt", type=str, required=True)
    ap.add_argument("--explainer-ckpt", type=str, required=True)
    ap.add_argument("--rarity-stats", type=str, required=True)
    ap.add_argument("--rarity-normalize", type=str, default="")

    # Hunt settings
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--device", type=str, default="cpu")

    # Filters (pass-through; keep as a single string blob for convenience)
    ap.add_argument(
        "--filter-args",
        type=str,
        default="--filter-net --filter-net-ip --filter-system-files --filter-tmp-tempfile --filter-cmd-noise --dedup-dst --max-per-etype 8",
        help="Extra args appended to eval_hunt_rarity.py (quoted string).",
    )

    # Grid
    ap.add_argument(
        "--etypes",
        type=str,
        default="NET_ONLY;CONNECT_ONLY;ALL",
        help="Semicolon-separated presets: NET_ONLY, CONNECT_ONLY, ALL, or a raw comma-separated etype list.",
    )
    ap.add_argument("--lambdas", type=str, default="0.3", help="Comma-separated floats")
    ap.add_argument("--idf-caps", type=str, default="2", help="Comma-separated floats")
    ap.add_argument("--op-fprs", type=str, default="0.01,0.05,0.10", help="Operating points (passed to eval script)")

    ap.add_argument("--timeout-s", type=int, default=0, help="Per-run timeout (0 = no timeout)")
    ap.add_argument("--dry-run", action="store_true", default=False)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.name.strip() or f"grid_{_now_ms()}"

    # Presets
    preset_map = {
        "NET_ONLY": "PROC|DNS_QUERY|NET,PROC|RESOLVE|NET,PROC|CONNECT|NET",
        "CONNECT_ONLY": "PROC|CONNECT|NET",
        # Empty string means "all etypes" in eval_hunt_rarity.py
        "ALL": "",
        # A minimal mixed preset (useful sanity)
        "NET_AND_CMD": "PROC|DNS_QUERY|NET,PROC|RESOLVE|NET,PROC|CONNECT|NET,PROC|EXEC|CMD",
    }
    etype_tokens = [t.strip() for t in str(args.etypes).split(";") if t.strip()]
    etypes: list[str] = []
    for tok in etype_tokens:
        etypes.append(preset_map.get(tok, tok))

    lambdas = [float(x) for x in str(args.lambdas).split(",") if x.strip()]
    idf_caps = [float(x) for x in str(args.idf_caps).split(",") if x.strip()]
    specs = _iter_specs(etypes=etypes, lambdas=lambdas, idf_caps=idf_caps)

    tsv_path = out_dir / f"{tag}.tsv"
    log_path = out_dir / f"{tag}.log.txt"
    meta_path = out_dir / f"{tag}.meta.json"

    _ensure_parent(tsv_path)
    _ensure_parent(log_path)
    _ensure_parent(meta_path)

    meta = {
        "tag": tag,
        "created_ms": _now_ms(),
        "val_list": str(args.val_list),
        "test_list": str(args.test_list),
        "backbone_ckpt": str(args.backbone_ckpt),
        "explainer_ckpt": str(args.explainer_ckpt),
        "rarity_stats": str(args.rarity_stats),
        "rarity_normalize": str(args.rarity_normalize),
        "k": int(args.k),
        "device": str(args.device),
        "filter_args": str(args.filter_args),
        "op_fprs": str(args.op_fprs),
        "grid": {"etypes": etype_tokens, "lambdas": lambdas, "idf_caps": idf_caps},
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    header_cols = [
        "rarity_etypes",
        "rarity_lambda",
        "rarity_idf_cap",
        "base_auroc",
        "base_auprc",
        "rarity_auroc",
        "rarity_auprc",
        # operating points: base/rarity @ 0.01/0.05/0.10
        "base_thr@0.01",
        "base_prec@0.01",
        "base_rec@0.01",
        "base_f1@0.01",
        "rarity_thr@0.01",
        "rarity_prec@0.01",
        "rarity_rec@0.01",
        "rarity_f1@0.01",
        "base_thr@0.05",
        "base_prec@0.05",
        "base_rec@0.05",
        "base_f1@0.05",
        "rarity_thr@0.05",
        "rarity_prec@0.05",
        "rarity_rec@0.05",
        "rarity_f1@0.05",
        "base_thr@0.10",
        "base_prec@0.10",
        "base_rec@0.10",
        "base_f1@0.10",
        "rarity_thr@0.10",
        "rarity_prec@0.10",
        "rarity_rec@0.10",
        "rarity_f1@0.10",
        "returncode",
    ]
    tsv_path.write_text("\t".join(header_cols) + "\n")

    def fmt(x: Any) -> str:
        if x is None:
            return ""
        if isinstance(x, float):
            if x != x:  # nan
                return "nan"
            if x == float("inf"):
                return "inf"
        return str(x)

    # Runs
    timeout_s = int(args.timeout_s) if int(args.timeout_s) > 0 else None
    filter_args = [x for x in str(args.filter_args).split(" ") if x.strip()]

    with log_path.open("w", encoding="utf-8") as flog:
        for i, spec in enumerate(specs):
            cmd = [
                sys.executable,
                "scripts/eval_hunt_rarity.py",
                "--val-list",
                str(args.val_list),
                "--test-list",
                str(args.test_list),
                "--backbone-ckpt",
                str(args.backbone_ckpt),
                "--explainer-ckpt",
                str(args.explainer_ckpt),
                "--rarity-stats",
                str(args.rarity_stats),
                "--rarity-normalize",
                str(args.rarity_normalize),
                "--rarity-etypes",
                spec.rarity_etypes,
                "--rarity-lambda",
                str(spec.rarity_lambda),
                "--rarity-idf-cap",
                str(spec.rarity_idf_cap),
                "--k",
                str(int(args.k)),
                "--device",
                str(args.device),
                "--op-fprs",
                str(args.op_fprs),
            ] + filter_args

            flog.write(f"\n## RUN {i+1}/{len(specs)}\n")
            flog.write("CMD: " + " ".join(cmd) + "\n")
            flog.flush()

            if args.dry_run:
                continue

            t0 = time.perf_counter()
            rc, out = _run_eval(cmd, timeout_s=timeout_s)
            dt = time.perf_counter() - t0

            flog.write(f"returncode={rc} elapsed_s={dt:.3f}\n")
            flog.write(out)
            flog.write("\n")
            flog.flush()

            m = _parse_metrics(out)

            def opv(which: str, tgt: str, key: str) -> Any:
                try:
                    return m["operating_points"][which][tgt][key]
                except Exception:
                    return None

            row: list[Any] = [
                spec.rarity_etypes,
                spec.rarity_lambda,
                spec.rarity_idf_cap,
                m["base_auroc"],
                m["base_auprc"],
                m["rarity_auroc"],
                m["rarity_auprc"],
                opv("base", "0.01", "thr"),
                opv("base", "0.01", "prec"),
                opv("base", "0.01", "rec"),
                opv("base", "0.01", "f1"),
                opv("rarity", "0.01", "thr"),
                opv("rarity", "0.01", "prec"),
                opv("rarity", "0.01", "rec"),
                opv("rarity", "0.01", "f1"),
                opv("base", "0.05", "thr"),
                opv("base", "0.05", "prec"),
                opv("base", "0.05", "rec"),
                opv("base", "0.05", "f1"),
                opv("rarity", "0.05", "thr"),
                opv("rarity", "0.05", "prec"),
                opv("rarity", "0.05", "rec"),
                opv("rarity", "0.05", "f1"),
                opv("base", "0.10", "thr"),
                opv("base", "0.10", "prec"),
                opv("base", "0.10", "rec"),
                opv("base", "0.10", "f1"),
                opv("rarity", "0.10", "thr"),
                opv("rarity", "0.10", "prec"),
                opv("rarity", "0.10", "rec"),
                opv("rarity", "0.10", "f1"),
                rc,
            ]

            with tsv_path.open("a", encoding="utf-8") as f:
                f.write("\t".join(fmt(x) for x in row) + "\n")

    print(f"WROTE {tsv_path.as_posix()}")
    print(f"WROTE {log_path.as_posix()}")
    print(f"WROTE {meta_path.as_posix()}")


if __name__ == "__main__":
    main()

