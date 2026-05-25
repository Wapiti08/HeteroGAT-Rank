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
from typing import Any, Optional, Sequence


@dataclass(frozen=True)
class RunSpec:
    sparsity_coef: float
    entropy_coef: float


_RE_EPOCH_LOSS = re.compile(r"^epoch\s+(?P<epoch>\d+)\s+explainer_loss=(?P<loss>[-+.0-9eE]+)\s*$")
_RE_AUC = re.compile(r"^base_auroc=(?P<base_auc>[-+.0-9nan]+)\s+base_auprc=(?P<base_auprc>[-+.0-9nan]+)\s*$")
_RE_RAR = re.compile(r"^rarity_auroc=(?P<rar_auc>[-+.0-9nan]+)\s+rarity_auprc=(?P<rar_auprc>[-+.0-9nan]+)\s*$")
_RE_OP = re.compile(
    r"^(?P<which>base|rarity)\s+fpr<=\s+(?P<tgt>[0-9.]+)@val\s+thr=(?P<thr>[-+.0-9eEinf]+)\s+"
    r"prec=(?P<prec>[-+.0-9nan]+)\s+rec=(?P<rec>[-+.0-9nan]+)\s+f1=(?P<f1>[-+.0-9nan]+)\s+"
    r"tp=(?P<tp>[-+.0-9]+)\s+fp=(?P<fp>[-+.0-9]+)\s+tn=(?P<tn>[-+.0-9]+)\s+fn=(?P<fn>[-+.0-9]+)\s+"
    r"fpr=(?P<fpr>[-+.0-9nan]+)\s*$"
)


def _now_ms() -> int:
    return int(time.time() * 1000)


def _parse_float_list(value: str) -> list[float]:
    return [float(x) for x in str(value).split(",") if x.strip()]


def _run(cmd: list[str], *, timeout_s: Optional[int] = None) -> tuple[int, str]:
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    return int(p.returncode), str(p.stdout)


def _check_python_env(python_bin: str, *, timeout_s: Optional[int] = None) -> tuple[bool, str]:
    cmd = [python_bin, "-c", "import torch; import torch_geometric; print('python_ok')"]
    rc, out = _run(cmd, timeout_s=timeout_s)
    return rc == 0, out


def _last_epoch_loss(stdout: str) -> Optional[float]:
    last: Optional[float] = None
    for line in stdout.splitlines():
        m = _RE_EPOCH_LOSS.match(line.strip())
        if m:
            last = float(m.group("loss"))
    return last


def _to_float(value: str) -> float:
    if value == "nan":
        return float("nan")
    if value in ("inf", "+inf"):
        return float("inf")
    if value == "-inf":
        return float("-inf")
    return float(value)


def _parse_eval_metrics(stdout: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "base_auroc": None,
        "base_auprc": None,
        "rarity_auroc": None,
        "rarity_auprc": None,
        "operating_points": {"base": {}, "rarity": {}},
    }
    for line in stdout.splitlines():
        stripped = line.strip()
        m = _RE_AUC.match(stripped)
        if m:
            metrics["base_auroc"] = _to_float(m.group("base_auc"))
            metrics["base_auprc"] = _to_float(m.group("base_auprc"))
            continue
        m = _RE_RAR.match(stripped)
        if m:
            metrics["rarity_auroc"] = _to_float(m.group("rar_auc"))
            metrics["rarity_auprc"] = _to_float(m.group("rar_auprc"))
            continue
        m = _RE_OP.match(stripped)
        if m:
            which = m.group("which")
            tgt = m.group("tgt")
            metrics["operating_points"][which][tgt] = {
                "thr": _to_float(m.group("thr")),
                "prec": _to_float(m.group("prec")),
                "rec": _to_float(m.group("rec")),
                "f1": _to_float(m.group("f1")),
                "fpr": _to_float(m.group("fpr")),
            }
    return metrics


def _op_metric(metrics: dict[str, Any], which: str, target: str, key: str) -> Any:
    table = metrics.get("operating_points", {}).get(which, {})
    if target in table:
        return table[target].get(key)
    try:
        target_f = float(target)
    except ValueError:
        return None
    best_key = None
    best_delta = None
    for k in table:
        try:
            delta = abs(float(k) - target_f)
        except ValueError:
            continue
        if best_delta is None or delta < best_delta:
            best_delta = delta
            best_key = k
    if best_key is None or best_delta is None or best_delta > 1e-9:
        return None
    return table[best_key].get(key)


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if value != value:
            return "nan"
        if value == float("inf"):
            return "inf"
        if value == float("-inf"):
            return "-inf"
    return str(value)


def _iter_specs(sparsity_coefs: Sequence[float], entropy_coefs: Sequence[float]) -> list[RunSpec]:
    return [RunSpec(float(sp), float(ent)) for sp, ent in itertools.product(sparsity_coefs, entropy_coefs)]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train PGExplainer over a small lambda_sp/lambda_ent grid and evaluate hunt metrics."
    )
    ap.add_argument("--name", type=str, default="", help="Run name tag used for output filenames")
    ap.add_argument("--checkpoint-tag", type=str, default="", help="Checkpoint filename tag to reuse with --eval-only")
    ap.add_argument("--out-dir", type=str, required=True)

    ap.add_argument("--graphs", type=str, nargs="+", required=True)
    ap.add_argument("--backbone-ckpt", type=str, required=True)
    ap.add_argument("--val-list", type=str, default="")
    ap.add_argument("--test-list", type=str, required=True)

    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--sparsity-target", type=float, default=0.05)
    ap.add_argument("--sparsity-coefs", type=str, default="0.1,1.0,10.0")
    ap.add_argument("--entropy-coefs", type=str, default="0.0,0.01,0.1")

    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--ks", type=str, default="10,50,100")
    ap.add_argument("--op-fprs", type=str, default="0.01,0.05,0.10")
    ap.add_argument("--rarity-stats", type=str, default="")
    ap.add_argument("--rarity-normalize", type=str, default="")
    ap.add_argument("--rarity-etypes", type=str, default="")
    ap.add_argument("--rarity-lambda", type=float, default=0.0)
    ap.add_argument("--rarity-idf-cap", type=float, default=0.0)
    ap.add_argument(
        "--filter-args",
        type=str,
        default="--filter-net --filter-net-ip --filter-system-files --filter-tmp-tempfile --filter-cmd-noise --dedup-dst --max-per-etype 8",
        help="Extra args appended to eval_hunt_rarity.py.",
    )
    ap.add_argument("--timeout-s", type=int, default=0, help="Per train/eval command timeout; 0 disables it")
    ap.add_argument("--python", type=str, default=sys.executable, help="Python executable used for train/eval commands")
    ap.add_argument("--eval-only", action="store_true", default=False, help="Reuse existing checkpoints and only run evaluation")
    ap.add_argument("--train-only", action="store_true", default=False, help="Train checkpoints but skip evaluation")
    ap.add_argument("--skip-preflight", action="store_true", default=False)
    ap.add_argument("--dry-run", action="store_true", default=False)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    tag = args.name.strip() or f"pgexp_sensitivity_{_now_ms()}"
    checkpoint_tag = args.checkpoint_tag.strip() or tag
    tsv_path = out_dir / f"{tag}.tsv"
    log_path = out_dir / f"{tag}.log.txt"
    meta_path = out_dir / f"{tag}.meta.json"
    timeout_s = int(args.timeout_s) if int(args.timeout_s) > 0 else None

    sparsity_coefs = _parse_float_list(args.sparsity_coefs)
    entropy_coefs = _parse_float_list(args.entropy_coefs)
    specs = _iter_specs(sparsity_coefs, entropy_coefs)

    meta = {
        "tag": tag,
        "checkpoint_tag": checkpoint_tag,
        "created_ms": _now_ms(),
        "python": str(args.python),
        "graphs": list(args.graphs),
        "backbone_ckpt": str(args.backbone_ckpt),
        "val_list": str(args.val_list),
        "test_list": str(args.test_list),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "temp": float(args.temp),
        "sparsity_target": float(args.sparsity_target),
        "grid": {"sparsity_coefs": sparsity_coefs, "entropy_coefs": entropy_coefs},
        "eval": {
            "device": str(args.device),
            "k": int(args.k),
            "ks": str(args.ks),
            "op_fprs": str(args.op_fprs),
            "rarity_stats": str(args.rarity_stats),
            "rarity_normalize": str(args.rarity_normalize),
            "rarity_etypes": str(args.rarity_etypes),
            "rarity_lambda": float(args.rarity_lambda),
            "rarity_idf_cap": float(args.rarity_idf_cap),
            "filter_args": str(args.filter_args),
        },
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    op_targets = [x.strip() for x in str(args.op_fprs).split(",") if x.strip()]
    header = [
        "sparsity_coef",
        "entropy_coef",
        "explainer_ckpt",
        "train_returncode",
        "eval_returncode",
        "final_train_loss",
        "base_auroc",
        "base_auprc",
        "rarity_auroc",
        "rarity_auprc",
    ]
    for tgt in op_targets:
        header.extend([f"base_f1@{tgt}", f"rarity_f1@{tgt}"])
    tsv_path.write_text("\t".join(header) + "\n", encoding="utf-8")

    filter_args = [x for x in str(args.filter_args).split(" ") if x.strip()]
    with log_path.open("w", encoding="utf-8") as log:
        if not args.skip_preflight and not args.dry_run:
            ok, preflight_out = _check_python_env(str(args.python), timeout_s=timeout_s)
            log.write("PREFLIGHT CMD: " + " ".join([str(args.python), "-c", "import torch; import torch_geometric"]) + "\n")
            log.write(preflight_out + "\n")
            log.flush()
            if not ok:
                raise SystemExit(
                    f"Preflight failed for {args.python!r}. Install torch_geometric in that environment "
                    "or rerun with --python pointing to the environment used for the main experiments."
                )
        if not args.train_only and not args.rarity_stats:
            raise SystemExit(
                "Evaluation requires --rarity-stats because scripts/eval_hunt_rarity.py requires it. "
                "Use --train-only to skip evaluation."
            )

        for i, spec in enumerate(specs, start=1):
            ckpt_path = ckpt_dir / f"{checkpoint_tag}_sp{spec.sparsity_coef:g}_ent{spec.entropy_coef:g}.pt"
            train_cmd = [
                str(args.python),
                "ranking_explain/train_pgexplainer.py",
                "--graphs",
                *[str(x) for x in args.graphs],
                "--epochs",
                str(int(args.epochs)),
                "--batch-size",
                str(int(args.batch_size)),
                "--lr",
                str(float(args.lr)),
                "--temp",
                str(float(args.temp)),
                "--sparsity",
                str(float(args.sparsity_target)),
                "--sparsity-coef",
                str(spec.sparsity_coef),
                "--entropy",
                str(spec.entropy_coef),
                "--backbone-ckpt",
                str(args.backbone_ckpt),
                "--save-ckpt",
                str(ckpt_path),
            ]

            eval_cmd = [
                str(args.python),
                "scripts/eval_hunt_rarity.py",
                "--test-list",
                str(args.test_list),
                "--backbone-ckpt",
                str(args.backbone_ckpt),
                "--explainer-ckpt",
                str(ckpt_path),
                "--k",
                str(int(args.k)),
                "--ks",
                str(args.ks),
                "--device",
                str(args.device),
                "--op-fprs",
                str(args.op_fprs),
                "--rarity-lambda",
                str(float(args.rarity_lambda)),
                "--rarity-idf-cap",
                str(float(args.rarity_idf_cap)),
                "--rarity-etypes",
                str(args.rarity_etypes),
                "--rarity-normalize",
                str(args.rarity_normalize),
            ]
            if args.val_list:
                eval_cmd.extend(["--val-list", str(args.val_list)])
            if args.rarity_stats:
                eval_cmd.extend(["--rarity-stats", str(args.rarity_stats)])
            eval_cmd.extend(filter_args)

            log.write(f"\n## RUN {i}/{len(specs)} sp={spec.sparsity_coef:g} ent={spec.entropy_coef:g}\n")
            log.write("TRAIN CMD: " + " ".join(train_cmd) + "\n")
            log.flush()

            if args.dry_run:
                train_rc, train_out = 0, ""
                eval_rc, eval_out = 0, ""
            else:
                if args.eval_only:
                    train_rc, train_out = 0, ""
                    if not ckpt_path.exists():
                        raise SystemExit(f"Missing checkpoint for --eval-only: {ckpt_path.as_posix()}")
                    log.write("TRAIN SKIPPED: --eval-only\n")
                    log.flush()
                else:
                    train_rc, train_out = _run(train_cmd, timeout_s=timeout_s)
                    log.write(train_out + "\n")
                    log.write(f"train_returncode={train_rc}\n")
                    log.flush()

                if train_rc == 0 and not args.train_only:
                    log.write("EVAL CMD: " + " ".join(eval_cmd) + "\n")
                    log.flush()
                    eval_rc, eval_out = _run(eval_cmd, timeout_s=timeout_s)
                    log.write(eval_out + "\n")
                    log.write(f"eval_returncode={eval_rc}\n")
                    log.flush()
                elif args.train_only:
                    eval_rc, eval_out = 0, ""
                else:
                    eval_rc, eval_out = -1, ""

            metrics = _parse_eval_metrics(eval_out)
            row: list[Any] = [
                spec.sparsity_coef,
                spec.entropy_coef,
                ckpt_path.as_posix(),
                train_rc,
                eval_rc,
                _last_epoch_loss(train_out),
                metrics["base_auroc"],
                metrics["base_auprc"],
                metrics["rarity_auroc"],
                metrics["rarity_auprc"],
            ]
            for tgt in op_targets:
                row.extend([
                    _op_metric(metrics, "base", tgt, "f1"),
                    _op_metric(metrics, "rarity", tgt, "f1"),
                ])

            with tsv_path.open("a", encoding="utf-8") as f:
                f.write("\t".join(_fmt(x) for x in row) + "\n")

    print(f"WROTE {tsv_path.as_posix()}")
    print(f"WROTE {log_path.as_posix()}")
    print(f"WROTE {meta_path.as_posix()}")


if __name__ == "__main__":
    main()
