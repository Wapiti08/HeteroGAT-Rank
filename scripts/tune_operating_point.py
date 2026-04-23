from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)


Calib = Literal["none", "platt", "isotonic"]


@dataclass(frozen=True)
class Metrics:
    acc: float
    precision: float
    recall: float
    f1: float
    auroc: float
    auprc: float
    tn: int
    fp: int
    fn: int
    tp: int


def _safe_auroc(y: np.ndarray, s: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def _safe_auprc(y: np.ndarray, s: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, s))


def _eval_at_threshold(y: np.ndarray, s: np.ndarray, thr: float) -> Metrics:
    pred = (s >= float(thr)).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = (int(x) for x in cm.ravel())
    acc = float(accuracy_score(y, pred))
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary", pos_label=1, zero_division=0)
    return Metrics(
        acc=acc,
        precision=float(p),
        recall=float(r),
        f1=float(f1),
        auroc=_safe_auroc(y, s),
        auprc=_safe_auprc(y, s),
        tn=tn,
        fp=fp,
        fn=fn,
        tp=tp,
    )


def _pick_threshold_max_f1(y: np.ndarray, s: np.ndarray) -> float:
    p, r, thr = precision_recall_curve(y, s)
    if len(thr) == 0:
        return 0.5
    f1 = (2 * p[:-1] * r[:-1]) / (p[:-1] + r[:-1] + 1e-12)
    return float(thr[int(np.nanargmax(f1))])


def _pick_threshold_at_precision(y: np.ndarray, s: np.ndarray, target_p: float) -> Optional[float]:
    p, r, thr = precision_recall_curve(y, s)
    if len(thr) == 0:
        return None
    ok = p[:-1] >= float(target_p)
    if not np.any(ok):
        return None
    idx = int(np.nanargmax(np.where(ok, r[:-1], -1.0)))
    return float(thr[idx])


def _pick_threshold_at_fpr(y: np.ndarray, s: np.ndarray, target_fpr: float) -> Optional[float]:
    fpr, tpr, thr = roc_curve(y, s)
    if len(thr) == 0:
        return None
    ok = fpr <= float(target_fpr)
    if not np.any(ok):
        return None
    idx = int(np.nanargmax(np.where(ok, tpr, -1.0)))
    return float(thr[idx])


def _fit_calibrator(y_val: np.ndarray, s_val: np.ndarray, mode: Calib):
    if mode == "none":
        return None
    if mode == "platt":
        lr = LogisticRegression(solver="lbfgs", max_iter=10_000)
        lr.fit(s_val.reshape(-1, 1), y_val.astype(int))
        return lr
    if mode == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(s_val, y_val.astype(int))
        return iso
    raise KeyError(mode)


def _apply_calibrator(cal, s: np.ndarray, mode: Calib) -> np.ndarray:
    if cal is None or mode == "none":
        return s
    if mode == "platt":
        return cal.predict_proba(s.reshape(-1, 1))[:, 1]
    if mode == "isotonic":
        return cal.predict(s)
    raise KeyError(mode)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=str, required=True, help="CSV with columns: split,y,score (optionally other columns)")
    ap.add_argument("--split-col", type=str, default="split")
    ap.add_argument("--y-col", type=str, default="y")
    ap.add_argument("--score-col", type=str, default="score")
    ap.add_argument("--val-split", type=str, default="val")
    ap.add_argument("--test-split", type=str, default="test")
    ap.add_argument("--calibrate", type=str, default="none", choices=["none", "platt", "isotonic"])
    ap.add_argument("--fpr-targets", type=str, default="0.01,0.05")
    ap.add_argument("--precision-targets", type=str, default="0.90,0.95")
    ap.add_argument("--out-json", type=str, default="", help="Optional: write report json")
    args = ap.parse_args()

    df = pd.read_csv(args.scores)
    for c in (args.split_col, args.y_col, args.score_col):
        if c not in df.columns:
            raise KeyError(f"Missing column {c} in {args.scores}")

    dval = df[df[args.split_col].astype(str) == str(args.val_split)].copy()
    dtest = df[df[args.split_col].astype(str) == str(args.test_split)].copy()
    if dval.empty or dtest.empty:
        raise SystemExit(f"Empty splits: val={len(dval)} test={len(dtest)}. Check --val-split/--test-split.")

    y_val = dval[args.y_col].astype(int).to_numpy()
    s_val = dval[args.score_col].astype(float).to_numpy()
    y_test = dtest[args.y_col].astype(int).to_numpy()
    s_test = dtest[args.score_col].astype(float).to_numpy()

    mode: Calib = str(args.calibrate)  # type: ignore[assignment]
    cal = _fit_calibrator(y_val, s_val, mode)
    s_val_c = _apply_calibrator(cal, s_val, mode)
    s_test_c = _apply_calibrator(cal, s_test, mode)

    ops: list[dict] = []

    thr_default = 0.5
    ops.append({"name": "thr=0.5", "threshold": thr_default, "metrics": _eval_at_threshold(y_test, s_test_c, thr_default).__dict__})

    thr_f1 = _pick_threshold_max_f1(y_val, s_val_c)
    ops.append({"name": "max_f1@val", "threshold": thr_f1, "metrics": _eval_at_threshold(y_test, s_test_c, thr_f1).__dict__})

    fpr_targets = [float(x) for x in str(args.fpr_targets).split(",") if x.strip()]
    for t in fpr_targets:
        thr = _pick_threshold_at_fpr(y_val, s_val_c, t)
        if thr is None:
            continue
        ops.append({"name": f"fpr<={t:g}@val", "threshold": thr, "metrics": _eval_at_threshold(y_test, s_test_c, thr).__dict__})

    p_targets = [float(x) for x in str(args.precision_targets).split(",") if x.strip()]
    for t in p_targets:
        thr = _pick_threshold_at_precision(y_val, s_val_c, t)
        if thr is None:
            continue
        ops.append({"name": f"precision>={t:g}@val", "threshold": thr, "metrics": _eval_at_threshold(y_test, s_test_c, thr).__dict__})

    print(f"scores={args.scores} val_n={len(dval)} test_n={len(dtest)} calibrate={mode}")
    print(f"val_pos={int((y_val==1).sum())} val_neg={int((y_val==0).sum())}")
    print(f"test_pos={int((y_test==1).sum())} test_neg={int((y_test==0).sum())}")
    print("")
    for op in ops:
        mm = op["metrics"]
        print(f"== {op['name']} ==")
        print(
            f"thr={op['threshold']:.6f} acc={mm['acc']:.4f} precision={mm['precision']:.4f} recall={mm['recall']:.4f} "
            f"f1={mm['f1']:.4f} auroc={mm['auroc']:.4f} auprc={mm['auprc']:.4f} "
            f"cm=[[{mm['tn']},{mm['fp']}],[{mm['fn']},{mm['tp']}]]"
        )

    report = {
        "scores": args.scores,
        "calibrate": mode,
        "val": {"n": int(len(dval)), "pos": int((y_val == 1).sum()), "neg": int((y_val == 0).sum())},
        "test": {"n": int(len(dtest)), "pos": int((y_test == 1).sum()), "neg": int((y_test == 0).sum())},
        "operating_points": ops,
    }
    if args.out_json:
        p = Path(args.out_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"\nwrote={p.as_posix()}")


if __name__ == "__main__":
    main()

