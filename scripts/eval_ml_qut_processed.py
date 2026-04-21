from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


QUT_BASE = Path("data/QUT-DV25_Datasets/QUT-DV25_Processed_Datasets")
QUT_TABLES: Dict[str, Path] = {
    "install": QUT_BASE / "QUT-DV25_Install_Traces/QUT-DV25_Install_Traces.csv",
    "syscall": QUT_BASE / "QUT-DV25_SysCall_Traces/QUT-DV25_SysCall_Traces.csv",
    "filetop": QUT_BASE / "QUT-DV25_Filetop_Traces/QUT-DV25_Filetop_Traces.csv",
    "opensnoop": QUT_BASE / "QUT-DV25_Opensnoop_Traces/QUT-DV25_Opensnoop_Traces.csv",
    "tcp": QUT_BASE / "QUT-DV25_TCP_Traces/QUT-DV25_TCP_Traces.csv",
    "pattern": QUT_BASE / "QUT-DV25_Pattern_Traces/QUT-DV25_Pattern_Traces.csv",
}


def _read_list(path: str | None) -> list[str]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p.as_posix())
    return [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]


def _pkg_name_from_graph_path(p: str) -> str:
    """Extract QUT Package_Name from a *.graph.pt path or stem.

    Canonical QUT graph keys are `qut::<Package_Name>`, but split files usually
    list the *.graph.pt paths; we just use the file stem.
    """
    s = Path(p).name
    if s.endswith(".graph.pt"):
        s = s[: -len(".graph.pt")]
    return s


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


def _load_features(*, kinds: Sequence[str]) -> pd.DataFrame:
    """Load and join processed QUT tables on Package_Name.

    We intentionally use only numeric columns for a stable baseline.
    """
    base: Optional[pd.DataFrame] = None
    for k in kinds:
        if k not in QUT_TABLES:
            raise KeyError(f"Unknown kind: {k}. Available: {sorted(QUT_TABLES)}")
        path = QUT_TABLES[k]
        if not path.exists():
            raise FileNotFoundError(path.as_posix())
        df = pd.read_csv(path)
        if "Package_Name" not in df.columns:
            raise KeyError(f"Missing Package_Name in {path}")

        # Keep label only once (from the first table that provides it).
        # Many QUT processed tables include `Level`, and merging them would create
        # overlapping columns.
        keep_level = (base is None) and ("Level" in df.columns)
        keep_cols = ["Package_Name"] + (["Level"] if keep_level else [])
        num_cols = [c for c in df.columns if c not in keep_cols and pd.api.types.is_numeric_dtype(df[c])]
        df = df[keep_cols + num_cols].copy()

        # Prefix feature columns by table kind to avoid collisions.
        rename = {c: f"{k}__{c}" for c in num_cols}
        df = df.rename(columns=rename)

        if base is None:
            base = df
        else:
            # `Level` is intentionally not merged from subsequent tables.
            base = base.merge(df, on=["Package_Name"], how="inner")

    assert base is not None
    if "Level" not in base.columns:
        raise KeyError("Joined table missing Level column; ensure at least one kind includes it")
    return base


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-list", type=str, required=True)
    ap.add_argument("--test-list", type=str, required=True)
    ap.add_argument(
        "--kinds",
        type=str,
        default="install,syscall,filetop,opensnoop,tcp,pattern",
        help="Comma-separated subset of processed tables",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ks", type=str, default="10,50,100")
    ap.add_argument("--rf-trees", type=int, default=300)
    args = ap.parse_args()

    kinds = [k.strip() for k in str(args.kinds).split(",") if k.strip()]
    df = _load_features(kinds=kinds)

    train_pkgs = {_pkg_name_from_graph_path(p) for p in _read_list(args.train_list)}
    test_pkgs = {_pkg_name_from_graph_path(p) for p in _read_list(args.test_list)}

    dtrain = df[df["Package_Name"].isin(train_pkgs)].copy()
    dtest = df[df["Package_Name"].isin(test_pkgs)].copy()

    if dtrain.empty or dtest.empty:
        raise SystemExit(
            f"Empty train/test after join: train={len(dtrain)} test={len(dtest)}. "
            "Check that split lists contain QUT *.graph.pt paths and that processed CSVs match those Package_Name values."
        )

    y_train = dtrain["Level"].astype(int).to_numpy()
    y_test = dtest["Level"].astype(int).to_numpy()

    feature_cols = [c for c in df.columns if c not in ("Package_Name", "Level")]
    X_train = dtrain[feature_cols].fillna(0.0).to_numpy(dtype=float)
    X_test = dtest[feature_cols].fillna(0.0).to_numpy(dtype=float)

    ks = [int(x) for x in str(args.ks).split(",") if x.strip()]

    models = {
        "DT": DecisionTreeClassifier(random_state=args.seed),
        "RF": RandomForestClassifier(
            n_estimators=int(args.rf_trees),
            random_state=args.seed,
            n_jobs=-1,
        ),
        "GB": GradientBoostingClassifier(random_state=args.seed),
        # SVM needs scaling; probability=True enables predict_proba
        "SVM": make_pipeline(StandardScaler(with_mean=False), SVC(probability=True, random_state=args.seed)),
    }

    print(f"train_n={len(dtrain)} test_n={len(dtest)} feats={len(feature_cols)} kinds={kinds}")
    print(f"test_pos={int((y_test==1).sum())} test_neg={int((y_test==0).sum())}")

    for name, clf in models.items():
        clf.fit(X_train, y_train)
        prob1 = clf.predict_proba(X_test)[:, 1]
        pred = (prob1 >= 0.5).astype(int)

        cm = confusion_matrix(y_test, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        acc = float(accuracy_score(y_test, pred))
        p, r, f1, _ = precision_recall_fscore_support(y_test, pred, average="binary", pos_label=1, zero_division=0)
        auroc = _safe_auc(y_test, prob1)
        auprc = _safe_auprc(y_test, prob1)

        print(f"\n== {name} ==")
        print(f"acc={acc:.4f} precision={p:.4f} recall={r:.4f} f1={f1:.4f} auroc={auroc:.4f} auprc={auprc:.4f}")
        print(f"confusion_matrix [[tn, fp],[fn, tp]] = [[{tn}, {fp}], [{fn}, {tp}]]")
        for k, pk, rk in _precision_recall_at_k(y_test, prob1, ks):
            tp_at_k = int(round(pk * min(k, len(y_test))))
            print(f"P@{k}={pk:.4f} R@{k}={rk:.4f}")


if __name__ == "__main__":
    main()

