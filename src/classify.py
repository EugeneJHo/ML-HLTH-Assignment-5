"""Cross-validated hypoxia classification with ROC/AUC bootstrap CIs."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    auc,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import DEMOGRAPHIC_COLS, N_BOOTSTRAP, N_CV_FOLDS, RANDOM_STATE

SIGNAL_FEATURE_COLS = [
    "fhr_mean",
    "fhr_std",
    "fhr_min",
    "fhr_max",
    "baseline_mean",
    "baseline_std",
    "variability",
    "n_accelerations",
    "n_decelerations",
    "n_contractions",
    "accel_rate_per_10min",
    "decel_rate_per_10min",
    "contraction_rate_per_10min",
    "mean_decel_amp",
    "mean_accel_amp",
    "uc_mean",
    "uc_std",
    "fhr_missing_frac",
    "uc_missing_frac",
]


def feature_columns(include_demographics: bool = False) -> List[str]:
    cols = list(SIGNAL_FEATURE_COLS)
    if include_demographics:
        cols = cols + list(DEMOGRAPHIC_COLS)
    return cols


def make_classifier(random_state: int = RANDOM_STATE) -> Pipeline:
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", clf),
        ]
    )


def prepare_xy(
    df: pd.DataFrame,
    include_demographics: bool = False,
    max_missing: Optional[float] = None,
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    data = df.copy()
    if max_missing is not None and "fhr_missing_frac" in data.columns:
        data = data.loc[data["fhr_missing_frac"] <= max_missing].copy()
    cols = [c for c in feature_columns(include_demographics) if c in data.columns]
    X = data[cols]
    y = data["label"].astype(int).values
    return X, y, cols


def bootstrap_roc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_boot: int = N_BOOTSTRAP,
    seed: int = RANDOM_STATE,
    grid_size: int = 101,
) -> Dict:
    rng = np.random.default_rng(seed)
    fpr_grid = np.linspace(0, 1, grid_size)
    tprs = []
    aucs = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true[idx], y_score[idx])
        tprs.append(np.interp(fpr_grid, fpr, tpr))
        aucs.append(auc(fpr_grid, tprs[-1]))
    tprs = np.asarray(tprs)
    aucs = np.asarray(aucs)
    return {
        "fpr": fpr_grid,
        "tpr_mean": tprs.mean(axis=0),
        "tpr_lo": np.percentile(tprs, 2.5, axis=0),
        "tpr_hi": np.percentile(tprs, 97.5, axis=0),
        "auc_mean": float(aucs.mean()),
        "auc_lo": float(np.percentile(aucs, 2.5)),
        "auc_hi": float(np.percentile(aucs, 97.5)),
        "aucs": aucs,
    }


def operating_point(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_specificity: float = 0.90,
) -> Dict:
    fpr, tpr, thr = roc_curve(y_true, y_score)
    spec = 1 - fpr
    # Choose threshold with specificity closest to target, preferring higher sensitivity on ties
    idx = int(np.argmin(np.abs(spec - target_specificity) + (1 - tpr) * 1e-6))
    threshold = float(thr[idx]) if idx < len(thr) else 0.5
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    speci = tn / (tn + fp) if (tn + fp) else np.nan
    return {
        "threshold": threshold,
        "sensitivity": float(sens),
        "specificity": float(speci),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def cross_validated_scores(
    X: pd.DataFrame,
    y: np.ndarray,
    n_folds: int = N_CV_FOLDS,
    seed: int = RANDOM_STATE,
) -> Tuple[np.ndarray, Pipeline]:
    pipe = make_classifier(seed)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    pipe.fit(X, y)
    return scores, pipe


def evaluate_setting(
    df: pd.DataFrame,
    include_demographics: bool = False,
    max_missing: Optional[float] = None,
    label: str = "baseline",
) -> Dict:
    X, y, cols = prepare_xy(df, include_demographics=include_demographics, max_missing=max_missing)
    if len(y) < 20 or len(np.unique(y)) < 2:
        return {"label": label, "n": len(y), "error": "insufficient data"}
    scores, model = cross_validated_scores(X, y)
    roc = bootstrap_roc(y, scores)
    op = operating_point(y, scores, target_specificity=0.90)
    # Also Youden-optimal point
    fpr, tpr, thr = roc_curve(y, scores)
    youden = tpr - fpr
    j = int(np.argmax(youden))
    op_youden = operating_point(y, scores, target_specificity=float(1 - fpr[j]))
    return {
        "label": label,
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "n_neg": int((1 - y).sum()),
        "feature_cols": cols,
        "y_true": y,
        "y_score": scores,
        "roc": roc,
        "auc": roc["auc_mean"],
        "auc_ci": (roc["auc_lo"], roc["auc_hi"]),
        "operating_point_spec90": op,
        "operating_point_youden": op_youden,
        "model": model,
        "X": X,
    }


def false_examples(result: Dict, n_each: int = 1) -> Dict[str, pd.DataFrame]:
    y = result["y_true"]
    s = result["y_score"]
    thr = result["operating_point_youden"]["threshold"]
    pred = (s >= thr).astype(int)
    X = result["X"].copy()
    X = X.assign(y_true=y, y_score=s, y_pred=pred)
    fp = X[(X.y_true == 0) & (X.y_pred == 1)].sort_values("y_score", ascending=False).head(n_each)
    fn = X[(X.y_true == 1) & (X.y_pred == 0)].sort_values("y_score", ascending=True).head(n_each)
    return {"false_positives": fp, "false_negatives": fn}
