"""Plotting helpers for Assignment 5 figures."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import FS, FIGURES_DIR, PH_CUTOFF, PH_SEVERE
from .features import Event


def savefig(name: str, fig: Optional[plt.Figure] = None) -> Path:
    path = FIGURES_DIR / name
    (fig or plt.gcf()).savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig or plt.gcf())
    return path


def plot_trace_pair(
    fhr_n,
    uc_n,
    fhr_h,
    uc_h,
    title_n: str,
    title_h: str,
    out_name: str = "01a_example_traces.png",
):
    t_n = np.arange(len(fhr_n)) / FS / 60
    t_h = np.arange(len(fhr_h)) / FS / 60
    fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharex="col")
    axes[0, 0].plot(t_n, fhr_n, color="#1f4e79", lw=0.8)
    axes[0, 0].set_title(title_n)
    axes[0, 0].set_ylabel("FHR (bpm)")
    axes[1, 0].plot(t_n, uc_n, color="#8b4513", lw=0.8)
    axes[1, 0].set_ylabel("UC")
    axes[1, 0].set_xlabel("Time (min before end →)")
    axes[0, 1].plot(t_h, fhr_h, color="#9b2226", lw=0.8)
    axes[0, 1].set_title(title_h)
    axes[1, 1].plot(t_h, uc_h, color="#8b4513", lw=0.8)
    axes[1, 1].set_xlabel("Time (min before end →)")
    for ax in axes.ravel():
        ax.grid(alpha=0.25)
    fig.suptitle("Example CTG traces: normal vs hypoxic", y=1.02)
    return savefig(out_name, fig)


def plot_ph_distribution(meta: pd.DataFrame, out_name: str = "01b_ph_distribution.png"):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(meta["pH"], bins=30, color="#4c6ef5", edgecolor="white", alpha=0.9)
    ax.axvline(PH_CUTOFF, color="#e03131", ls="--", label=f"pH < {PH_CUTOFF}")
    ax.axvline(PH_SEVERE, color="#862e9c", ls="--", label=f"pH < {PH_SEVERE}")
    ax.set_xlabel("Umbilical artery cord pH")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of cord pH")
    ax.legend()
    ax.grid(alpha=0.25)
    return savefig(out_name, fig)


def plot_acquisition_methods(meta: pd.DataFrame, out_name: str = "01c_acquisition_methods.png"):
    counts = meta["rec_type_label"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 4))
    counts.plot(kind="bar", ax=ax, color=["#228be6", "#12b886", "#fab005"])
    ax.set_ylabel("Number of records")
    ax.set_xlabel("")
    ax.set_title("Data acquisition methods")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 3, str(v), ha="center")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=20, ha="right")
    return savefig(out_name, fig)


def plot_demo_correlations(meta: pd.DataFrame, cols: List[str], out_name: str = "01d_demo_corr_ph.png"):
    use = [c for c in cols if c in meta.columns] + ["pH"]
    corr = meta[use].corr(numeric_only=True)
    ph_corr = corr["pH"].drop("pH").sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#e03131" if v < 0 else "#2f9e44" for v in ph_corr.values]
    ph_corr.plot(kind="barh", ax=ax, color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Pearson correlation with cord pH")
    ax.set_title("Demographic / clinical correlations with cord pH")
    ax.grid(axis="x", alpha=0.25)
    return savefig(out_name, fig)


def plot_missingness(miss: pd.DataFrame, out_name: str = "02a_missingness.png"):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    sns.histplot(miss["fhr_missing"], bins=25, ax=axes[0], color="#4c6ef5")
    axes[0].set_title("FHR missingness (cohort)")
    axes[0].set_xlabel("Fraction missing")

    sns.boxplot(data=miss, x="hypoxia_7_20", y="fhr_missing", ax=axes[1], palette=["#2f9e44", "#e03131"])
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["Normal", "Hypoxic"])
    axes[1].set_xlabel("")
    axes[1].set_title("Missingness by outcome")

    order = miss.groupby("rec_type_label")["fhr_missing"].median().sort_values().index
    sns.boxplot(data=miss, x="rec_type_label", y="fhr_missing", ax=axes[2], order=order)
    axes[2].set_title("Missingness by acquisition")
    axes[2].tick_params(axis="x", rotation=25)
    axes[2].set_xlabel("")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    return savefig(out_name, fig)


def plot_segmentation(seg: Dict, out_name: str = "03_segmentation_example.png", minutes: float = 20.0):
    n = int(minutes * 60 * FS)
    fhr = seg["fhr"][-n:]
    uc = seg["uc"][-n:]
    baseline = seg["baseline"][-n:]
    offset = len(seg["fhr"]) - len(fhr)
    t = np.arange(len(fhr)) / FS / 60

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(t, fhr, color="#1f4e79", lw=0.9, label="FHR")
    axes[0].plot(t, baseline, color="#e8590c", lw=1.2, label="Baseline")
    for ev in seg["accelerations"]:
        if ev.end < offset:
            continue
        s = max(0, ev.start - offset) / FS / 60
        e = max(0, ev.end - offset) / FS / 60
        axes[0].axvspan(s, e, color="#2f9e44", alpha=0.25)
    for ev in seg["decelerations"]:
        if ev.end < offset:
            continue
        s = max(0, ev.start - offset) / FS / 60
        e = max(0, ev.end - offset) / FS / 60
        axes[0].axvspan(s, e, color="#e03131", alpha=0.25)
    axes[0].set_ylabel("FHR (bpm)")
    axes[0].legend(loc="upper right")
    axes[0].set_title("Accelerations (green) / decelerations (red) over estimated baseline")

    axes[1].plot(t, uc, color="#8b4513", lw=0.9, label="UC")
    for ev in seg["contractions"]:
        if ev.end < offset:
            continue
        s = max(0, ev.start - offset) / FS / 60
        e = max(0, ev.end - offset) / FS / 60
        axes[1].axvspan(s, e, color="#fab005", alpha=0.3)
    axes[1].set_ylabel("UC")
    axes[1].set_xlabel("Time within last window (min)")
    axes[1].set_title("Detected contractions (yellow)")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    return savefig(out_name, fig)


def plot_roc_with_ci(result: Dict, out_name: str = "04a_roc.png", title: str = "ROC"):
    roc = result["roc"]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(roc["fpr"], roc["tpr_mean"], color="#1c7ed6", lw=2, label=f"AUC={roc['auc_mean']:.3f}")
    ax.fill_between(roc["fpr"], roc["tpr_lo"], roc["tpr_hi"], color="#1c7ed6", alpha=0.2, label="95% CI")
    ax.plot([0, 1], [0, 1], ls="--", color="gray")
    op = result["operating_point_spec90"]
    ax.scatter([1 - op["specificity"]], [op["sensitivity"]], color="#e03131", zorder=5, label="Spec≈0.90 point")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    return savefig(out_name, fig)


def plot_horizon_aucs(rows: List[Dict], out_name: str = "04f_horizon_auc.png"):
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(
        df["offset_min"],
        df["auc"],
        yerr=[df["auc"] - df["auc_lo"], df["auc_hi"] - df["auc"]],
        fmt="-o",
        color="#1c7ed6",
        capsize=4,
    )
    ax.set_xlabel("Minutes before delivery (window end offset)")
    ax.set_ylabel("AUC")
    ax.set_title("Predictiveness vs time before delivery (30-min windows)")
    ax.grid(alpha=0.25)
    return savefig(out_name, fig)


def plot_subgroup_aucs(rows: List[Dict], out_name: str = "05_subgroup_auc.png"):
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=df, x="subgroup", y="auc", hue="level", ax=ax)
    ax.set_ylim(0.4, 1.0)
    ax.set_title("Subgroup AUCs")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    return savefig(out_name, fig)


def plot_shap_importance(shap_vals, feature_names, out_name: str = "06_shap_importance.png"):
    mean_abs = np.abs(shap_vals).mean(axis=0)
    order = np.argsort(mean_abs)[-15:]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(np.array(feature_names)[order], mean_abs[order], color="#7048e8")
    ax.set_xlabel("Mean |SHAP|")
    ax.set_title("Feature importance (SHAP)")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return savefig(out_name, fig)


def plot_risk_over_time(times_min, risks, label: str, out_name: str = "07_risk_over_time.png"):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(times_min, risks, color="#c92a2a", lw=2)
    ax.set_xlabel("Minutes before delivery")
    ax.set_ylabel("Predicted hypoxia risk")
    ax.set_title(f"Time-varying risk score ({label})")
    ax.invert_xaxis()
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    return savefig(out_name, fig)
