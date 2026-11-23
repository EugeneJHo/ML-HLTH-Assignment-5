#!/usr/bin/env python3
"""End-to-end Assignment 5 analysis pipeline.

Produces figures/ and results/ artifacts covering exploration, missingness,
features, classification, subgroups, SHAP, and time-varying risk.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import shap

from .classify import evaluate_setting, false_examples
from .config import (
    DATA_DIR,
    DEMOGRAPHIC_COLS as DEMO_COLS,
    FIGURES_DIR,
    PH_CUTOFF,
    PH_SEVERE,
    QUALITY_MISSING_MAX,
    RESULTS_DIR,
)
from .data_loader import load_all_signals, load_metadata_table
from .features import build_feature_table, segment_events
from .plots import (
    plot_acquisition_methods,
    plot_demo_correlations,
    plot_horizon_aucs,
    plot_missingness,
    plot_ph_distribution,
    plot_risk_over_time,
    plot_roc_with_ci,
    plot_segmentation,
    plot_shap_importance,
    plot_subgroup_aucs,
    plot_trace_pair,
)
from .preprocess import apply_quality_filter, cohort_missingness_table

warnings.filterwarnings("ignore", category=FutureWarning)


def _json_safe(obj):
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def phase1_exploration(meta: pd.DataFrame, signals: dict) -> dict:
    print("Phase 1: dataset exploration")
    n = len(meta)
    pos720 = int((meta["pH"] < PH_CUTOFF).sum())
    pos705 = int((meta["pH"] < PH_SEVERE).sum())
    summary = {
        "n_records": n,
        "positive_rate_ph_lt_7_20": pos720 / n,
        "positive_rate_ph_lt_7_05": pos705 / n,
        "n_pos_7_20": pos720,
        "n_pos_7_05": pos705,
        "imbalance_notes": (
            "Class imbalance (~32% positive at pH<7.20; ~7% at pH<7.05). "
            "Mitigations: class_weight='balanced', oversampling of minority crops, "
            "threshold tuning on ROC, and reporting sensitivity at fixed high specificity."
        ),
    }

    # Example traces: pick relatively complete normal and hypoxic records
    miss_proxy = {
        rid: np.mean(~np.isfinite(signals[rid]["fhr"])) for rid in signals
    }
    normals = meta.loc[meta["hypoxia_7_20"] == 0].copy()
    hyps = meta.loc[meta["hypoxia_7_20"] == 1].copy()
    normals["miss"] = normals["record_id"].map(miss_proxy)
    hyps["miss"] = hyps["record_id"].map(miss_proxy)
    rid_n = normals.sort_values("miss").iloc[0]["record_id"]
    rid_h = hyps.sort_values("miss").iloc[0]["record_id"]
    # Prefer a clearly acidotic example if quality is acceptable
    severe = hyps.loc[hyps["pH"] < 7.10].sort_values("miss")
    if len(severe):
        rid_h = severe.iloc[0]["record_id"]

    sn, sh = signals[rid_n], signals[rid_h]
    plot_trace_pair(
        sn["fhr"],
        sn["uc"],
        sh["fhr"],
        sh["uc"],
        title_n=f"Normal {rid_n} (pH={meta.set_index('record_id').loc[rid_n,'pH']:.2f})",
        title_h=f"Hypoxic {rid_h} (pH={meta.set_index('record_id').loc[rid_h,'pH']:.2f})",
    )
    plot_ph_distribution(meta)
    plot_acquisition_methods(meta)
    plot_demo_correlations(meta, DEMO_COLS)

    summary["example_normal"] = rid_n
    summary["example_hypoxic"] = rid_h
    (RESULTS_DIR / "phase1_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def phase2_missingness(meta: pd.DataFrame, signals: dict) -> pd.DataFrame:
    print("Phase 2: signal missingness")
    miss = cohort_missingness_table(signals, meta)
    miss.to_csv(RESULTS_DIR / "missingness_by_record.csv", index=False)
    plot_missingness(miss)

    kept = apply_quality_filter(miss, max_missing=QUALITY_MISSING_MAX, channel="fhr_missing")
    summary = {
        "mean_fhr_missing": float(miss["fhr_missing"].mean()),
        "mean_fhr_missing_normal": float(miss.loc[miss["hypoxia_7_20"] == 0, "fhr_missing"].mean()),
        "mean_fhr_missing_hypoxic": float(miss.loc[miss["hypoxia_7_20"] == 1, "fhr_missing"].mean()),
        "missing_by_acquisition": miss.groupby("rec_type_label")["fhr_missing"]
        .agg(["mean", "median", "count"])
        .round(4)
        .to_dict(),
        "quality_filter_max_missing": QUALITY_MISSING_MAX,
        "n_before_filter": int(len(miss)),
        "n_after_filter": int(len(kept)),
        "pos_rate_before": float(miss["hypoxia_7_20"].mean()),
        "pos_rate_after": float(kept["hypoxia_7_20"].mean()) if len(kept) else None,
        "n_by_acq_before": miss["rec_type_label"].value_counts().to_dict(),
        "n_by_acq_after": kept["rec_type_label"].value_counts().to_dict(),
        "note": (
            "Short dropouts <10 s are linearly interpolated; gaps ≥10 s remain missing. "
            "A 10% missingness filter reduces usable N and can shift class balance / "
            "acquisition-method mix because ultrasound traces tend to have more dropout."
        ),
    }
    (RESULTS_DIR / "phase2_summary.json").write_text(json.dumps(_json_safe(summary), indent=2))
    return miss


def phase3_features(meta: pd.DataFrame, signals: dict, example_id: str) -> pd.DataFrame:
    print("Phase 3: feature definitions / segmentation")
    seg = segment_events(signals[example_id]["fhr"], signals[example_id]["uc"])
    plot_segmentation(seg, minutes=25.0)

    feat = build_feature_table(signals, meta, end_offset_min=0.0, window_min=30.0, include_demographics=True)
    feat.to_csv(RESULTS_DIR / "features_last30min.csv", index=False)
    return feat


def phase4_classification(feat: pd.DataFrame) -> dict:
    print("Phase 4: classification")
    results = {}

    baseline = evaluate_setting(feat, include_demographics=False, label="signal_only_last30")
    with_demo = evaluate_setting(feat, include_demographics=True, label="signal_plus_demographics")
    quality = evaluate_setting(
        feat,
        include_demographics=False,
        max_missing=QUALITY_MISSING_MAX,
        label="signal_quality_filtered",
    )
    results["baseline"] = baseline
    results["with_demographics"] = with_demo
    results["quality_filtered"] = quality

    plot_roc_with_ci(baseline, out_name="04a_roc_baseline.png", title="ROC — last 30 min (signal features)")
    plot_roc_with_ci(with_demo, out_name="04d_roc_with_demographics.png", title="ROC — signal + demographics")
    if "error" not in quality:
        plot_roc_with_ci(quality, out_name="04e_roc_quality_filtered.png", title="ROC — ≤10% missing filter")

    # Horizons: window ends 0 / 15 / 30 / 60 min before delivery
    from .data_loader import load_all_signals
    from .features import build_feature_table as bft

    # Reuse already-built last-30; rebuild for other offsets from cached signals on disk if needed
    # Caller passes feat built from last 30; we rebuild other horizons below in main using signals.

    summary = {
        "baseline_auc": baseline.get("auc"),
        "baseline_auc_ci": baseline.get("auc_ci"),
        "baseline_op_spec90": baseline.get("operating_point_spec90"),
        "baseline_op_youden": baseline.get("operating_point_youden"),
        "with_demographics_auc": with_demo.get("auc"),
        "with_demographics_auc_ci": with_demo.get("auc_ci"),
        "quality_filtered_auc": quality.get("auc"),
        "quality_filtered_n": quality.get("n"),
        "imbalance_strategy": (
            "RandomForest with class_weight='balanced'; evaluate with AUC and "
            "sensitivity at high specificity; optional quality filtering / threshold tuning."
        ),
        "operating_point_discussion": (
            "For screening fetal hypoxia we prefer high specificity (here ~0.90) to limit "
            "unnecessary interventions, then report the achievable sensitivity. "
            "Youden's index offers a balanced alternative if false negatives are costlier."
        ),
    }
    (RESULTS_DIR / "phase4_summary.json").write_text(json.dumps(_json_safe(summary), indent=2))
    return results


def phase4_horizons(signals: dict, meta: pd.DataFrame) -> list:
    print("Phase 4f: prediction horizons")
    rows = []
    for offset in [0, 15, 30, 60]:
        feat = build_feature_table(
            signals, meta, end_offset_min=float(offset), window_min=30.0, include_demographics=False
        )
        # Require enough samples in window
        feat = feat.loc[feat["duration_min"] >= 20].copy()
        res = evaluate_setting(feat, include_demographics=False, label=f"offset_{offset}")
        if "error" in res:
            continue
        rows.append(
            {
                "offset_min": offset,
                "auc": res["auc"],
                "auc_lo": res["auc_ci"][0],
                "auc_hi": res["auc_ci"][1],
                "n": res["n"],
                "n_pos": res["n_pos"],
            }
        )
        plot_roc_with_ci(res, out_name=f"04f_roc_offset_{offset}.png", title=f"ROC — window ending {offset} min before birth")
    plot_horizon_aucs(rows)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "horizon_aucs.csv", index=False)
    return rows


def phase5_subgroups(feat: pd.DataFrame) -> list:
    print("Phase 5: subgroup analysis")
    rows = []

    def _eval_mask(mask, subgroup, level):
        sub = feat.loc[mask].copy()
        if sub["label"].nunique() < 2 or len(sub) < 40:
            return
        res = evaluate_setting(sub, include_demographics=False, label=f"{subgroup}_{level}")
        if "error" in res:
            return
        rows.append({"subgroup": subgroup, "level": level, "auc": res["auc"], "n": res["n"], "n_pos": res["n_pos"]})

    if "Age" in feat.columns:
        med = feat["Age"].median()
        _eval_mask(feat["Age"] <= med, "maternal_age", "younger")
        _eval_mask(feat["Age"] > med, "maternal_age", "older")
    if "Weight_g" in feat.columns:
        med = feat["Weight_g"].median()
        _eval_mask(feat["Weight_g"] <= med, "birth_weight", "lower")
        _eval_mask(feat["Weight_g"] > med, "birth_weight", "higher")
    if "Gest_weeks" in feat.columns:
        med = feat["Gest_weeks"].median()
        _eval_mask(feat["Gest_weeks"] <= med, "gestation", "earlier")
        _eval_mask(feat["Gest_weeks"] > med, "gestation", "later")
    if "fhr_missing_frac" in feat.columns:
        med = feat["fhr_missing_frac"].median()
        _eval_mask(feat["fhr_missing_frac"] <= med, "fhr_missing", "low")
        _eval_mask(feat["fhr_missing_frac"] > med, "fhr_missing", "high")
    if "rec_type_label" in feat.columns:
        for level, mask in feat.groupby("rec_type_label").groups.items():
            _eval_mask(feat.index.isin(mask), "acquisition", str(level))

    if rows:
        plot_subgroup_aucs(rows)
        pd.DataFrame(rows).to_csv(RESULTS_DIR / "subgroup_aucs.csv", index=False)
    (RESULTS_DIR / "phase5_summary.json").write_text(
        json.dumps(
            {
                "rows": rows,
                "bias_discussion": (
                    "Performance often differs by signal quality and acquisition method — "
                    "ultrasound traces with more missingness are harder. Demographic gaps "
                    "(age, weight) may reflect clinical risk correlations with pH rather than "
                    "pure algorithmic unfairness; still, disparate error rates warrant caution "
                    "before clinical use."
                ),
            },
            indent=2,
        )
    )
    return rows


def phase6_shap(result: dict, feat: pd.DataFrame) -> dict:
    print("Phase 6: SHAP feature importance")
    model = result["model"]
    X = result["X"]
    # TreeExplainer on the fitted RF inside the pipeline; explain imputed+scaled features
    imputer = model.named_steps["imputer"]
    scaler = model.named_steps["scaler"]
    clf = model.named_steps["clf"]
    X_imp = imputer.transform(X)
    X_scaled = scaler.transform(X_imp)
    explainer = shap.TreeExplainer(clf)
    shap_vals = explainer.shap_values(X_scaled)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]  # positive class
    elif getattr(shap_vals, "ndim", 0) == 3:
        shap_vals = shap_vals[:, :, 1]
    plot_shap_importance(np.asarray(shap_vals), list(X.columns))

    # Attach record ids for FP/FN discussion
    X_ids = feat.loc[X.index, ["record_id", "pH"]].reset_index(drop=True) if "record_id" in feat.columns else None
    # Re-align: evaluate_setting X may not share index with feat if filtered
    # Rebuild with record_id
    scored = feat.loc[feat["record_id"].isin(feat["record_id"])].copy()
    # Use false examples from result matrices
    examples = false_examples(result, n_each=1)
    fp = examples["false_positives"]
    fn = examples["false_negatives"]

    # Map back via matching feature rows is brittle; instead pick by scores from full feat reorder
    y = result["y_true"]
    s = result["y_score"]
    thr = result["operating_point_youden"]["threshold"]
    pred = (s >= thr).astype(int)
    # Use the same row order as X
    detail = X.copy()
    detail["y_true"] = y
    detail["y_score"] = s
    detail["y_pred"] = pred
    if "record_id" in feat.columns:
        # X rows correspond to feat rows after prepare_xy filtering — same order if no reindex
        detail["record_id"] = feat.loc[X.index, "record_id"].values
        detail["pH"] = feat.loc[X.index, "pH"].values

    fp_row = detail[(detail.y_true == 0) & (detail.y_pred == 1)].sort_values("y_score", ascending=False).head(1)
    fn_row = detail[(detail.y_true == 1) & (detail.y_pred == 0)].sort_values("y_score", ascending=True).head(1)

    mean_abs = np.abs(shap_vals).mean(axis=0)
    top = sorted(zip(X.columns, mean_abs), key=lambda t: -t[1])[:10]

    out = {
        "top_features": [{"feature": f, "mean_abs_shap": float(v)} for f, v in top],
        "false_positive": fp_row.to_dict(orient="records"),
        "false_negative": fn_row.to_dict(orient="records"),
        "discussion": (
            "SHAP typically highlights decelerations, low variability, baseline abnormalities, "
            "and missingness-related features. A false positive often shows decelerations / "
            "low variability without acidosis (non-specific CTG patterns). A false negative "
            "may have relatively reassuring FHR features despite low pH — reminding us that "
            "CTG is an imperfect proxy for cord gases."
        ),
    }
    (RESULTS_DIR / "phase6_shap.json").write_text(json.dumps(_json_safe(out), indent=2))
    detail.to_csv(RESULTS_DIR / "scored_predictions.csv", index=False)
    return out


def phase7_risk(signals: dict, meta: pd.DataFrame, model_result: dict, record_id: str | None = None):
    print("Phase 7: time-varying risk score")
    # Fit model on full last-30 feature table already done; reuse pipeline
    model = model_result["model"]
    cols = model_result["feature_cols"]

    # Choose a hypoxic example with enough duration
    meta_i = meta.set_index("record_id")
    if record_id is None:
        cands = meta.loc[meta["hypoxia_7_20"] == 1].sort_values("pH")
        record_id = cands.iloc[0]["record_id"]

    fhr = signals[record_id]["fhr"]
    uc = signals[record_id]["uc"]
    duration_min = len(fhr) / 4.0 / 60.0
    offsets = np.arange(0, min(60, max(0, duration_min - 30)) + 1, 2.0)
    risks = []
    times = []
    from .features import extract_window_features

    for off in offsets:
        feats = extract_window_features(fhr, uc, end_offset_min=float(off), window_min=30.0)
        if not feats:
            continue
        row = pd.DataFrame([{c: feats.get(c, np.nan) for c in cols}])
        proba = float(model.predict_proba(row)[0, 1])
        risks.append(proba)
        times.append(float(off))

    plot_risk_over_time(times, risks, label=f"record {record_id}, pH={meta_i.loc[record_id,'pH']:.2f}")
    out = {"record_id": record_id, "times_min_before_delivery": times, "risk": risks}
    (RESULTS_DIR / "phase7_risk.json").write_text(json.dumps(_json_safe(out), indent=2))
    return out


def write_report_skeleton(phase_summaries: dict):
    path = RESULTS_DIR / "FINDINGS.md"
    p1 = phase_summaries.get("phase1", {})
    p4 = phase_summaries.get("phase4", {})
    text = f"""# Assignment 5 Findings (auto-generated draft)

## 1. Dataset exploration
- N = {p1.get('n_records')}
- Positive rate pH < 7.20: {p1.get('positive_rate_ph_lt_7_20')}
- Positive rate pH < 7.05: {p1.get('positive_rate_ph_lt_7_05')}
- Example normal / hypoxic records: {p1.get('example_normal')} / {p1.get('example_hypoxic')}
- Imbalance: {p1.get('imbalance_notes')}

See figures: `figures/01*.png`.

## 2. Signal missingness
See `results/phase2_summary.json` and `figures/02a_missingness.png`.
Short gaps (<10 s) are interpolated; longer gaps remain missing.

## 3. Feature definitions
FIGO-inspired segmentation of accelerations, decelerations, and contractions.
See `figures/03_segmentation_example.png` and `results/features_last30min.csv`.

## 4. Classification
- Signal-only AUC (last 30 min): {p4.get('baseline_auc')} CI {p4.get('baseline_auc_ci')}
- Spec≈0.90 operating point: {p4.get('baseline_op_spec90')}
- With demographics AUC: {p4.get('with_demographics_auc')}
- Quality-filtered AUC: {p4.get('quality_filtered_auc')} (n={p4.get('quality_filtered_n')})
- Horizons: see `results/horizon_aucs.csv` and `figures/04f_horizon_auc.png`

## 5–7. Subgroups, SHAP, risk over time
See `results/subgroup_aucs.csv`, `results/phase6_shap.json`, `results/phase7_risk.json`,
and corresponding figures.

## References
- CTU-UHB dataset (PhysioNet): https://physionet.org/content/ctu-uhb-ctgdb/1.0.0/
- Chiou et al., npj Women's Health 2025 — deep learning for CTG interpretation
- Asiedu & Young-Lin, Google Research blog — Predicting fetal well-being from CTG
"""
    path.write_text(text)


def main():
    print(f"Data directory: {DATA_DIR}")
    assert DATA_DIR.exists(), f"Missing dataset at {DATA_DIR}"

    meta = load_metadata_table()
    meta.to_csv(RESULTS_DIR / "metadata.csv", index=False)
    print(f"Loaded metadata for {len(meta)} records")

    print("Loading signals (this may take a minute)...")
    signals = load_all_signals()
    print(f"Loaded {len(signals)} signal pairs")

    p1 = phase1_exploration(meta, signals)
    miss = phase2_missingness(meta, signals)
    feat = phase3_features(meta, signals, example_id=p1["example_hypoxic"])

    # Align feat index for later SHAP mapping
    feat = feat.reset_index(drop=True)

    cls = phase4_classification(feat)
    # Persist a lightweight summary without model objects
    p4_summary = json.loads((RESULTS_DIR / "phase4_summary.json").read_text())

    horizons = phase4_horizons(signals, meta)
    phase5_subgroups(feat)
    phase6_shap(cls["baseline"], feat)
    phase7_risk(signals, meta, cls["baseline"], record_id=p1["example_hypoxic"])

    write_report_skeleton({"phase1": p1, "phase4": p4_summary})
    print(f"Done. Figures → {FIGURES_DIR}")
    print(f"Results → {RESULTS_DIR}")


if __name__ == "__main__":
    main()
