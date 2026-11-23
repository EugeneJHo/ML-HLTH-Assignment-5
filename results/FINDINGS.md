# Assignment 5 Findings (auto-generated draft)

## 1. Dataset exploration
- N = 552
- Positive rate pH < 7.20: 0.32065217391304346
- Positive rate pH < 7.05: 0.07246376811594203
- Example normal / hypoxic records: 1409 / 1180
- Imbalance: Class imbalance (~32% positive at pH<7.20; ~7% at pH<7.05). Mitigations: class_weight='balanced', oversampling of minority crops, threshold tuning on ROC, and reporting sensitivity at fixed high specificity.

See figures: `figures/01*.png`.

## 2. Signal missingness
See `results/phase2_summary.json` and `figures/02a_missingness.png`.
Short gaps (<10 s) are interpolated; longer gaps remain missing.

## 3. Feature definitions
FIGO-inspired segmentation of accelerations, decelerations, and contractions.
See `figures/03_segmentation_example.png` and `results/features_last30min.csv`.

## 4. Classification
- Signal-only AUC (last 30 min): 0.7010678564667673 CI [0.6472213639753871, 0.7442158707865169]
- Spec≈0.90 operating point: {'threshold': 0.654769040314894, 'sensitivity': 0.3559322033898305, 'specificity': 0.9013333333333333, 'tp': 63, 'fp': 37, 'tn': 338, 'fn': 114}
- With demographics AUC: 0.7199885295856049
- Quality-filtered AUC: 0.5352953791021935 (n=148)
- Horizons: see `results/horizon_aucs.csv` and `figures/04f_horizon_auc.png`

## 5–7. Subgroups, SHAP, risk over time
See `results/subgroup_aucs.csv`, `results/phase6_shap.json`, `results/phase7_risk.json`,
and corresponding figures.

## References
- CTU-UHB dataset (PhysioNet): https://physionet.org/content/ctu-uhb-ctgdb/1.0.0/
- Chiou et al., npj Women's Health 2025 — deep learning for CTG interpretation
- Asiedu & Young-Lin, Google Research blog — Predicting fetal well-being from CTG
