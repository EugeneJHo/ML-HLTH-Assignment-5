# ML-HLTH Assignment 5 — Clinical CTG time series

Machine learning for fetal hypoxia detection from the **CTU-UHB Intrapartum Cardiotocography Database** (552 records of FHR + uterine contractions at 4 Hz, with cord-blood pH and clinical metadata).

**Label:** fetal acidosis / hypoxia defined as umbilical artery cord **pH < 7.20**.

GitHub: https://github.com/EugeneJHo/ML-HLTH-Assignment-5

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place (or symlink) the PhysioNet CTU-UHB release so that either of these paths exists:

- `data/ctu-uhb/` (preferred)
- `ctu-chb-intrapartum-cardiotocography-database-1.0.0/`

Dataset source: https://physionet.org/content/ctu-uhb-ctgdb/1.0.0/

## Run the full analysis

```bash
source .venv/bin/activate
python -m src.run_all
```

Outputs:

| Path | Contents |
|------|----------|
| `figures/` | Plots for exploration, missingness, segmentation, ROC, subgroups, SHAP, risk |
| `results/` | CSV/JSON summaries + draft `FINDINGS.md` |
| `report/` | Place the final slide deck / PDF here |

## Pipeline map (assignment phases)

1. **Exploration** — variables, example traces, pH distribution / imbalance, acquisition method, demographic correlations with pH  
2. **Missingness** — dropout by outcome and acquisition; effect of a ≤10% missingness filter; short-gap (<10 s) interpolation  
3. **Features** — baseline, variability, accelerations / decelerations / contractions segmentation  
4. **Classification** — last-30-min feature vector, stratified CV RandomForest, ROC + bootstrap CI, demographics ablation, quality filter, earlier horizons (15 / 30 / 60 min)  
5. **Subgroups** — performance by age, birth weight, gestation, missingness, acquisition  
6. **SHAP** — global importance + FP/FN discussion hooks  
7. **Time-varying risk** — sliding 30-min window scores before delivery  

## Notes

- Classifier uses `class_weight="balanced"` to address label imbalance.
- Short FHR/UC gaps **&lt; 10 s** are linearly interpolated; longer gaps stay missing (assignment convention).
- This is research / coursework analysis on a public de-identified dataset — not a clinical device.
