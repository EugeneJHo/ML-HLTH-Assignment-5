"""Shared paths and analysis constants for Assignment 5."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "ctu-uhb"
if not DATA_DIR.exists():
    DATA_DIR = ROOT / "ctu-chb-intrapartum-cardiotocography-database-1.0.0"

FIGURES_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"
REPORT_DIR = ROOT / "report"

for d in (FIGURES_DIR, RESULTS_DIR, REPORT_DIR):
    d.mkdir(parents=True, exist_ok=True)

FS = 4.0  # Hz
PH_CUTOFF = 7.20
PH_SEVERE = 7.05
SHORT_GAP_SEC = 10.0  # interpolate gaps shorter than this
QUALITY_MISSING_MAX = 0.10  # discard if >10% missing
LAST_MINUTES = 30
RANDOM_STATE = 42
N_CV_FOLDS = 5
N_BOOTSTRAP = 200

# Rec. type codes from CTU-UHB / PhysioNet documentation
REC_TYPE_MAP = {
    1: "Doppler ultrasound",
    2: "fECG (scalp electrode)",
    12: "Combined US + fECG",
}

DEMOGRAPHIC_COLS = [
    "Age",
    "Gravidity",
    "Parity",
    "Gest_weeks",
    "Weight_g",
    "Sex",
    "Diabetes",
    "Hypertension",
    "Preeclampsia",
    "Liq_praecox",
    "Pyrexia",
    "Meconium",
    "Induced",
    "Deliv_type",
    "I_stage",
    "II_stage",
]
