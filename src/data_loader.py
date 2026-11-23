"""Load CTU-UHB WFDB records and parse clinical metadata from .hea files."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import wfdb

from .config import DATA_DIR, FS, PH_CUTOFF, PH_SEVERE, REC_TYPE_MAP

HEADER_KEYS = {
    "pH": "pH",
    "BDecf": "BDecf",
    "pCO2": "pCO2",
    "BE": "BE",
    "Apgar1": "Apgar1",
    "Apgar5": "Apgar5",
    "Gest. weeks": "Gest_weeks",
    "Weight(g)": "Weight_g",
    "Sex": "Sex",
    "Age": "Age",
    "Gravidity": "Gravidity",
    "Parity": "Parity",
    "Diabetes": "Diabetes",
    "Hypertension": "Hypertension",
    "Preeclampsia": "Preeclampsia",
    "Liq. praecox": "Liq_praecox",
    "Pyrexia": "Pyrexia",
    "Meconium": "Meconium",
    "Presentation": "Presentation",
    "Induced": "Induced",
    "I.stage": "I_stage",
    "NoProgress": "NoProgress",
    "CK/KP": "CK_KP",
    "II.stage": "II_stage",
    "Deliv. type": "Deliv_type",
    "Rec. type": "Rec_type",
    "Pos. II.st.": "Pos_II_st",
    "Sig2Birth": "Sig2Birth",
    "dbID": "dbID",
}


def list_record_ids(data_dir: Path = DATA_DIR) -> List[str]:
    records_file = data_dir / "RECORDS"
    if records_file.exists():
        return [line.strip() for line in records_file.read_text().splitlines() if line.strip()]
    return sorted(p.stem for p in data_dir.glob("*.hea"))


def parse_header_metadata(hea_path: Path) -> Dict[str, float]:
    text = hea_path.read_text(encoding="utf-8", errors="ignore")
    meta: Dict[str, float] = {}
    for raw_key, col in HEADER_KEYS.items():
        # Matches lines like: #pH           7.14
        pattern = rf"^#{re.escape(raw_key)}\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$"
        m = re.search(pattern, text, flags=re.MULTILINE)
        if m:
            meta[col] = float(m.group(1))
    return meta


def load_record(record_id: str, data_dir: Path = DATA_DIR) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Return FHR, UC arrays (float, NaN for missing) and metadata dict."""
    path = data_dir / record_id
    rec = wfdb.rdrecord(str(path))
    # Channel order in CTU-UHB: FHR, UC
    sig = rec.p_signal.astype(float)
    fhr = sig[:, 0].copy()
    uc = sig[:, 1].copy() if sig.shape[1] > 1 else np.full_like(fhr, np.nan)

    # PhysioNet / CTU convention: zeros (and often values outside physiologic range) are missing
    fhr[(fhr <= 0) | (fhr > 250)] = np.nan
    uc[uc < 0] = np.nan

    meta = parse_header_metadata(data_dir / f"{record_id}.hea")
    meta["record_id"] = record_id
    meta["n_samples"] = len(fhr)
    meta["duration_min"] = len(fhr) / FS / 60.0
    meta["fs"] = FS
    return fhr, uc, meta


def load_metadata_table(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    rows = []
    for rid in list_record_ids(data_dir):
        meta = parse_header_metadata(data_dir / f"{rid}.hea")
        meta["record_id"] = rid
        # Duration from header first line: n_samples at fs
        hea = (data_dir / f"{rid}.hea").read_text(encoding="utf-8", errors="ignore").splitlines()[0]
        parts = hea.split()
        n_samples = int(parts[3])
        fs = float(parts[2])
        meta["n_samples"] = n_samples
        meta["duration_min"] = n_samples / fs / 60.0
        rows.append(meta)

    df = pd.DataFrame(rows)
    df["hypoxia_7_20"] = (df["pH"] < PH_CUTOFF).astype(int)
    df["hypoxia_7_05"] = (df["pH"] < PH_SEVERE).astype(int)
    df["rec_type_label"] = df["Rec_type"].map(lambda x: REC_TYPE_MAP.get(int(x), f"Unknown({x})"))
    return df.sort_values("record_id").reset_index(drop=True)


def load_all_signals(
    data_dir: Path = DATA_DIR,
    record_ids: Optional[List[str]] = None,
) -> Dict[str, Dict]:
    """Load all (or selected) records into memory."""
    ids = record_ids or list_record_ids(data_dir)
    out: Dict[str, Dict] = {}
    for rid in ids:
        fhr, uc, meta = load_record(rid, data_dir)
        out[rid] = {"fhr": fhr, "uc": uc, "meta": meta}
    return out
