"""Signal quality / missingness utilities and short-gap interpolation."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .config import FS, SHORT_GAP_SEC


def missing_mask(x: np.ndarray) -> np.ndarray:
    return ~np.isfinite(x)


def missing_fraction(x: np.ndarray) -> float:
    if len(x) == 0:
        return 1.0
    return float(np.mean(missing_mask(x)))


def gap_lengths_samples(mask: np.ndarray) -> np.ndarray:
    """Return lengths (in samples) of contiguous True runs."""
    if mask.size == 0:
        return np.array([], dtype=int)
    padded = np.concatenate([[False], mask, [False]])
    diff = np.diff(padded.astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return ends - starts


def interpolate_short_gaps(x: np.ndarray, max_gap_sec: float = SHORT_GAP_SEC, fs: float = FS) -> np.ndarray:
    """Linearly interpolate NaN gaps shorter than max_gap_sec; leave longer gaps as NaN."""
    y = x.astype(float).copy()
    mask = missing_mask(y)
    if not mask.any():
        return y

    max_gap = int(round(max_gap_sec * fs))
    padded = np.concatenate([[False], mask, [False]])
    diff = np.diff(padded.astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    idx = np.arange(len(y))
    valid = ~mask
    if valid.sum() < 2:
        return y

    for s, e in zip(starts, ends):
        gap_len = e - s
        if gap_len <= max_gap and gap_len > 0:
            left = s - 1
            right = e
            if left >= 0 and right < len(y) and np.isfinite(y[left]) and np.isfinite(y[right]):
                y[s:e] = np.linspace(y[left], y[right], gap_len + 2)[1:-1]
            elif valid.sum() >= 2:
                # fallback: interpolate from global valid samples for short edge gaps
                y[s:e] = np.interp(idx[s:e], idx[valid], y[valid])
    return y


def smooth_signal(x: np.ndarray, window_sec: float = 0.75, fs: float = FS) -> np.ndarray:
    """Simple moving average on finite samples (preserves NaNs)."""
    w = max(1, int(round(window_sec * fs)))
    if w % 2 == 0:
        w += 1
    y = x.astype(float).copy()
    finite = np.isfinite(y)
    if finite.sum() == 0:
        return y
    # Temporarily fill for smoothing, then restore long gaps
    filled = y.copy()
    idx = np.arange(len(y))
    filled[~finite] = np.interp(idx[~finite], idx[finite], y[finite])
    kernel = np.ones(w) / w
    smoothed = np.convolve(filled, kernel, mode="same")
    # Restore samples that were originally missing (long gaps stay missing after short-gap fill)
    out = smoothed
    out[~finite] = np.nan
    return out


def prepare_signal(fhr: np.ndarray, uc: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Apply short-gap interpolation; report quality stats before/after."""
    stats = {
        "fhr_missing_raw": missing_fraction(fhr),
        "uc_missing_raw": missing_fraction(uc),
        "fhr_long_gaps": int(np.sum(gap_lengths_samples(missing_mask(fhr)) >= int(SHORT_GAP_SEC * FS))),
        "uc_long_gaps": int(np.sum(gap_lengths_samples(missing_mask(uc)) >= int(SHORT_GAP_SEC * FS))),
    }
    fhr_i = interpolate_short_gaps(fhr)
    uc_i = interpolate_short_gaps(uc)
    stats["fhr_missing_after_interp"] = missing_fraction(fhr_i)
    stats["uc_missing_after_interp"] = missing_fraction(uc_i)
    return fhr_i, uc_i, stats


def cohort_missingness_table(signals: Dict[str, Dict], metadata: pd.DataFrame) -> pd.DataFrame:
    rows = []
    meta_idx = metadata.set_index("record_id")
    for rid, payload in signals.items():
        fhr, uc = payload["fhr"], payload["uc"]
        fhr_i, uc_i, stats = prepare_signal(fhr, uc)
        row = {
            "record_id": rid,
            **stats,
            "fhr_missing": stats["fhr_missing_after_interp"],
            "uc_missing": stats["uc_missing_after_interp"],
        }
        if rid in meta_idx.index:
            m = meta_idx.loc[rid]
            row["pH"] = m["pH"]
            row["hypoxia_7_20"] = int(m["hypoxia_7_20"])
            row["Rec_type"] = m["Rec_type"]
            row["rec_type_label"] = m["rec_type_label"]
        rows.append(row)
    return pd.DataFrame(rows)


def apply_quality_filter(miss_df: pd.DataFrame, max_missing: float = 0.10, channel: str = "fhr_missing") -> pd.DataFrame:
    return miss_df.loc[miss_df[channel] <= max_missing].copy()
