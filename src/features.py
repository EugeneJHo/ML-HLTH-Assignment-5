"""FIGO-inspired CTG feature extraction and event segmentation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import find_peaks, medfilt

from .config import FS, DEMOGRAPHIC_COLS
from .preprocess import interpolate_short_gaps, missing_fraction, prepare_signal


@dataclass
class Event:
    start: int
    end: int
    kind: str  # acceleration | deceleration | contraction
    amplitude: float


def _finite_series(x: np.ndarray) -> np.ndarray:
    y = x.astype(float).copy()
    mask = ~np.isfinite(y)
    if mask.all():
        return y
    if mask.any():
        idx = np.arange(len(y))
        valid = ~mask
        y[mask] = np.interp(idx[mask], idx[valid], y[valid])
    return y


def estimate_baseline(fhr: np.ndarray, fs: float = FS, win_sec: float = 60.0) -> np.ndarray:
    """Slow-varying baseline via median filter (~2 min), then light smoothing."""
    y = _finite_series(fhr)
    k = max(3, int(round(win_sec * fs)) | 1)  # odd
    base = medfilt(y, kernel_size=k)
    # Additional smoothing
    k2 = max(3, int(round(30 * fs)) | 1)
    base = medfilt(base, kernel_size=k2)
    return base


def detect_accel_decel(
    fhr: np.ndarray,
    baseline: np.ndarray,
    fs: float = FS,
    amp_thresh: float = 15.0,
    min_sec: float = 15.0,
) -> Tuple[List[Event], List[Event]]:
    """Detect accelerations / decelerations relative to baseline."""
    min_len = int(round(min_sec * fs))
    delta = fhr - baseline
    accel, decel = [], []

    def _runs(mask: np.ndarray, kind: str, signed: np.ndarray) -> List[Event]:
        events = []
        padded = np.concatenate([[False], mask, [False]])
        d = np.diff(padded.astype(int))
        starts = np.where(d == 1)[0]
        ends = np.where(d == -1)[0]
        for s, e in zip(starts, ends):
            if e - s < min_len:
                continue
            amp = float(np.nanmax(np.abs(signed[s:e])))
            if amp < amp_thresh:
                continue
            events.append(Event(s, e, kind, amp))
        return events

    finite = np.isfinite(fhr)
    accel = _runs((delta >= amp_thresh) & finite, "acceleration", delta)
    decel = _runs((delta <= -amp_thresh) & finite, "deceleration", delta)
    return accel, decel


def detect_contractions(
    uc: np.ndarray,
    fs: float = FS,
    min_distance_sec: float = 30.0,
    prominence: float = 10.0,
) -> List[Event]:
    y = _finite_series(uc)
    # Light smoothing
    k = max(3, int(round(2 * fs)) | 1)
    y_s = medfilt(y, kernel_size=k)
    distance = max(1, int(round(min_distance_sec * fs)))
    peaks, props = find_peaks(y_s, distance=distance, prominence=prominence)
    events = []
    half = int(round(15 * fs))
    for i, p in enumerate(peaks):
        s = max(0, p - half)
        e = min(len(y_s), p + half)
        amp = float(props["prominences"][i]) if "prominences" in props else float(y_s[p])
        events.append(Event(s, e, "contraction", amp))
    return events


def segment_events(fhr: np.ndarray, uc: np.ndarray, fs: float = FS) -> Dict:
    fhr_i = interpolate_short_gaps(fhr)
    uc_i = interpolate_short_gaps(uc)
    baseline = estimate_baseline(fhr_i, fs=fs)
    accels, decels = detect_accel_decel(fhr_i, baseline, fs=fs)
    contractions = detect_contractions(uc_i, fs=fs)
    return {
        "fhr": fhr_i,
        "uc": uc_i,
        "baseline": baseline,
        "accelerations": accels,
        "decelerations": decels,
        "contractions": contractions,
    }


def window_slice(x: np.ndarray, end_offset_min: float, window_min: float = 30.0, fs: float = FS) -> np.ndarray:
    """Slice a window ending `end_offset_min` minutes before the recording end.

    end_offset_min=0 → last `window_min` minutes.
    end_offset_min=15 → window from T-45 to T-15.
    """
    n = len(x)
    end = n - int(round(end_offset_min * 60 * fs))
    start = end - int(round(window_min * 60 * fs))
    start = max(0, start)
    end = max(start, min(n, end))
    return x[start:end]


def extract_window_features(
    fhr: np.ndarray,
    uc: np.ndarray,
    fs: float = FS,
    end_offset_min: float = 0.0,
    window_min: float = 30.0,
) -> Dict[str, float]:
    fhr_w = window_slice(fhr, end_offset_min, window_min, fs)
    uc_w = window_slice(uc, end_offset_min, window_min, fs)
    if len(fhr_w) < fs * 60:  # less than 1 minute
        return {}

    fhr_i, uc_i, qstats = prepare_signal(fhr_w, uc_w)
    seg = segment_events(fhr_i, uc_i, fs=fs)
    baseline = seg["baseline"]
    duration_min = len(fhr_i) / fs / 60.0

    finite = np.isfinite(fhr_i)
    fhr_f = fhr_i[finite]
    base_f = baseline[finite] if finite.any() else baseline

    # Short-term variability: mean absolute minute-to-minute difference proxy
    # Use std of residual around baseline
    residual = fhr_i - baseline
    stv = float(np.nanstd(residual)) if np.isfinite(residual).any() else np.nan

    feats = {
        "fhr_mean": float(np.nanmean(fhr_f)) if fhr_f.size else np.nan,
        "fhr_std": float(np.nanstd(fhr_f)) if fhr_f.size else np.nan,
        "fhr_min": float(np.nanmin(fhr_f)) if fhr_f.size else np.nan,
        "fhr_max": float(np.nanmax(fhr_f)) if fhr_f.size else np.nan,
        "baseline_mean": float(np.nanmean(base_f)) if np.size(base_f) else np.nan,
        "baseline_std": float(np.nanstd(base_f)) if np.size(base_f) else np.nan,
        "variability": stv,
        "n_accelerations": len(seg["accelerations"]),
        "n_decelerations": len(seg["decelerations"]),
        "n_contractions": len(seg["contractions"]),
        "accel_rate_per_10min": len(seg["accelerations"]) / max(duration_min, 1e-6) * 10.0,
        "decel_rate_per_10min": len(seg["decelerations"]) / max(duration_min, 1e-6) * 10.0,
        "contraction_rate_per_10min": len(seg["contractions"]) / max(duration_min, 1e-6) * 10.0,
        "mean_decel_amp": float(np.mean([e.amplitude for e in seg["decelerations"]])) if seg["decelerations"] else 0.0,
        "mean_accel_amp": float(np.mean([e.amplitude for e in seg["accelerations"]])) if seg["accelerations"] else 0.0,
        "uc_mean": float(np.nanmean(uc_i)) if np.isfinite(uc_i).any() else np.nan,
        "uc_std": float(np.nanstd(uc_i)) if np.isfinite(uc_i).any() else np.nan,
        "fhr_missing_frac": missing_fraction(fhr_w),
        "uc_missing_frac": missing_fraction(uc_w),
        "fhr_missing_after_interp": qstats["fhr_missing_after_interp"],
        "window_min": window_min,
        "end_offset_min": end_offset_min,
        "duration_min": duration_min,
    }
    return feats


def build_feature_table(
    signals: Dict[str, Dict],
    metadata,
    end_offset_min: float = 0.0,
    window_min: float = 30.0,
    include_demographics: bool = True,
) -> "pd.DataFrame":
    import pandas as pd

    rows = []
    meta_idx = metadata.set_index("record_id")
    for rid, payload in signals.items():
        feats = extract_window_features(
            payload["fhr"], payload["uc"], end_offset_min=end_offset_min, window_min=window_min
        )
        if not feats:
            continue
        feats["record_id"] = rid
        if rid in meta_idx.index:
            m = meta_idx.loc[rid]
            feats["pH"] = m["pH"]
            feats["label"] = int(m["hypoxia_7_20"])
            feats["Rec_type"] = m["Rec_type"]
            feats["rec_type_label"] = m["rec_type_label"]
            if include_demographics:
                for c in DEMOGRAPHIC_COLS:
                    if c in m.index:
                        feats[c] = m[c]
        rows.append(feats)
    return pd.DataFrame(rows)
