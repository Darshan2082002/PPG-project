"""
signal_processing.py
====================
PPG signal conditioning pipeline:
  1. Bandpass filtering   (0.5 – 8 Hz, Butterworth 4th order)
  2. Resampling           (any Fs → TARGET_FS = 125 Hz)
  3. Windowing / segmentation (10 s windows, 50 % overlap)
  4. Min-max normalisation per window → [0, 1]

All operations are applied consistently to both MIMIC-IV and PTT signals
so they share the same input shape for the deep learning model.
"""

import numpy as np
from scipy import signal as sp_signal
import logging

logger = logging.getLogger(__name__)

TARGET_FS   = 125      # Hz  – model input sampling rate
WINDOW_SEC  = 10       # seconds per segment
OVERLAP     = 0.5      # 50 % overlap between consecutive windows
BP_LOW      = 0.5      # bandpass low cut-off  (Hz)
BP_HIGH     = 8.0      # bandpass high cut-off (Hz)
BP_ORDER    = 4        # Butterworth filter order
MIN_WINDOW_SAMPLES = int(TARGET_FS * WINDOW_SEC)  # 1250 samples


# ── Bandpass filter ────────────────────────────────────────────────────────────

def bandpass_filter(ppg: np.ndarray, fs: float) -> np.ndarray:
    """
    Apply a 4th-order Butterworth bandpass filter (0.5 – 8 Hz).

    Parameters
    ----------
    ppg : np.ndarray, 1-D
    fs  : float  – sampling frequency in Hz

    Returns
    -------
    filtered : np.ndarray, 1-D, same length as ppg
    """
    nyq = fs / 2.0
    low  = BP_LOW  / nyq
    high = BP_HIGH / nyq

    # Clamp to valid range
    low  = max(low,  1e-4)
    high = min(high, 0.9999)

    if low >= high:
        logger.warning("Invalid bandpass range for fs=%.1f Hz – returning raw", fs)
        return ppg

    b, a = sp_signal.butter(BP_ORDER, [low, high], btype="bandpass")
    try:
        # filtfilt requires matching dtypes; cast to float64, return float32
        filtered = sp_signal.filtfilt(b, a, ppg.astype(np.float64)).astype(np.float32)
    except Exception as e:
        logger.warning("filtfilt failed (%s) – returning raw", e)
        filtered = ppg.astype(np.float32)
    return filtered


# ── Resampling ─────────────────────────────────────────────────────────────────

import math

def get_rational_ratio(src: float, target: float) -> tuple[int, int]:
    factor = 1000
    up = int(target * factor)
    down = int(src * factor)
    g = math.gcd(up, down)
    return up // g, down // g

def resample_to_target(ppg: np.ndarray, src_fs: float) -> np.ndarray:
    """
    Resample ppg from src_fs to TARGET_FS using polyphase resampling.
    This avoids std::bad_alloc MemoryErrors on very long signals.
    """
    if abs(src_fs - TARGET_FS) < 0.1:
        return ppg  # already at target rate

    up, down = get_rational_ratio(src_fs, float(TARGET_FS))
    try:
        resampled = sp_signal.resample_poly(ppg, up, down).astype(np.float32)
    except Exception as e:
        logger.warning(f"resample_poly failed ({e}), falling back to linear interpolation")
        num_samples = int(round(len(ppg) * TARGET_FS / src_fs))
        x_old = np.linspace(0, 1, len(ppg))
        x_new = np.linspace(0, 1, num_samples)
        resampled = np.interp(x_new, x_old, ppg).astype(np.float32)
    
    return resampled


# ── Normalisation ──────────────────────────────────────────────────────────────

def minmax_normalize(window: np.ndarray) -> np.ndarray:
    """
    Scale a single window to [0, 1].  Returns zeros if flat (std ≈ 0).
    """
    mn, mx = window.min(), window.max()
    rng = mx - mn
    if rng < 1e-6:
        return np.zeros_like(window)
    return ((window - mn) / rng).astype(np.float32)


# ── Segmentation ───────────────────────────────────────────────────────────────

# Maximum windows kept per single recording to avoid memory blowouts.
# Integrated GPU (AMD Vega 10, 6 GB shared) needs a low cap.
# 50 windows × 2838 MIMIC segs ≈ 142K total – manageable.
MAX_WINDOWS_PER_RECORDING = 50


def segment_signal(
    ppg: np.ndarray,
    fs: float = TARGET_FS,
    window_sec: float = WINDOW_SEC,
    overlap: float = OVERLAP,
    max_windows: int = MAX_WINDOWS_PER_RECORDING,
) -> np.ndarray:
    """
    Split a continuous PPG into overlapping fixed-length windows
    and normalise each window independently.

    Parameters
    ----------
    ppg        : np.ndarray, 1-D (already at target_fs)
    fs         : float
    window_sec : float
    overlap    : float  (0 to < 1)
    max_windows: int    – cap on windows kept per recording (avoids OOM)

    Returns
    -------
    windows : np.ndarray, shape (n_windows, window_samples, 1)
    """
    win_len  = int(fs * window_sec)
    step_len = int(win_len * (1.0 - overlap))

    if len(ppg) < win_len:
        # Pad with zeros if signal is shorter than one window
        pad = np.zeros(win_len - len(ppg), dtype=np.float32)
        ppg = np.concatenate([ppg, pad])

    # Pre-calculate total possible windows and cap early
    n_possible = max(0, (len(ppg) - win_len) // step_len + 1)
    n_alloc    = min(n_possible, max_windows)

    if n_alloc == 0:
        return np.empty((0, win_len, 1), dtype=np.float32)

    # Pre-allocate output buffer – avoids building a Python list of arrays
    buf   = np.empty((n_alloc, win_len), dtype=np.float32)
    count = 0
    start = 0

    while start + win_len <= len(ppg) and count < max_windows:
        win = ppg[start:start + win_len]
        win = minmax_normalize(win)
        # Skip flat / near-flat windows (artefact / flatline)
        if win.std() > 1e-4:
            if count < n_alloc:
                buf[count] = win
            else:
                # Buffer full (shouldn't happen given cap), stop
                break
            count += 1
        start += step_len

    if count == 0:
        return np.empty((0, win_len, 1), dtype=np.float32)

    return buf[:count, :, np.newaxis]   # (n, win_len, 1)


# ── Full pipeline for a single recording ──────────────────────────────────────

def process_ppg(
    ppg_raw: np.ndarray,
    src_fs: float,
) -> np.ndarray:
    """
    Full conditioning pipeline for one PPG recording.

    Steps:
      1. Resample to TARGET_FS
      2. Bandpass filter (0.5 – 8 Hz)
      3. Segment into overlapping windows
      4. Normalise each window

    Parameters
    ----------
    ppg_raw : np.ndarray, 1-D
    src_fs  : float  – original sampling frequency

    Returns
    -------
    windows : np.ndarray, shape (n_windows, 1250, 1)
    """
    # 1. Resample
    ppg_rs = resample_to_target(ppg_raw, src_fs)

    # 2. Filter
    ppg_filt = bandpass_filter(ppg_rs, TARGET_FS)

    # 3. Segment + normalise
    windows = segment_signal(ppg_filt)

    return windows


# ── Batch processing ───────────────────────────────────────────────────────────

def batch_process(
    ppg_list: list,
    labels:   list,
    src_fs:   float,
) -> tuple:
    """
    Apply the full pipeline to a list of raw PPG arrays.

    Returns
    -------
    X : np.ndarray, shape (total_windows, 1250, 1)
    y : np.ndarray, shape (total_windows,)
    """
    X_parts, y_parts = [], []
    skipped = 0

    for ppg, label in zip(ppg_list, labels):
        wins = process_ppg(ppg, src_fs)
        if wins.shape[0] == 0:
            skipped += 1
            continue
        X_parts.append(wins)
        y_parts.append(np.full(wins.shape[0], label, dtype=np.int32))

    if not X_parts:
        return np.empty((0, MIN_WINDOW_SAMPLES, 1), np.float32), np.empty((0,), np.int32)

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    logger.info("Batch processing complete: %d windows from %d recordings (%d skipped)",
                len(X), len(ppg_list) - skipped, skipped)
    return X, y


if __name__ == "__main__":
    # Quick test
    fs = 62.5
    t = np.linspace(0, 120, int(fs * 120))
    ppg_test = (np.sin(2 * np.pi * 1.1 * t) + 0.3 * np.random.randn(len(t))).astype(np.float32)
    wins = process_ppg(ppg_test, fs)
    print(f"Output shape: {wins.shape}")   # expected: (N, 1250, 1)
    print(f"Value range:  [{wins.min():.3f}, {wins.max():.3f}]")
