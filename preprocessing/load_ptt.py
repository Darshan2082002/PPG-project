"""
load_ptt.py
===========
Loads PPG and accelerometer signals from the Pulse Transit Time (PTT) PPG Dataset.

Dataset info:
  - 22 healthy subjects × 3 activities (sit, walk, run)
  - 500 Hz sampling rate (PPG + ACC same clock)
  - CSV files with columns: time, ecg, peaks, pleth_1..6, lc_1, lc_2,
                             temp_1..3, a_x, a_y, a_z, g_x, g_y, g_z
  - All subjects are healthy → label = 0 (Normal Sinus Rhythm)

PPG channel used: pleth_2 (highest variance / dynamic range per dataset analysis)
ACC channels:     a_x, a_y, a_z (for motion artifact removal)
"""

import os
import glob
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.join(
    os.path.dirname(__file__), "..",
    "Pulse Transit Time PPG Dataset",
    "physionet.org", "files", "pulse-transit-time-ppg", "1.1.0", "csv"
)

PTT_FS      = 500   # Hz – native sampling rate
TARGET_FS   = 125   # Hz – target (downsample 4x)
PPG_CHANNEL = "pleth_2"
ACC_CHANNELS = ["a_x", "a_y", "a_z"]

# Subjects excluded (s18_walk has no .dat file, s21_sit has no .dat file,
# s5_run and others may have missing headers – handled gracefully)
EXCLUDED = {"s18_walk", "s21_sit"}


def _load_csv(csv_path: str) -> pd.DataFrame | None:
    """Read a PTT CSV file with robust error handling."""
    try:
        df = pd.read_csv(csv_path, on_bad_lines="skip")
        return df
    except Exception as e:
        logger.warning("Failed to read %s: %s", csv_path, e)
        return None


def _downsample(arr: np.ndarray, factor: int) -> np.ndarray:
    """Simple decimation by factor (anti-alias via numpy slicing).
    For production use, use scipy.signal.decimate with an AA filter."""
    return arr[::factor]


def load_ptt(max_files: int = None, verbose: bool = True):
    """
    Load all PTT CSV files and return PPG arrays + ACC arrays.

    Parameters
    ----------
    max_files : int or None
        Limit number of CSV files to load (for quick testing).
    verbose : bool
        Print progress info.

    Returns
    -------
    ppg_list : list of np.ndarray  – PPG at TARGET_FS, shape (N,)
    acc_list : list of np.ndarray  – ACC at TARGET_FS, shape (N, 3)
    labels   : list of int         – always 0 (Normal)
    fs       : float               – TARGET_FS
    """
    base = os.path.abspath(BASE_DIR)
    if not os.path.isdir(base):
        raise FileNotFoundError(f"PTT CSV directory not found: {base}")

    csv_files = sorted(glob.glob(os.path.join(base, "s*.csv")))
    if max_files:
        csv_files = csv_files[:max_files]

    ppg_list: list[np.ndarray] = []
    acc_list: list[np.ndarray] = []
    labels:   list[int]        = []

    downsample_factor = PTT_FS // TARGET_FS  # 4

    loaded = 0
    skipped = 0

    for csv_path in csv_files:
        fname = os.path.splitext(os.path.basename(csv_path))[0]

        if fname in EXCLUDED:
            logger.debug("Skipping excluded recording: %s", fname)
            skipped += 1
            continue

        df = _load_csv(csv_path)
        if df is None:
            skipped += 1
            continue

        # ── Validate required columns ──────────────────────────────────────────
        required = [PPG_CHANNEL] + ACC_CHANNELS
        missing = [c for c in required if c not in df.columns]
        if missing:
            logger.warning("Missing columns %s in %s – skipping", missing, fname)
            skipped += 1
            continue

        # ── Extract signals ────────────────────────────────────────────────────
        ppg_raw = df[PPG_CHANNEL].values.astype(np.float32)
        acc_raw = df[ACC_CHANNELS].values.astype(np.float32)

        # Drop rows where PPG or any ACC axis is NaN
        valid_mask = ~(np.isnan(ppg_raw) | np.any(np.isnan(acc_raw), axis=1))
        ppg_raw = ppg_raw[valid_mask]
        acc_raw = acc_raw[valid_mask]

        if len(ppg_raw) < TARGET_FS * 5:  # must be at least 5 seconds
            logger.debug("Recording too short after NaN removal: %s", fname)
            skipped += 1
            continue

        # ── Downsample 500 → 125 Hz ────────────────────────────────────────────
        ppg_ds = _downsample(ppg_raw, downsample_factor)
        acc_ds = np.stack([_downsample(acc_raw[:, i], downsample_factor)
                           for i in range(3)], axis=1)

        ppg_list.append(ppg_ds)
        acc_list.append(acc_ds)
        labels.append(0)  # PTT = healthy = Normal
        loaded += 1

    if verbose:
        logger.info("PTT Dataset: %d recordings loaded, %d skipped", loaded, skipped)
        logger.info("  All labelled as Class 0 (Normal Sinus Rhythm)")
        if ppg_list:
            durations = [len(p) / TARGET_FS for p in ppg_list]
            logger.info("  Duration: min=%.1fs  mean=%.1fs  max=%.1fs",
                        min(durations), np.mean(durations), max(durations))

    return ppg_list, acc_list, labels, float(TARGET_FS)


if __name__ == "__main__":
    ppg, acc, labs, fs = load_ptt(max_files=5)
    logger.info("Example PPG shape: %s @ %.0f Hz", ppg[0].shape if ppg else "N/A", fs)
    logger.info("Example ACC shape: %s", acc[0].shape if acc else "N/A")
