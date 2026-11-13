"""
feature_extraction.py
=====================
Beat-level and window-level feature extraction from segmented PPG signals.

Extracts a rich feature vector per window that spans four categories:
  1. Time-domain & Statistical  – mean, variance, skewness, kurtosis, ZCR
  2. PPG Morphology              – pulse amplitude, peak-to-peak interval,
                                   pulse width at 50 % height, dicrotic notch
  3. Heart Rate Variability (HRV) – SDNN, RMSSD, pNN50, LF/HF power ratio
  4. Frequency-domain            – spectral centroid, dominant frequency,
                                   band-power ratios

Input
-----
  outputs/dataset.npz  – pre-processed PPG windows from fuse_datasets.py
  Shape: X_{train,val,test}  (N, 1250, 1)  @ 125 Hz  ->  10-second windows
  Labels: y_{train,val,test}  (N,)

Output
------
  outputs/features_train.parquet
  outputs/features_val.parquet
  outputs/features_test.parquet

  Each parquet file is a DataFrame with one row per window and columns:
    feat_* (float32)  +  label (int32)

Usage
-----
  python feature_extraction.py
  python feature_extraction.py --fs 125 --input outputs/dataset.npz
"""

from __future__ import annotations

import os
import sys
import argparse
import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy import signal as sp_signal
from scipy.stats import skew, kurtosis
from tqdm import tqdm

# Suppress the scipy Lomb-Scargle RuntimeWarning that fires when an RR
# interval series is too short or has zero variance (degenerate windows).
warnings.filterwarnings(
    "ignore",
    message="invalid value encountered",
    category=RuntimeWarning,
    module="scipy",
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# -- Constants -----------------------------------------------------------------
FS_DEFAULT   = 125      # sampling frequency (Hz) as set by fuse_datasets.py
NUM_CLASSES  = 5
OUTPUT_DIR   = os.path.join(os.path.dirname(__file__), "outputs")
DATASET_PATH = os.path.join(OUTPUT_DIR, "dataset.npz")


# ==============================================================================
#  PARQUET / CSV STORAGE HELPERS
# ==============================================================================

def _detect_parquet_engine() -> Optional[str]:
    """
    Probe for an available parquet engine.

    Returns the engine name ('pyarrow' or 'fastparquet') if found,
    or None if neither is installed.
    """
    for engine in ("pyarrow", "fastparquet"):
        try:
            if engine == "pyarrow":
                import pyarrow  # noqa: F401
            else:
                import fastparquet  # noqa: F401
            return engine
        except ImportError:
            continue
    return None


_PARQUET_ENGINE: Optional[str] = _detect_parquet_engine()

if _PARQUET_ENGINE is None:
    logger.warning(
        "Neither 'pyarrow' nor 'fastparquet' is installed.\n"
        "  Feature files will be saved as CSV (larger, slower) instead of Parquet.\n"
        "  To enable Parquet (recommended):\n"
        "    pip install pyarrow"
    )
else:
    logger.info("Parquet engine: %s", _PARQUET_ENGINE)


def save_features(df: pd.DataFrame, path: str) -> str:
    """
    Save a feature DataFrame to Parquet if an engine is available,
    otherwise fall back to CSV.

    Parameters
    ----------
    df   : DataFrame to save (columns = feature names + 'label')
    path : desired output path (*.parquet or *.csv)

    Returns
    -------
    str  -- actual path written (may differ from `path` if .parquet -> .csv)
    """
    if _PARQUET_ENGINE is not None:
        actual_path = path if path.endswith(".parquet") else path + ".parquet"
        df.to_parquet(actual_path, index=False, engine=_PARQUET_ENGINE)
    else:
        actual_path = path.replace(".parquet", ".csv")
        df.to_csv(actual_path, index=False)
    return actual_path


def load_features(path: str) -> pd.DataFrame:
    """
    Load a feature DataFrame, trying Parquet first then CSV fallback.

    Looks for `path` as-is, then swaps the extension if not found.
    """
    # Try exact path first
    if os.path.isfile(path):
        if path.endswith(".parquet") and _PARQUET_ENGINE is not None:
            return pd.read_parquet(path, engine=_PARQUET_ENGINE)
        if path.endswith(".parquet"):
            # parquet file exists but no engine -- try CSV sibling
            csv_path = path.replace(".parquet", ".csv")
            if os.path.isfile(csv_path):
                return pd.read_csv(csv_path)
        if path.endswith(".csv"):
            return pd.read_csv(path)

    # Try the other extension
    if path.endswith(".parquet"):
        alt = path.replace(".parquet", ".csv")
    else:
        alt = path.replace(".csv", ".parquet")

    if os.path.isfile(alt):
        if alt.endswith(".parquet") and _PARQUET_ENGINE is not None:
            return pd.read_parquet(alt, engine=_PARQUET_ENGINE)
        if alt.endswith(".csv"):
            return pd.read_csv(alt)

    raise FileNotFoundError(
        f"Feature file not found: {path}\n"
        "  Run:  python feature_extraction.py  first."
    )


# ==============================================================================
#  UTILITY: PEAK DETECTION
# ==============================================================================

def _find_ppg_peaks(
    window: np.ndarray,
    fs: float,
    min_distance_s: float = 0.4,
) -> np.ndarray:
    """
    Detect systolic (positive) peaks in a 1-D PPG window.

    Uses scipy.signal.find_peaks with a minimum inter-peak distance of
    min_distance_s seconds (default 0.4 s = 150 bpm upper bound).

    Parameters
    ----------
    window          : 1-D float array (length L)
    fs              : sampling frequency (Hz)
    min_distance_s  : minimum distance between peaks in seconds

    Returns
    -------
    peaks : np.ndarray of int indices
    """
    min_dist_samples = int(min_distance_s * fs)
    peaks, _ = sp_signal.find_peaks(
        window,
        distance=min_dist_samples,
        prominence=0.05 * (window.max() - window.min() + 1e-9),
    )
    return peaks


# ==============================================================================
#  1. TIME-DOMAIN & STATISTICAL FEATURES
# ==============================================================================

def time_domain_features(window: np.ndarray) -> dict[str, float]:
    """
    Compute basic statistical descriptors of the raw PPG window.

    Features
    --------
    mean        : signal mean (baseline proxy)
    variance    : signal variance (amplitude spread)
    std         : standard deviation
    skewness    : third moment (waveform asymmetry)
    kurt        : excess kurtosis (peakedness)
    zcr         : zero-crossing rate (frequency proxy)
    p2p         : peak-to-trough amplitude (global)
    rms         : root-mean-square energy
    """
    w = window.astype(np.float64)
    mean_val = float(np.mean(w))
    centered = w - mean_val

    # Zero-crossing rate: fraction of samples where sign changes
    signs = np.sign(centered)
    zcr   = float(np.sum(np.diff(signs) != 0)) / max(len(w) - 1, 1)

    return {
        "td_mean":     mean_val,
        "td_variance": float(np.var(w)),
        "td_std":      float(np.std(w)),
        "td_skewness": float(skew(w, bias=False)),
        "td_kurtosis": float(kurtosis(w, bias=False)),
        "td_zcr":      zcr,
        "td_p2p":      float(np.ptp(w)),
        "td_rms":      float(np.sqrt(np.mean(w ** 2))),
    }


# ==============================================================================
#  2. PPG MORPHOLOGICAL FEATURES
# ==============================================================================

def morphological_features(window: np.ndarray, fs: float) -> dict[str, float]:
    """
    Extract beat-morphology features from detected systolic peaks.

    Features
    --------
    morph_num_peaks          : number of detected systolic peaks
    morph_mean_amplitude     : average peak amplitude
    morph_std_amplitude      : std of peak amplitudes (amplitude variability)
    morph_mean_ppi           : mean peak-to-peak interval (PPI) in seconds
    morph_std_ppi            : std of PPI (equivalent to HRV in time domain)
    morph_mean_hr_bpm        : mean heart rate derived from PPI
    morph_mean_pulse_width   : mean pulse width at 50% of peak height
    morph_dicrotic_ratio     : mean dicrotic notch ratio (valley / peak amplitude)
    """
    peaks = _find_ppg_peaks(window, fs)

    if len(peaks) < 2:
        # Insufficient peaks — fill with sentinel zero values
        return {
            "morph_num_peaks":        float(len(peaks)),
            "morph_mean_amplitude":   float(np.max(window)) if len(peaks) == 1 else 0.0,
            "morph_std_amplitude":    0.0,
            "morph_mean_ppi":         0.0,
            "morph_std_ppi":          0.0,
            "morph_mean_hr_bpm":      0.0,
            "morph_mean_pulse_width": 0.0,
            "morph_dicrotic_ratio":   0.0,
        }

    amplitudes   = window[peaks]
    ppis_samples = np.diff(peaks)           # peak-to-peak intervals (samples)
    ppis_seconds = ppis_samples / fs        # convert to seconds

    feats: dict[str, float] = {
        "morph_num_peaks":    float(len(peaks)),
        "morph_mean_amplitude": float(np.mean(amplitudes)),
        "morph_std_amplitude":  float(np.std(amplitudes)),
        "morph_mean_ppi":     float(np.mean(ppis_seconds)),
        "morph_std_ppi":      float(np.std(ppis_seconds)),
        "morph_mean_hr_bpm":  float(60.0 / np.mean(ppis_seconds))
                               if np.mean(ppis_seconds) > 1e-9 else 0.0,
    }

    # -- Pulse width at 50% height ---------------------------------------------
    pulse_widths: list[float] = []
    for i, pk in enumerate(peaks[:-1]):
        beat_end = peaks[i + 1]
        beat_seg = window[pk:beat_end]
        if len(beat_seg) < 3:
            continue
        pk_val   = float(beat_seg[0])
        trough   = float(np.min(beat_seg))
        half_amp = (pk_val + trough) / 2.0
        above    = np.where(beat_seg >= half_amp)[0]
        if len(above) >= 2:
            pulse_widths.append((above[-1] - above[0]) / fs)
    feats["morph_mean_pulse_width"] = float(np.mean(pulse_widths)) if pulse_widths else 0.0

    # -- Dicrotic notch ratio --------------------------------------------------
    # A dicrotic notch appears as a secondary local minimum ~60% into the
    # descending limb of each beat.  Estimated as the valley between 60-90% of
    # the beat period, compared to the systolic peak.
    notch_ratios: list[float] = []
    for i, pk in enumerate(peaks[:-1]):
        beat_end  = peaks[i + 1]
        beat_seg  = window[pk:beat_end]
        L = len(beat_seg)
        if L < 6:
            continue
        ns = int(0.60 * L)
        ne = int(0.90 * L)
        if ns >= ne:
            continue
        notch_val = float(np.min(beat_seg[ns:ne]))
        pk_val    = float(beat_seg[0])
        if abs(pk_val) > 1e-9:
            notch_ratios.append(notch_val / pk_val)
    feats["morph_dicrotic_ratio"] = float(np.mean(notch_ratios)) if notch_ratios else 0.0

    return feats


# ==============================================================================
#  3. HRV FEATURES
# ==============================================================================

def hrv_features(window: np.ndarray, fs: float) -> dict[str, float]:
    """
    Compute standard HRV metrics from RR (peak-to-peak) intervals.

    Time-domain HRV
    ---------------
    hrv_sdnn    : SDNN -- standard deviation of all RR intervals
    hrv_rmssd   : RMSSD -- root mean square of successive differences
    hrv_pnn50   : pNN50 -- proportion of RR diffs > 50 ms

    Frequency-domain HRV (via Lomb-Scargle periodogram on RR series)
    ----------------
    hrv_lf_power  : power in Low-Frequency band  (0.04 - 0.15 Hz)
    hrv_hf_power  : power in High-Frequency band (0.15 - 0.40 Hz)
    hrv_lf_hf     : LF/HF ratio (sympatho-vagal balance proxy)
    """
    peaks = _find_ppg_peaks(window, fs)

    empty = {
        "hrv_sdnn":     0.0,
        "hrv_rmssd":    0.0,
        "hrv_pnn50":    0.0,
        "hrv_lf_power": 0.0,
        "hrv_hf_power": 0.0,
        "hrv_lf_hf":    0.0,
    }

    if len(peaks) < 3:
        return empty

    rr_ms = np.diff(peaks) / fs * 1000.0     # RR intervals in ms

    # -- Time-domain HRV -------------------------------------------------------
    sdnn  = float(np.std(rr_ms, ddof=1)) if len(rr_ms) > 1 else 0.0
    diffs = np.diff(rr_ms)
    rmssd = float(np.sqrt(np.mean(diffs ** 2))) if len(diffs) > 0 else 0.0
    pnn50 = float(np.sum(np.abs(diffs) > 50.0) / len(diffs)) \
             if len(diffs) > 0 else 0.0

    # -- Frequency-domain HRV (Lomb-Scargle on irregular RR series) -----------
    peak_times_s = peaks[:-1] / fs
    rr_s         = rr_ms / 1000.0

    lf_power = hf_power = lf_hf = 0.0
    if len(rr_s) >= 4:
        try:
            freqs    = np.linspace(0.01, 0.5, 512)
            pgram    = sp_signal.lombscargle(
                peak_times_s, rr_s - rr_s.mean(), freqs, normalize=True
            )
            df_hz    = freqs[1] - freqs[0]
            lf_mask  = (freqs >= 0.04) & (freqs <= 0.15)
            hf_mask  = (freqs >= 0.15) & (freqs <= 0.40)
            lf_power = float(np.sum(pgram[lf_mask]) * df_hz)
            hf_power = float(np.sum(pgram[hf_mask]) * df_hz)
            lf_hf    = lf_power / hf_power if hf_power > 1e-12 else 0.0
        except Exception:
            pass   # Lombscargle can fail for degenerate inputs

    return {
        "hrv_sdnn":     sdnn,
        "hrv_rmssd":    rmssd,
        "hrv_pnn50":    pnn50,
        "hrv_lf_power": lf_power,
        "hrv_hf_power": hf_power,
        "hrv_lf_hf":    lf_hf,
    }


# ==============================================================================
#  4. FREQUENCY-DOMAIN FEATURES (whole-window FFT)
# ==============================================================================

def frequency_features(window: np.ndarray, fs: float) -> dict[str, float]:
    """
    Spectral features extracted from the FFT magnitude spectrum.

    Features
    --------
    freq_dominant_hz       : dominant (peak) frequency in Hz
    freq_spectral_centroid : power-weighted mean frequency
    freq_band_0_1          : power fraction in 0-1 Hz (respiration / very-slow)
    freq_band_1_4          : power fraction in 1-4 Hz (cardiac fundamental + harmonics)
    freq_band_4_8          : power fraction in 4-8 Hz (higher harmonics / noise)
    freq_spectral_entropy  : Shannon entropy of normalised power spectrum
    """
    w    = window.astype(np.float64)
    N    = len(w)
    hann = np.hanning(N)
    w_win = w * hann

    fft_mag = np.abs(np.fft.rfft(w_win))
    freqs   = np.fft.rfftfreq(N, d=1.0 / fs)

    mask  = freqs <= 16.0
    f_sub = freqs[mask]
    m_sub = fft_mag[mask]

    power   = m_sub ** 2
    total_p = power.sum() + 1e-12

    dom_idx  = int(np.argmax(m_sub))
    dom_hz   = float(f_sub[dom_idx])
    centroid = float(np.sum(f_sub * power) / total_p)

    def _band_power(flo: float, fhi: float) -> float:
        bm = (f_sub >= flo) & (f_sub < fhi)
        return float(power[bm].sum() / total_p)

    pn  = power / total_p
    pn  = pn[pn > 0]
    ent = float(-np.sum(pn * np.log2(pn + 1e-12)))

    return {
        "freq_dominant_hz":       dom_hz,
        "freq_spectral_centroid": centroid,
        "freq_band_0_1":          _band_power(0.0, 1.0),
        "freq_band_1_4":          _band_power(1.0, 4.0),
        "freq_band_4_8":          _band_power(4.0, 8.0),
        "freq_spectral_entropy":  ent,
    }


# ==============================================================================
#  COMBINED EXTRACTION
# ==============================================================================

def extract_window_features(window: np.ndarray, fs: float) -> dict[str, float]:
    """
    Extract all features for a single PPG window (26 features total).

    Parameters
    ----------
    window : 1-D float array of shape (L,) or (L, 1)
    fs     : sampling frequency in Hz

    Returns
    -------
    dict[str, float]  --  flat feature dictionary with 26 keys
    """
    w = window.ravel().astype(np.float32)

    feats: dict[str, float] = {}
    feats.update(time_domain_features(w))          #  8 features
    feats.update(morphological_features(w, fs))    #  8 features
    feats.update(hrv_features(w, fs))              #  6 features
    feats.update(frequency_features(w, fs))        #  6 features

    return feats                                   # 28 total


def extract_features_batch(
    X: np.ndarray,
    fs: float,
    desc: str = "Extracting features",
) -> pd.DataFrame:
    """
    Extract features for every window in X.

    Parameters
    ----------
    X    : float array of shape (N, L) or (N, L, 1)
    fs   : sampling frequency
    desc : tqdm progress bar label

    Returns
    -------
    pd.DataFrame  with N rows and one column per feature
    """
    rows = []
    for i in tqdm(range(len(X)), desc=desc, unit="win"):
        rows.append(extract_window_features(X[i], fs))

    df = pd.DataFrame(rows, dtype=np.float32)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0.0, inplace=True)
    return df


# ==============================================================================
#  MAIN
# ==============================================================================

def main(
    dataset_path: str = DATASET_PATH,
    fs: float = FS_DEFAULT,
    output_dir: str = OUTPUT_DIR,
) -> None:
    """
    Load dataset.npz, extract features, save parquet files.

    Output files
    ------------
    outputs/features_train.parquet
    outputs/features_val.parquet
    outputs/features_test.parquet
    """
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}\n"
            "Run:  python preprocessing/fuse_datasets.py  first."
        )
    logger.info("Loading dataset from %s ...", dataset_path)
    data = np.load(dataset_path, allow_pickle=True)

    splits = {
        "train": (data["X_train"], data["y_train"]),
        "val":   (data["X_val"],   data["y_val"]),
        "test":  (data["X_test"],  data["y_test"]),
    }
    for split_name, (X, y) in splits.items():
        logger.info("  %s: X=%s  y=%s", split_name, X.shape, y.shape)

    saved_paths: dict[str, str] = {}
    for split_name, (X, y) in splits.items():
        logger.info("Extracting %s features (%d windows) ...", split_name, len(X))
        df = extract_features_batch(X, fs=fs, desc=split_name)
        df["label"] = y.astype(np.int32)

        desired_path = os.path.join(output_dir, f"features_{split_name}.parquet")
        actual_path  = save_features(df, desired_path)
        saved_paths[split_name] = actual_path
        logger.info(
            "  Saved -> %s  (%d rows x %d feature columns)",
            actual_path, len(df), df.shape[1] - 1,
        )

    # -- Feature summary -------------------------------------------------------
    train_df = load_features(saved_paths["train"])
    logger.info("\n%s\nFeature Summary\n%s", "=" * 50, "=" * 50)
    logger.info("Total features : %d", train_df.shape[1] - 1)
    fmt = "  (parquet via %s)" % _PARQUET_ENGINE if _PARQUET_ENGINE else "  (csv fallback)"
    logger.info("Storage format : %s", fmt)
    for grp, prefix in [
        ("Time-domain / Statistical", "td_"),
        ("PPG Morphological",         "morph_"),
        ("HRV",                       "hrv_"),
        ("Frequency-domain",          "freq_"),
    ]:
        cols = [c for c in train_df.columns if c.startswith(prefix)]
        logger.info("  %-32s %d features", grp, len(cols))

    logger.info("\nFeature extraction complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract PPG window features for tree-based models"
    )
    parser.add_argument(
        "--input", default=DATASET_PATH,
        help="Path to dataset.npz (default: outputs/dataset.npz)",
    )
    parser.add_argument(
        "--fs", type=float, default=FS_DEFAULT,
        help="Sampling frequency in Hz (default: 125)",
    )
    parser.add_argument(
        "--output-dir", default=OUTPUT_DIR,
        help="Directory to save parquet files (default: outputs/)",
    )
    args = parser.parse_args()
    main(dataset_path=args.input, fs=args.fs, output_dir=args.output_dir)
