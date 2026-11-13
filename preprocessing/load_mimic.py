"""
load_mimic.py
=============
Loads PPG waveform data and arrhythmia labels from the MIMIC-IV Waveform Database.

Dataset structure (WFDB multi-segment layout):
  waves/pXXX/pXXXXXXXX/<record_id>/<record_id>_NNNN.hea   -- segment headers
  waves/pXXX/pXXXXXXXX/<record_id>/<record_id>n.csv.gz    -- numeric parameters

Label derivation from MIMIC numerics (HR [bpm] and PVC [/min] columns):
  0 - Normal Sinus Rhythm   (60 <= HR <= 100, PVC <= 10)
  1 - Atrial Fibrillation   (detected via HR irregularity proxy - HR > 100 AND PVC > 5)
  2 - Bradycardia            (HR < 60)
  3 - Tachycardia            (HR > 100, no high PVC)
  4 - Other Arrhythmia       (PVC > 10, normal HR range)
"""

import os
import re
import gzip
import glob
import logging
import numpy as np
import pandas as pd
import wfdb

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.join(
    os.path.dirname(__file__), "..",
    "MIMIC_2", "physionet.org", "files", "mimic4wdb", "0.1.0", "waves"
)

# ── clinical thresholds for label derivation ────────────────────────────────
HR_BRADY_MAX  = 60    # bpm
HR_TACHY_MIN  = 100   # bpm
PVC_HIGH_MIN  = 10    # /min  → Other Arrhythmia
PVC_AF_MIN    = 5     # /min  combined with tachy → proxy for AF


def _load_numerics(numerics_gz_path: str) -> pd.DataFrame:
    """Load a gzip-compressed CSV of MIMIC numeric parameters."""
    try:
        with gzip.open(numerics_gz_path, "rt") as f:
            df = pd.read_csv(f, on_bad_lines="skip")
        return df
    except Exception as e:
        logger.debug("Could not read numerics %s: %s", numerics_gz_path, e)
        return pd.DataFrame()


def _get_rhythm_label(record_dir: str) -> int:
    """
    Derive an arrhythmia label from clinical HR and PVC parameters.

    Priority order:
      1. Bradycardia  (median HR < 60)
      2. AF proxy     (median HR > 100 AND median PVC > PVC_AF_MIN)
      3. Tachycardia  (median HR > 100)
      4. Other        (PVC > 10, normal HR)
      5. Normal       (default)
    """
    gz_files = glob.glob(os.path.join(record_dir, "*n.csv.gz"))
    if not gz_files:
        return 0

    df = _load_numerics(gz_files[0])
    if df.empty:
        return 0

    # Find HR column (may be named 'HR [bpm]' or 'Heart Rate')
    hr_col  = next((c for c in df.columns if "HR" in c and "bpm" in c), None)
    pvc_col = next((c for c in df.columns if "PVC" in c), None)

    med_hr  = df[hr_col].median()  if hr_col  and not df[hr_col].dropna().empty  else None
    med_pvc = df[pvc_col].median() if pvc_col and not df[pvc_col].dropna().empty else None

    if med_hr is not None and med_hr < HR_BRADY_MAX:
        return 2  # Bradycardia
    if med_hr is not None and med_hr > HR_TACHY_MIN:
        if med_pvc is not None and med_pvc > PVC_AF_MIN:
            return 1  # AF proxy (tachy + frequent ectopic beats)
        return 3  # Tachycardia
    if med_pvc is not None and med_pvc > PVC_HIGH_MIN:
        return 4  # Other Arrhythmia
    return 0  # Normal


def _load_segment(hea_path: str) -> np.ndarray | None:
    """
    Load a single WFDB segment, extract the Pleth channel.
    Returns a 1-D float32 numpy array, or None on failure.
    """
    record_name = hea_path[:-4]  # strip .hea
    try:
        record = wfdb.rdrecord(record_name)
    except Exception as e:
        logger.debug("Failed to read %s: %s", record_name, e)
        return None

    sig_names = [s.lower() for s in record.sig_name]
    pleth_indices = [i for i, n in enumerate(sig_names) if "pleth" in n]

    if not pleth_indices:
        return None

    ppg = record.p_signal[:, pleth_indices[0]].astype(np.float32)
    # Drop NaN samples
    ppg = ppg[~np.isnan(ppg)]
    if len(ppg) < 100:
        return None

    return ppg


def load_mimic(max_patients: int = None, verbose: bool = True):
    """
    Walk the waves directory and load all PPG segments with rhythm labels.

    Parameters
    ----------
    max_patients : int or None
        If set, limits the number of patient directories processed.
    verbose : bool
        Print progress.

    Returns
    -------
    segments : list of np.ndarray  – raw PPG arrays (variable length)
    labels   : list of int         – integer class per segment
    fs       : float               – nominal sampling frequency (Hz)
    """
    base = os.path.abspath(BASE_DIR)
    if not os.path.isdir(base):
        raise FileNotFoundError(f"MIMIC waves directory not found: {base}")

    segments: list[np.ndarray] = []
    labels:   list[int]        = []
    fs = 62.5   # MIMIC-IV Pleth native Fs (~62.5 Hz); resampled to 125 Hz in signal_processing

    # ── iterate patient folders ────────────────────────────────────────────────
    patient_dirs = sorted([
        d for d in os.listdir(base)
        if os.path.isdir(os.path.join(base, d)) and d.startswith("p")
    ])
    if max_patients:
        patient_dirs = patient_dirs[:max_patients]

    for p_group in patient_dirs:
        group_path = os.path.join(base, p_group)
        for patient in sorted(os.listdir(group_path)):
            patient_path = os.path.join(group_path, patient)
            if not os.path.isdir(patient_path):
                continue
            for record_id in sorted(os.listdir(patient_path)):
                record_path = os.path.join(patient_path, record_id)
                if not os.path.isdir(record_path):
                    continue

                # Get dominant rhythm label for this recording
                label = _get_rhythm_label(record_path)

                # Find all segment .hea files (exclude master header)
                seg_pattern = os.path.join(record_path, f"{record_id}_????.hea")
                seg_headers = sorted(glob.glob(seg_pattern))

                for hea in seg_headers:
                    ppg = _load_segment(hea)
                    if ppg is not None:
                        segments.append(ppg)
                        labels.append(label)

    if verbose:
        total = len(segments)
        unique, counts = np.unique(labels, return_counts=True)
        logger.info("MIMIC-IV: %d segments loaded", total)
        for cls, cnt in zip(unique, counts):
            class_names = ["Normal", "AF", "Bradycardia", "Tachycardia", "Other"]
            name = class_names[cls] if cls < len(class_names) else f"Class {cls}"
            logger.info("  Class %d (%s): %d segments", cls, name, cnt)

    return segments, labels, fs


if __name__ == "__main__":
    segs, labs, fs = load_mimic(max_patients=5)
    logger.info("Sample segment shape: %s, label: %d", segs[0].shape if segs else "N/A", labs[0] if labs else -1)
