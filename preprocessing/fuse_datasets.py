"""
fuse_datasets.py
================
Master preprocessing orchestrator that:

  1. Loads MIMIC-IV PPG segments + labels
  2. Loads PTT PPG + accelerometer data
  3. Applies motion artifact removal to PTT
  4. Runs the full signal-processing pipeline on both datasets
  5. Fuses them into a unified dataset
  6. Applies SMOTE to the TRAINING split for class balancing
  7. Saves train/val/test arrays to  outputs/dataset.npz

Run this script once before training:
  python preprocessing/fuse_datasets.py
"""

import os
import sys
import logging
import argparse
import numpy as np
from sklearn.model_selection import train_test_split

# ── make project root importable ──────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from preprocessing.load_mimic        import load_mimic
from preprocessing.load_ptt          import load_ptt
from preprocessing.motion_artifact   import batch_remove_motion_artifacts
from preprocessing.signal_processing import batch_process, TARGET_FS
from preprocessing.label_mapping     import print_class_distribution, NUM_CLASSES

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── GPU detection (informational) ──────────────────────────────────────────────
try:
    import cupy as cp
    logger.info("fuse_datasets: GPU DETECTED – preprocessing will use GPU (device %d, %.0f MB free)",
                cp.cuda.Device().id,
                cp.cuda.Device().mem_info[0] / 1e6)
except ImportError:
    logger.info("fuse_datasets: CuPy not installed – preprocessing will run on CPU")

# ── output directory ──────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "dataset.npz")

# ── MIMIC source Fs (Pleth is ~62.5 Hz in MIMIC-IV waveform db) ──────────────
MIMIC_FS = 62.5


def apply_smote(X_train: np.ndarray, y_train: np.ndarray):
    """
    Apply SMOTE to training data to balance class distribution.
    Falls back to random oversampling if SMOTE fails (e.g. too few samples).
    """
    try:
        from imblearn.over_sampling import SMOTE
        n_samples, window_len, n_ch = X_train.shape
        X_2d = X_train.reshape(n_samples, -1)

        # Check minimum samples per class
        unique, counts = np.unique(y_train, return_counts=True)
        min_count = counts.min()
        k_neighbors = min(5, min_count - 1)
        if k_neighbors < 1:
            raise ValueError(f"Too few samples in smallest class ({min_count}) for SMOTE.")

        sm = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_res, y_res = sm.fit_resample(X_2d, y_train)
        X_res = X_res.reshape(-1, window_len, n_ch).astype(np.float32)
        logger.info("SMOTE: %d → %d training samples", n_samples, len(X_res))
        return X_res, y_res.astype(np.int32)
    except Exception as e:
        logger.warning("SMOTE failed (%s) – trying RandomOverSampler", e)
        try:
            from imblearn.over_sampling import RandomOverSampler
            n_samples, window_len, n_ch = X_train.shape
            X_2d = X_train.reshape(n_samples, -1)
            ros = RandomOverSampler(random_state=42)
            X_res, y_res = ros.fit_resample(X_2d, y_train)
            X_res = X_res.reshape(-1, window_len, n_ch).astype(np.float32)
            logger.info("RandomOverSampler: %d → %d training samples", n_samples, len(X_res))
            return X_res, y_res.astype(np.int32)
        except Exception as e2:
            logger.warning("RandomOverSampler also failed (%s) – proceeding without oversampling", e2)
            return X_train, y_train


def fuse_and_save(
    max_mimic_patients: int = None,
    max_ptt_files: int = None,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    apply_oversampling: bool = True,
    seed: int = 42,
):
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    max_mimic_patients : limit MIMIC patients (None = all)
    max_ptt_files      : limit PTT CSV files  (None = all)
    val_ratio          : validation split fraction
    test_ratio         : test split fraction
    apply_oversampling : use SMOTE on training split
    seed               : random seed
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 1. Load MIMIC-IV ──────────────────────────────────────────────────────
    logger.info("=" * 55)
    logger.info("STEP 1 – Loading MIMIC-IV dataset …")
    mimic_segs, mimic_labels, mimic_fs = load_mimic(max_patients=max_mimic_patients)

    # ── 2. Load PTT PPG ───────────────────────────────────────────────────────
    logger.info("=" * 55)
    logger.info("STEP 2 – Loading PTT PPG dataset …")
    ptt_ppg, ptt_acc, ptt_labels, ptt_fs = load_ptt(max_files=max_ptt_files)

    # ── 3. Motion artifact removal on PTT ────────────────────────────────────
    logger.info("=" * 55)
    logger.info("STEP 3 – Motion artifact removal on PTT data …")
    ptt_ppg_clean = batch_remove_motion_artifacts(ptt_ppg, ptt_acc)

    # ── 4. Signal processing – MIMIC ─────────────────────────────────────────
    logger.info("=" * 55)
    logger.info("STEP 4a – Signal processing: MIMIC-IV …")
    X_mimic, y_mimic = batch_process(mimic_segs, mimic_labels, src_fs=MIMIC_FS)

    # ── 5. Signal processing – PTT ───────────────────────────────────────────
    logger.info("STEP 4b – Signal processing: PTT …")
    X_ptt, y_ptt = batch_process(ptt_ppg_clean, ptt_labels, src_fs=ptt_fs)

    # ── 6. Fuse ───────────────────────────────────────────────────────────────
    logger.info("=" * 55)
    logger.info("STEP 5 – Fusing datasets …")

    if X_mimic.shape[0] == 0 and X_ptt.shape[0] == 0:
        logger.error("No windows were produced from either dataset. Aborting.")
        sys.exit(1)
    elif X_mimic.shape[0] == 0:
        X_all, y_all = X_ptt, y_ptt
        logger.warning("MIMIC dataset produced 0 windows – using PTT only.")
    elif X_ptt.shape[0] == 0:
        X_all, y_all = X_mimic, y_mimic
        logger.warning("PTT dataset produced 0 windows – using MIMIC only.")
    else:
        X_all = np.concatenate([X_mimic, X_ptt], axis=0)
        y_all = np.concatenate([y_mimic, y_ptt], axis=0)

    logger.info("Total windows: %d  |  Shape: %s", len(X_all), X_all.shape)
    print_class_distribution(y_all, "Full Dataset Distribution")

    # ── 7. Train / Val / Test split ───────────────────────────────────────────
    logger.info("STEP 6 – Train / Val / Test split …")
    train_ratio = 1.0 - val_ratio - test_ratio

    # First split: train vs (val+test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_all, y_all,
        test_size=(val_ratio + test_ratio),
        stratify=y_all,
        random_state=seed,
    )
    # Second split: val vs test
    relative_test = test_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=relative_test,
        stratify=y_temp,
        random_state=seed,
    )

    logger.info("Split sizes – Train: %d | Val: %d | Test: %d",
                len(X_train), len(X_val), len(X_test))

    # ── 8. SMOTE on training split ────────────────────────────────────────────
    if apply_oversampling:
        logger.info("STEP 7 – Applying SMOTE on training split …")
        X_train, y_train = apply_smote(X_train, y_train)
        print_class_distribution(y_train, "Training Set (after SMOTE)")

    # ── 9. Save ───────────────────────────────────────────────────────────────
    logger.info("STEP 8 – Saving to %s …", OUTPUT_PATH)
    np.savez_compressed(
        OUTPUT_PATH,
        X_train=X_train, y_train=y_train,
        X_val=X_val,     y_val=y_val,
        X_test=X_test,   y_test=y_test,
        fs=np.array([TARGET_FS]),
        class_names=np.array(["Normal", "Atrial Fibrillation",
                               "Bradycardia", "Tachycardia", "Other Arrhythmia"]),
    )
    logger.info("Dataset saved: %s (%.1f MB)",
                OUTPUT_PATH, os.path.getsize(OUTPUT_PATH) / 1e6)
    logger.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fuse and preprocess PPG datasets")
    parser.add_argument("--max-mimic-patients", type=int, default=None,
                        help="Limit number of MIMIC patient dirs (None=all)")
    parser.add_argument("--max-ptt-files",      type=int, default=None,
                        help="Limit number of PTT CSV files (None=all)")
    parser.add_argument("--no-smote", action="store_true",
                        help="Skip SMOTE oversampling")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    fuse_and_save(
        max_mimic_patients=args.max_mimic_patients,
        max_ptt_files=args.max_ptt_files,
        apply_oversampling=not args.no_smote,
        seed=args.seed,
    )
