"""
motion_artifact.py
==================
Motion Artifact (MA) removal for PPG signals using an LMS adaptive filter.

The accelerometer magnitude is used as the reference noise signal.
The LMS filter learns to predict and cancel the motion-induced component
from the corrupted PPG signal, leaving a cleaner cardiac signal.

Reference:
  Widrow et al., "Adaptive Noise Cancelling: Principles and Applications",
  Proceedings of the IEEE, 1975.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


def acc_magnitude_reference(acc: np.ndarray) -> np.ndarray:
    """
    Compute zero-mean ACC magnitude for use as LMS reference signal.

    Parameters
    ----------
    acc : np.ndarray, shape (N, 3)
        Accelerometer data [a_x, a_y, a_z].

    Returns
    -------
    ref : np.ndarray, shape (N,)
        Zero-mean magnitude of acceleration.
    """
    mag = np.sqrt(np.sum(acc ** 2, axis=1)).astype(np.float32)
    mag -= mag.mean()  # remove DC / gravity component
    return mag


def lms_adaptive_filter(
    ppg_noisy: np.ndarray,
    reference: np.ndarray,
    order: int = 32,
    mu: float = 0.005,
) -> np.ndarray:
    """
    LMS Adaptive Noise Canceller.

    Estimates the noise component correlated with `reference` and subtracts
    it from `ppg_noisy`. The output is the estimated clean PPG.

    Parameters
    ----------
    ppg_noisy : np.ndarray, shape (N,)
        Raw PPG corrupted by motion artifacts.
    reference : np.ndarray, shape (N,)
        Reference signal (zero-mean ACC magnitude).
    order : int
        Number of filter taps (FIR length).
    mu : float
        LMS step size. Smaller = more stable, slower convergence.

    Returns
    -------
    ppg_clean : np.ndarray, shape (N,)
        Motion-artifact-reduced PPG signal.
    """
    N = len(ppg_noisy)
    if len(reference) != N:
        raise ValueError("ppg_noisy and reference must have the same length.")

    # Normalize reference to unit variance for numerical stability
    ref_std = reference.std()
    if ref_std < 1e-8:
        return ppg_noisy.copy()
    ref_norm = reference / ref_std
    # Clip to prevent massive outliers from destabilizing the filter
    ref_norm = np.clip(ref_norm, -20.0, 20.0)

    # Normalize PPG to zero-mean, unit-variance to prevent overflow
    # with large-amplitude raw ADC signals (e.g. PTT pleth_2 ~ 69000 counts)
    ppg_mean = float(ppg_noisy.mean())
    ppg_std  = float(ppg_noisy.std())
    if ppg_std < 1e-8:
        return ppg_noisy.copy()
    ppg_norm = (ppg_noisy.astype(np.float64) - ppg_mean) / ppg_std
    ppg_norm = np.clip(ppg_norm, -20.0, 20.0)

    # Initialise filter weights
    w = np.zeros(order, dtype=np.float64)
    filtered_norm = np.zeros(N, dtype=np.float64)

    for n in range(order, N):
        x = ref_norm[n - order:n][::-1]
        noise_est = np.dot(w, x)
        e = ppg_norm[n] - noise_est
        
        # NLMS update to prevent divergence. 
        # Multiply mu by order so the effective step size matches standard LMS
        # on a unit-variance reference signal.
        x_energy = np.dot(x, x) + 1e-8
        step_size = (2.0 * mu * order) / x_energy
        
        w += step_size * e * x
        w = np.clip(w, -50.0, 50.0)  # Safeguard against exploding weights
        
        filtered_norm[n] = e

    # Fill warmup region
    filtered_norm[:order] = ppg_norm[:order]

    # Restore original scale
    ppg_clean = (filtered_norm * ppg_std + ppg_mean).astype(np.float32)

    return ppg_clean


def remove_motion_artifacts(
    ppg_noisy: np.ndarray,
    acc: np.ndarray,
    order: int = 32,
    mu: float = 0.005,
) -> np.ndarray:
    """
    End-to-end motion artifact removal for a single PPG recording.

    Parameters
    ----------
    ppg_noisy : np.ndarray, shape (N,)
        Raw PPG signal (already downsampled to target Fs).
    acc : np.ndarray, shape (N, 3)
        Accelerometer data [a_x, a_y, a_z] at the same Fs.
    order : int
        LMS filter order (taps).
    mu : float
        LMS step size.

    Returns
    -------
    ppg_clean : np.ndarray, shape (N,)
        Motion-artifact-cancelled PPG.
    """
    reference = acc_magnitude_reference(acc)
    ppg_clean = lms_adaptive_filter(ppg_noisy, reference, order=order, mu=mu)
    return ppg_clean


# ── Batch processing ───────────────────────────────────────────────────────────

def batch_remove_motion_artifacts(
    ppg_list: list,
    acc_list: list,
    order: int = 32,
    mu: float = 0.005,
) -> list:
    """
    Apply MA removal to a list of (ppg, acc) pairs.

    Parameters
    ----------
    ppg_list : list of np.ndarray  – raw PPG arrays
    acc_list : list of np.ndarray  – ACC arrays (same length as PPG)

    Returns
    -------
    clean_ppg_list : list of np.ndarray
    """
    clean = []
    for i, (ppg, acc) in enumerate(zip(ppg_list, acc_list)):
        try:
            ppg_clean = remove_motion_artifacts(ppg, acc, order=order, mu=mu)
            clean.append(ppg_clean)
        except Exception as e:
            logger.warning("MA removal failed for recording %d: %s – using raw", i, e)
            clean.append(ppg)
    logger.info("Motion artifact removal complete: %d recordings processed", len(clean))
    return clean


if __name__ == "__main__":
    # Smoke test with synthetic signals
    fs = 125
    t = np.linspace(0, 10, 10 * fs)
    ppg_clean_gt = np.sin(2 * np.pi * 1.2 * t).astype(np.float32)  # 1.2 Hz heart rate
    motion = np.sin(2 * np.pi * 3.0 * t).astype(np.float32) * 0.5  # 3 Hz motion
    ppg_noisy = ppg_clean_gt + motion

    acc = np.column_stack([motion, motion * 0.8, motion * 0.6]).astype(np.float32)

    ppg_out = remove_motion_artifacts(ppg_noisy, acc)
    corr = np.corrcoef(ppg_clean_gt[32:], ppg_out[32:])[0, 1]
    print(f"Correlation with ground truth clean PPG: {corr:.4f}")
