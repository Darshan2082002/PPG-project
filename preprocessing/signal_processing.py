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

GPU Acceleration
----------------
When CuPy is installed the resampling fallback, segmentation, and
normalisation steps run on the GPU.  Bandpass filtering uses scipy on the
CPU (no GPU equivalent for `filtfilt`) but the array is moved to GPU
immediately afterwards.  Falls back transparently to NumPy if CuPy is
unavailable.
"""

import numpy as np
from scipy import signal as sp_signal
import logging

logger = logging.getLogger(__name__)

# ── GPU / CuPy initialisation ─────────────────────────────────────────────────

try:
    import cupy as cp
    _GPU_AVAILABLE = True
    logger.info("signal_processing: CuPy detected – GPU acceleration ENABLED (device %d)",
                cp.cuda.Device().id)
except ImportError:
    cp = None
    _GPU_AVAILABLE = False
    logger.info("signal_processing: CuPy not found – running on CPU")


def _to_gpu(arr: np.ndarray):
    """Move a numpy array to GPU (no-op if CuPy unavailable)."""
    if _GPU_AVAILABLE:
        return cp.asarray(arr)
    return arr


def _to_cpu(arr) -> np.ndarray:
    """Move an array to CPU numpy (no-op if already numpy)."""
    if _GPU_AVAILABLE and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)

TARGET_FS   = 125      # Hz  – model input sampling rate
WINDOW_SEC  = 10       # seconds per segment
OVERLAP     = 0.5      # 50 % overlap between consecutive windows
BP_LOW      = 0.5      # bandpass low cut-off  (Hz)
BP_HIGH     = 8.0      # bandpass high cut-off (Hz)
BP_ORDER    = 4        # Butterworth filter order
MIN_WINDOW_SAMPLES = int(TARGET_FS * WINDOW_SEC)  # 1250 samples

# Max signal length to process per recording (avoids OOM on ICU recordings
# that can be many hours long).  5 minutes → ~59 windows at 50 % overlap.
MAX_SIGNAL_SEC     = 300                        # seconds
MAX_SIGNAL_SAMPLES = int(TARGET_FS * MAX_SIGNAL_SEC)  # 37 500 samples @ 125 Hz

# GPU→CPU transfer chunk size (windows).  Keeps peak allocation ≤ ~6 MB.
_GPU_CHUNK = 512


# ── Bandpass filter ────────────────────────────────────────────────────────────

def bandpass_filter(ppg: np.ndarray, fs: float) -> np.ndarray:
    """
    Apply a 4th-order Butterworth bandpass filter (0.5 – 8 Hz).

    Uses `filtfilt` (zero-phase) when possible.  Falls back to a chunked
    forward SOS pass when the signal is too long for filtfilt to allocate
    its internal buffers (very long ICU recordings).

    Parameters
    ----------
    ppg : np.ndarray, 1-D
    fs  : float  – sampling frequency in Hz

    Returns
    -------
    filtered : np.ndarray, 1-D, same length as ppg
    """
    ppg_cpu = _to_cpu(ppg).astype(np.float32)

    nyq  = fs / 2.0
    low  = max(BP_LOW  / nyq, 1e-4)
    high = min(BP_HIGH / nyq, 0.9999)

    if low >= high:
        logger.warning("Invalid bandpass range for fs=%.1f Hz – returning raw", fs)
        return ppg_cpu

    sos = sp_signal.butter(BP_ORDER, [low, high], btype="bandpass", output="sos")

    # Try zero-phase filtfilt first (best quality)
    try:
        b, a = sp_signal.butter(BP_ORDER, [low, high], btype="bandpass")
        return sp_signal.filtfilt(b, a, ppg_cpu).astype(np.float32)
    except Exception:
        pass  # fall through to SOS forward pass

    # Fallback: forward-only SOS filter (causal but memory-efficient)
    try:
        return sp_signal.sosfilt(sos, ppg_cpu).astype(np.float32)
    except Exception as e:
        logger.warning("sosfilt also failed (%s) – returning raw", e)
        return ppg_cpu


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
    Resample ppg from src_fs to TARGET_FS using polyphase resampling (CPU).
    On failure, falls back to GPU linear interpolation when CuPy is available.
    """
    if abs(src_fs - TARGET_FS) < 0.1:
        return ppg  # already at target rate

    ppg_cpu = _to_cpu(ppg)
    up, down = get_rational_ratio(src_fs, float(TARGET_FS))
    try:
        # scipy polyphase is most accurate – runs on CPU
        resampled = sp_signal.resample_poly(ppg_cpu, up, down).astype(np.float32)
        return resampled
    except Exception as e:
        logger.warning("resample_poly failed (%s) – falling back to GPU linear interpolation", e)

    # GPU-accelerated linear interpolation fallback
    num_samples = int(round(len(ppg_cpu) * TARGET_FS / src_fs))
    if _GPU_AVAILABLE:
        ppg_g = cp.asarray(ppg_cpu)
        x_old = cp.linspace(0.0, 1.0, len(ppg_cpu), dtype=cp.float32)
        x_new = cp.linspace(0.0, 1.0, num_samples,   dtype=cp.float32)
        resampled = cp.interp(x_new, x_old, ppg_g)
        return cp.asnumpy(resampled).astype(np.float32)
    else:
        x_old = np.linspace(0, 1, len(ppg_cpu))
        x_new = np.linspace(0, 1, num_samples)
        return np.interp(x_new, x_old, ppg_cpu).astype(np.float32)


# ── Normalisation ──────────────────────────────────────────────────────────────

def minmax_normalize(window) -> np.ndarray:
    """
    Scale a single window to [0, 1].  Returns zeros if flat (std ≈ 0).
    Works with both numpy and cupy arrays; always returns a numpy array.
    """
    xp = cp if (_GPU_AVAILABLE and isinstance(window, cp.ndarray)) else np
    mn  = window.min()
    mx  = window.max()
    rng = mx - mn
    if float(rng) < 1e-6:
        return _to_cpu(xp.zeros_like(window))
    return _to_cpu(((window - mn) / rng).astype(xp.float32))


# ── Segmentation ───────────────────────────────────────────────────────────────

def segment_signal(
    ppg,
    fs: float = TARGET_FS,
    window_sec: float = WINDOW_SEC,
    overlap: float = OVERLAP,
) -> np.ndarray:
    """
    Split a continuous PPG into overlapping fixed-length windows
    and normalise each window independently.

    GPU-accelerated: stacking and normalisation happen on the GPU;
    result is transferred to CPU in chunks of _GPU_CHUNK windows to
    avoid large single allocations that cause OOM on low-RAM systems.

    Parameters
    ----------
    ppg        : np.ndarray or cp.ndarray, 1-D (already at target_fs)
    fs         : float
    window_sec : float
    overlap    : float  (0 to < 1)

    Returns
    -------
    windows : np.ndarray, shape (n_windows, window_samples, 1)
    """
    xp       = cp if (_GPU_AVAILABLE and isinstance(ppg, cp.ndarray)) else np
    win_len  = int(fs * window_sec)
    step_len = int(win_len * (1.0 - overlap))

    if len(ppg) < win_len:
        pad = xp.zeros(win_len - len(ppg), dtype=xp.float32)
        ppg = xp.concatenate([ppg, pad])

    starts = list(range(0, len(ppg) - win_len + 1, step_len))
    if not starts:
        return np.empty((0, win_len, 1), dtype=np.float32)

    # Process in chunks to avoid large GPU or CPU allocations
    cpu_chunks = []
    for chunk_start in range(0, len(starts), _GPU_CHUNK):
        chunk_starts = starts[chunk_start:chunk_start + _GPU_CHUNK]

        # Stack chunk windows on GPU → (chunk, win_len)
        raw = xp.stack([ppg[s:s + win_len] for s in chunk_starts], axis=0)

        # Vectorised min-max normalisation
        mn       = raw.min(axis=1, keepdims=True)
        mx       = raw.max(axis=1, keepdims=True)
        rng      = mx - mn
        safe_rng = xp.where(rng < 1e-6, xp.ones_like(rng), rng)
        normed   = ((raw - mn) / safe_rng).astype(xp.float32)

        # Zero out flat windows, drop near-flat
        flat_mask = (rng.squeeze(axis=1) < 1e-6)
        normed[flat_mask] = 0.0
        keep   = normed.std(axis=1) > 1e-4
        normed = normed[keep]

        if normed.shape[0] == 0:
            continue

        # Transfer this small chunk to CPU immediately (frees GPU memory)
        cpu_chunks.append(_to_cpu(normed))
        del raw, normed  # free GPU memory after each chunk

    if not cpu_chunks:
        return np.empty((0, win_len, 1), dtype=np.float32)

    arr_cpu = np.concatenate(cpu_chunks, axis=0)  # (n_kept, win_len)
    return arr_cpu[:, :, np.newaxis]              # (n_kept, win_len, 1)


# ── Full pipeline for a single recording ──────────────────────────────────────

def process_ppg(
    ppg_raw: np.ndarray,
    src_fs: float,
) -> np.ndarray:
    """
    Full conditioning pipeline for one PPG recording.

    Steps:
      0. Trim to MAX_SIGNAL_SEC      (avoids OOM on multi-hour ICU recordings)
      1. Resample to TARGET_FS        (CPU scipy; GPU interp on failure)
      2. Bandpass filter (0.5 – 8 Hz) (CPU scipy, SOS fallback for long signals)
      3. Move filtered signal to GPU  (if CuPy is available)
      4. Segment + normalise          (GPU-accelerated, chunked GPU→CPU transfer)

    Parameters
    ----------
    ppg_raw : np.ndarray, 1-D
    src_fs  : float  – original sampling frequency

    Returns
    -------
    windows : np.ndarray, shape (n_windows, 1250, 1)
    """
    # 0. Trim overly long recordings before resampling
    max_src_samples = int(MAX_SIGNAL_SEC * src_fs)
    if len(ppg_raw) > max_src_samples:
        ppg_raw = ppg_raw[:max_src_samples]

    # 1. Resample (returns CPU numpy array)
    ppg_rs = resample_to_target(ppg_raw, src_fs)

    # 2. Bandpass filter – always on CPU
    ppg_filt = bandpass_filter(_to_cpu(ppg_rs), TARGET_FS)

    # 3. Move to GPU for the vectorised segment + normalise step
    ppg_gpu = _to_gpu(ppg_filt)

    # 4. Segment + normalise (GPU, chunked GPU→CPU transfer)
    windows = segment_signal(ppg_gpu)

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
    t  = np.linspace(0, 120, int(fs * 120))
    ppg_test = (np.sin(2 * np.pi * 1.1 * t) + 0.3 * np.random.randn(len(t))).astype(np.float32)
    wins = process_ppg(ppg_test, fs)
    print(f"GPU available : {_GPU_AVAILABLE}")
    print(f"Output shape  : {wins.shape}")   # expected: (N, 1250, 1)
    print(f"Value range   : [{wins.min():.3f}, {wins.max():.3f}]")
