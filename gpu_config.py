"""
gpu_config.py
=============
Centralized GPU initialization for the PPG Arrhythmia Detection project.

Call `setup_gpu()` at the very top of every script BEFORE importing TensorFlow
so that device configuration is applied before the TF runtime initializes.

Usage
-----
    from gpu_config import setup_gpu
    strategy, num_replicas = setup_gpu(mixed_precision=True)

    with strategy.scope():
        model = build_model()

CUDA Version Compatibility
--------------------------
TensorFlow 2.10.1 was compiled against CUDA 11.x + cuDNN 8.1 and links
against cublasLt64_11.dll on Windows.  If CUDA 12.x is installed on the
system (cublasLt64_12.dll), TF 2.10.1 cannot use it and prints:

    "Could not locate cublasLt64_12.dll. Please make sure it is in your
     library path!"

This is a non-fatal warning.  TensorFlow silently falls back to CPU.
This config module detects the mismatch early and clearly logs it so you
know whether you are running on GPU or CPU.

Fix Options (choose one)
------------------------
  A. Add --no-gpu flag  →  explicit CPU-only (fastest fix)
  B. Install CUDA 11.8 + cuDNN 8.1 alongside CUDA 12
     https://developer.nvidia.com/cuda-11-8-0-download-archive
  C. Upgrade TF to a version that supports CUDA 12:
       pip install tensorflow>=2.13.0  (requires Python 3.9-3.11)

Features
--------
* CUDA DLL probe    – detects version mismatch before TF loads, avoids crash
* Per-device memory growth  – prevents TF grabbing all VRAM upfront
* Mixed Precision policy    – float16 compute + float32 variables
* MirroredStrategy          – transparent multi-GPU support
* Detailed logging          – reports device names, VRAM, driver, active policy
"""

from __future__ import annotations

import sys
import logging
import os

logger = logging.getLogger(__name__)

# TF 2.10.1 was compiled for CUDA 11.x -- it links against these DLLs on Windows.
# If we find CUDA 12 DLLs but NOT CUDA 11 DLLs, warn early and fall back to CPU.
_REQUIRED_CUDA_DLL  = "cublas64_11.dll"    # required by TF 2.10.1
_CONFLICT_CUDA_DLL  = "cublasLt64_12.dll"  # present when CUDA 12 is on PATH


def _probe_cuda_dlls() -> tuple[bool, str]:
    """
    Probe the Windows DLL search path for CUDA library compatibility.

    Returns
    -------
    (cuda_ok, reason) where:
      cuda_ok : True if TF 2.10.1-compatible CUDA 11 DLLs are available
      reason  : human-readable explanation of the probe result
    """
    if sys.platform != "win32":
        return True, "Non-Windows platform — DLL probe skipped"

    import ctypes.util

    has_cuda11 = ctypes.util.find_library(_REQUIRED_CUDA_DLL) is not None
    has_cuda12 = ctypes.util.find_library(_CONFLICT_CUDA_DLL) is not None

    if has_cuda11:
        return True, f"CUDA 11 DLL found ({_REQUIRED_CUDA_DLL}) — GPU compatible"

    if has_cuda12:
        return False, (
            f"CUDA 12 detected ({_CONFLICT_CUDA_DLL}) but TF 2.10.1 requires "
            f"CUDA 11.x ({_REQUIRED_CUDA_DLL}).\n"
            "  -> Falling back to CPU automatically.\n"
            "  Fix options:\n"
            "    A. python run_pipeline.py --no-gpu          (explicit CPU)\n"
            "    B. Install CUDA 11.8: https://developer.nvidia.com/cuda-11-8-0-download-archive\n"
            "    C. pip install tensorflow>=2.13.0           (supports CUDA 12)"
        )

    return False, (
        f"No CUDA DLLs found ({_REQUIRED_CUDA_DLL} or {_CONFLICT_CUDA_DLL}).\n"
        "  -> Falling back to CPU.\n"
        "  To enable GPU:\n"
        "    Install CUDA 11.8 + cuDNN 8.1:\n"
        "    https://developer.nvidia.com/cuda-11-8-0-download-archive"
    )


def _add_nvidia_dll_dirs() -> None:
    """
    Add NVIDIA pip-package DLL directories to the Windows search path.

    When TF or CUDA is installed via pip (e.g. nvidia-cuda-runtime-cu11),
    the DLLs land inside site-packages/nvidia/.../bin/ which is NOT on the
    system PATH.  os.add_dll_directory() (Python 3.8+) registers them.
    """
    if sys.platform != "win32":
        return
    import site
    try:
        site_dirs = site.getsitepackages() + [site.getusersitepackages()]
    except AttributeError:
        site_dirs = []
    for s_dir in site_dirs:
        nvidia_dir = os.path.join(s_dir, "nvidia")
        if not os.path.isdir(nvidia_dir):
            continue
        for root, dirs, files in os.walk(nvidia_dir):
            for sub in ("bin", "lib"):
                if sub in dirs:
                    target = os.path.join(root, sub)
                    try:
                        os.add_dll_directory(target)
                        os.environ["PATH"] = (
                            target + os.pathsep + os.environ.get("PATH", "")
                        )
                    except Exception:
                        pass


def setup_gpu(
    *,
    mixed_precision: bool = True,
    no_gpu: bool = False,
    log_device_placement: bool = False,
) -> tuple:
    """
    Configure TensorFlow GPU settings and return a distribution strategy.

    Parameters
    ----------
    mixed_precision : bool
        Enable ``keras.mixed_precision`` global policy ``"mixed_float16"``.
        Cuts VRAM usage roughly in half and accelerates matmuls on NVIDIA GPUs
        with Tensor Cores (Pascal / GTX 10xx and newer).
        The output layer is always cast to ``float32`` in the model definition
        so that loss computation is numerically stable.
        NOTE: auto-disabled for Pascal GPUs (Compute Capability < 7.0) and
        when no compatible CUDA runtime is found.
    no_gpu : bool
        Force CPU-only execution (equivalent to ``--no-gpu`` CLI flag).
    log_device_placement : bool
        Log which device each TF op is placed on (verbose, for debugging).

    Returns
    -------
    strategy : tf.distribute.Strategy
        ``OneDeviceStrategy(/gpu:0)`` for a single GPU,
        ``MirroredStrategy``       for multiple GPUs,
        ``OneDeviceStrategy(/cpu:0)`` as CPU fallback.
    num_replicas : int
        Number of active replicas (use to scale global batch size).
    """
    # ── Step 1: Add NVIDIA pip-package DLL directories (Windows) ─────────────
    _add_nvidia_dll_dirs()

    # ── Step 2: Probe for CUDA DLL version compatibility (Windows) ───────────
    cuda_ok = True
    if not no_gpu and sys.platform == "win32":
        cuda_ok, reason = _probe_cuda_dlls()
        if cuda_ok:
            logger.info("CUDA DLL probe: %s", reason)
        else:
            logger.warning("CUDA DLL probe: %s", reason)

    import tensorflow as tf

    # ── Step 3: Force CPU if requested or if CUDA probe failed ────────────────
    if no_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        logger.info("GPU disabled (--no-gpu) — running on CPU.")
        _set_precision(tf, mixed_precision=False)
        return tf.distribute.OneDeviceStrategy("/cpu:0"), 1

    if not cuda_ok:
        # CUDA DLL mismatch detected — avoid the TF DLL error spam by
        # hiding GPUs before TF's CUDA runtime tries to open them.
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
        logger.info("Falling back to CPU due to CUDA version mismatch.")
        _set_precision(tf, mixed_precision=False)
        return tf.distribute.OneDeviceStrategy("/cpu:0"), 1

    # ── Step 4: Optional op-placement logging ─────────────────────────────────
    if log_device_placement:
        tf.debugging.set_log_device_placement(True)

    # ── Step 5: Enumerate GPUs and enable memory growth ───────────────────────
    gpus = tf.config.list_physical_devices("GPU")

    if not gpus:
        logger.warning(
            "No GPU detected by TensorFlow — falling back to CPU.\n"
            "  Possible causes:\n"
            "    1. CUDA 11.8 + cuDNN 8.1 not installed (TF 2.10.1 requirement)\n"
            "    2. CUDA PATH not set correctly\n"
            "    3. CUDA 12 installed but not CUDA 11\n"
            "  Verify: python -c \"import tensorflow as tf; "
            "print(tf.config.list_physical_devices('GPU'))\""
        )
        _set_precision(tf, mixed_precision=False)
        return tf.distribute.OneDeviceStrategy("/cpu:0"), 1

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("Memory growth enabled  ->  %s", gpu.name)
        except RuntimeError as exc:
            # Memory growth must be set before any GPU ops run.
            logger.warning("Could not enable memory growth on %s: %s", gpu.name, exc)

    logger.info("GPU(s) detected: %s", [g.name for g in gpus])

    # ── Step 6: Mixed precision (auto-disable for Pascal / CC < 7.0) ──────────
    # GTX 1050 Ti = Compute Capability 6.1 — does NOT support mixed_float16 safely.
    # TF's cuBLAS half-precision kernels require CC >= 7.0 (Volta / RTX series).
    if mixed_precision:
        try:
            details = tf.config.experimental.get_device_details(gpus[0])
            cc = details.get("compute_capability", (0, 0))
            if cc and cc[0] < 7:
                logger.warning(
                    "GPU Compute Capability %s.%s < 7.0 (Pascal / GTX 10-series).\n"
                    "  mixed_float16 disabled — cuBLAS FP16 not reliable below CC 7.0.\n"
                    "  Training will use float32 (correct results, no crash).",
                    cc[0], cc[1],
                )
                mixed_precision = False
        except Exception:
            pass  # get_device_details may fail on some TF versions — safe to ignore

    _set_precision(tf, mixed_precision)

    # ── Step 7: Distribution strategy ─────────────────────────────────────────
    if len(gpus) == 1:
        strategy = tf.distribute.OneDeviceStrategy("/gpu:0")
        logger.info("Strategy  ->  OneDeviceStrategy  (/gpu:0  |  %s)", gpus[0].name)
    else:
        strategy = tf.distribute.MirroredStrategy()
        logger.info(
            "Strategy  ->  MirroredStrategy  (%d GPUs)",
            strategy.num_replicas_in_sync,
        )

    return strategy, strategy.num_replicas_in_sync


def _set_precision(tf, mixed_precision: bool) -> None:
    """Apply the Keras global dtype policy and log it."""
    if mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        logger.info(
            "Mixed precision ENABLED  ->  policy='mixed_float16'  "
            "(compute=float16, variables=float32)"
        )
    else:
        tf.keras.mixed_precision.set_global_policy("float32")
        logger.info("Mixed precision DISABLED  ->  policy='float32'")
