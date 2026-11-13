"""
run_pipeline.py
===============
End-to-end PPG Arrhythmia Detection pipeline orchestrator.

Runs all steps automatically in sequence:

  Phase 0  – GPU Verification
  Phase 1  – Data Preprocessing & Fusion     (preprocessing/fuse_datasets.py)
  Phase 2  – Feature Extraction              (feature_extraction.py)
  Phase 3  – Multi-Model Training            (train_models.py)
  Phase 4  – Model Benchmarking              (benchmark.py)
  Phase 5  – INT8 TFLite Export + C Headers  (export_models.py)
  Phase 6  – Legacy DS-CNN+LSTM Training     (model/train.py)      [--legacy]
  Phase 7  – Legacy Evaluation               (model/evaluate.py)   [--legacy]
  Phase 8  – Legacy Quantization             (deployment/quantize.py) [--legacy]
  Phase 9  – Legacy C Header                 (deployment/convert_to_c_array.py) [--legacy]

Usage:
    python run_pipeline.py                           # full run (Phases 0-5)
    python run_pipeline.py --quick                   # smoke test (3 epochs)
    python run_pipeline.py --skip-preprocess         # skip Phase 1
    python run_pipeline.py --skip-features           # skip Phase 2
    python run_pipeline.py --skip-train              # skip Phase 3
    python run_pipeline.py --skip-benchmark          # skip Phase 4
    python run_pipeline.py --skip-export             # skip Phase 5
    python run_pipeline.py --models xgb cnn         # train only XGBoost + Edge CNN
    python run_pipeline.py --legacy                  # also run Phases 6-9 (original pipeline)
    python run_pipeline.py --no-gpu                  # force CPU
    python run_pipeline.py --epochs 50 --batch 64
"""

from __future__ import annotations

import os
import sys
import time
import argparse
import logging
import subprocess
import textwrap
from datetime import timedelta

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(ROOT, "outputs")


def _resolve_python() -> str:
    """
    Return the Python interpreter that should be used for all subprocess calls.

    Resolution order
    ----------------
    1. The interpreter that launched THIS script (sys.executable) -- correct
       when the venv is activated before running  `python run_pipeline.py`.
    2. venv_gpu/Scripts/python.exe  in the project root -- used as a fallback
       when run_pipeline.py is launched with the system Python by mistake
       (e.g.  `py run_pipeline.py`  or a bare  `python`  on PATH).
    3. sys.executable as a last resort, with a loud warning.

    The function also performs a quick sanity check by importing `numpy` in
    the resolved interpreter to confirm packages are installed.
    """
    candidates: list[str] = []

    # Candidate 1: current interpreter (correct when venv is activated)
    candidates.append(sys.executable)

    # Candidate 2: venv_gpu in the project root (Windows path)
    venv_scripts = os.path.join(ROOT, "venv_gpu", "Scripts", "python.exe")
    if os.path.isfile(venv_scripts):
        candidates.append(venv_scripts)

    # Candidate 3: venv_gpu on Linux/macOS (fallback for cross-platform use)
    venv_bin = os.path.join(ROOT, "venv_gpu", "bin", "python")
    if os.path.isfile(venv_bin):
        candidates.append(venv_bin)

    for candidate in candidates:
        try:
            result = subprocess.run(
                [candidate, "-c", "import numpy; import pandas; print('ok')"],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode == 0 and "ok" in result.stdout:
                if candidate != sys.executable:
                    logger.warning(
                        "sys.executable (%s) is missing packages.\n"
                        "  Using venv interpreter instead: %s\n"
                        "  TIP: activate the venv first:\n"
                        "       .\\venv_gpu\\Scripts\\activate",
                        sys.executable, candidate,
                    )
                return candidate
        except Exception:
            continue

    # Last resort — return sys.executable with a warning
    logger.warning(
        "Could not find a Python interpreter with numpy/pandas installed.\n"
        "  sys.executable : %s\n"
        "  Searched venv  : %s\n"
        "  Subprocess calls may fail with ModuleNotFoundError.\n"
        "  Fix: activate the venv:  .\\venv_gpu\\Scripts\\activate",
        sys.executable,
        venv_scripts,
    )
    return sys.executable


PYTHON = _resolve_python()
logger.info("Active Python interpreter: %s", PYTHON)

# Legacy pipeline artefacts
DATASET_NPZ  = os.path.join(OUTPUTS_DIR, "dataset.npz")
BEST_MODEL   = os.path.join(OUTPUTS_DIR, "best_model.h5")
TFLITE_OUT   = os.path.join(OUTPUTS_DIR, "model_int8.tflite")
C_HEADER     = os.path.join(ROOT, "firmware", "src", "model.h")

# New pipeline artefacts
FEATURES_TRAIN  = os.path.join(OUTPUTS_DIR, "features_train.parquet")
BENCHMARK_CSV   = os.path.join(OUTPUTS_DIR, "benchmark_results", "benchmark_table.csv")
EXPORT_DIR      = os.path.join(OUTPUTS_DIR, "export")


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

class PipelineError(RuntimeError):
    """Raised when any pipeline step returns a non-zero exit code."""


def _banner(text: str, width: int = 70) -> None:
    """Print a prominent section banner."""
    print(f"\n{'═' * width}")
    print(f"  {text}")
    print(f"{'═' * width}")


def _step_banner(phase: int, name: str) -> None:
    _banner(f"PHASE {phase} — {name}")


def _ok(msg: str)   -> None: print(f"  ✅  {msg}")
def _warn(msg: str) -> None: print(f"  ⚠️   {msg}")
def _skip(msg: str) -> None: print(f"  ⏭️   SKIPPED — {msg}")


def run_step(
    args: list[str],
    phase_name: str,
    *,
    cwd: str = ROOT,
    allow_skip: bool = False,
) -> float:
    """
    Run a subprocess step and return its wall-clock duration in seconds.
    Raises PipelineError on non-zero exit code (unless allow_skip=True).
    """
    cmd_str = " ".join(str(a) for a in args)
    logger.info("Running: %s", cmd_str)
    t0 = time.monotonic()

    result = subprocess.run(args, cwd=cwd)

    elapsed = time.monotonic() - t0
    if result.returncode != 0:
        if allow_skip:
            _warn(f"{phase_name} exited with code {result.returncode} (continuing)")
            return elapsed
        raise PipelineError(
            f"{phase_name} failed (exit code {result.returncode}).\n"
            f"  Command: {cmd_str}\n"
            "  Check the output above for details."
        )
    return elapsed


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 0 — GPU VERIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def phase0_verify_gpu() -> float:
    """Verify TensorFlow can see the GPU (CPU fallback is acceptable)."""
    _step_banner(0, "GPU Verification")
    verify_cmd = [
        PYTHON, "-c",
        "import tensorflow as tf; "
        "gpus = tf.config.list_physical_devices('GPU'); "
        "print('TF:', tf.__version__); "
        "print('GPUs:', gpus); "
        "exit(0)"
    ]
    elapsed = run_step(verify_cmd, "GPU Verification", allow_skip=True)
    _ok("GPU verification complete (CPU fallback is fine)")
    return elapsed


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 1 — DATA PREPROCESSING & FUSION
# ══════════════════════════════════════════════════════════════════════════════

def phase1_preprocess(
    max_mimic: int | None,
    max_ptt:   int | None,
    no_smote:  bool,
    seed:      int,
) -> float:
    """Run the data preprocessing and fusion pipeline."""
    _step_banner(1, "Data Preprocessing & Fusion")

    if os.path.isfile(DATASET_NPZ):
        size_mb = os.path.getsize(DATASET_NPZ) / 1e6
        _warn(f"dataset.npz already exists ({size_mb:.1f} MB). Re-running preprocess.")

    cmd = [PYTHON, "preprocessing/fuse_datasets.py", "--seed", str(seed)]
    if max_mimic is not None:
        cmd += ["--max-mimic-patients", str(max_mimic)]
    if max_ptt is not None:
        cmd += ["--max-ptt-files", str(max_ptt)]
    if no_smote:
        cmd.append("--no-smote")

    elapsed = run_step(cmd, "Preprocessing")
    _ok(f"Dataset saved → {DATASET_NPZ}")
    return elapsed


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 2 — FEATURE EXTRACTION  (for XGBoost / classical ML)
# ══════════════════════════════════════════════════════════════════════════════

def phase2_feature_extraction(fs: float = 125.0) -> float:
    """
    Extract 28 per-window features (time-domain, morphological, HRV,
    frequency-domain) from dataset.npz and write parquet files.
    Required by the XGBoost model in Phase 3.
    """
    _step_banner(2, "Feature Extraction (PPG → Tabular Features)")

    if os.path.isfile(FEATURES_TRAIN):
        size_mb = os.path.getsize(FEATURES_TRAIN) / 1e6
        _warn(f"features_train.parquet already exists ({size_mb:.1f} MB). Re-extracting.")

    cmd = [
        PYTHON, "feature_extraction.py",
        "--fs", str(fs),
        "--input", DATASET_NPZ,
        "--output-dir", OUTPUTS_DIR,
    ]
    elapsed = run_step(cmd, "Feature Extraction")
    _ok(f"Feature parquet files saved → {OUTPUTS_DIR}")
    return elapsed


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 3 — MULTI-MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def phase3_train_models(
    epochs:     int,
    batch_size: int,
    no_gpu:     bool,
    no_mixed:   bool,
    seed:       int,
    models:     list[str],
) -> float:
    """
    Train the requested model architectures:
      xgb     – XGBoost on extracted features
      cnn     – Edge-optimised 1D DS-CNN
      bilstm  – Hybrid Conv1D + Bidirectional LSTM
      tcn     – Temporal Convolutional Network
    """
    _step_banner(3, f"Multi-Model Training  ({', '.join(models)})")

    cmd = [
        PYTHON, "train_models.py",
        "--models",     *models,
        "--epochs",      str(epochs),
        "--batch-size",  str(batch_size),
        "--seed",        str(seed),
    ]
    if no_gpu:
        cmd.append("--no-gpu")
    if no_mixed:
        cmd.append("--no-mixed-precision")

    elapsed = run_step(cmd, "Multi-Model Training")
    _ok(f"All model checkpoints saved → {OUTPUTS_DIR}/")
    return elapsed


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 4 — BENCHMARKING
# ══════════════════════════════════════════════════════════════════════════════

def phase4_benchmark(
    no_gpu: bool,
    models: list[str],
) -> float:
    """
    Evaluate all trained models on the held-out test set.
    Generates:
      - benchmark_table.csv / .txt  (clinical + edge metrics)
      - per-model confusion matrices
      - radar chart comparing all models
    """
    _step_banner(4, "Unified Benchmarking & Evaluation")

    # Map train_models.py names to benchmark.py names
    bench_model_map = {
        "xgb":    "xgb",
        "cnn":    "edge_cnn",
        "bilstm": "bilstm",
        "tcn":    "tcn",
    }
    bench_models = [bench_model_map[m] for m in models if m in bench_model_map]

    cmd = [
        PYTHON, "benchmark.py",
        "--models",      *bench_models,
        "--output-dir",  os.path.join(OUTPUTS_DIR, "benchmark_results"),
    ]
    if no_gpu:
        cmd.append("--no-gpu")

    elapsed = run_step(cmd, "Benchmarking")
    _ok(f"Benchmark results → {os.path.join(OUTPUTS_DIR, 'benchmark_results')}")
    return elapsed


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 5 — TINYML EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def phase5_export(
    rep_samples: int,
    models:      list[str],
) -> float:
    """
    Export all trained models to TinyML formats for ESP32-S3 deployment:
      DL models  → INT8 TFLite → C byte-array header (TFLM)
      XGBoost    → dependency-free C inference code (emlearn / m2cgen)
    """
    _step_banner(5, "TinyML Export (INT8 TFLite + C Headers / XGBoost C Code)")

    # Map train name → export name
    export_model_map = {
        "xgb":    "xgb",
        "cnn":    "edge_cnn",
        "bilstm": "bilstm",
        "tcn":    "tcn",
    }
    export_models = [export_model_map[m] for m in models if m in export_model_map]

    cmd = [
        PYTHON, "export_models.py",
        "--models",      *export_models,
        "--rep-samples",  str(rep_samples),
    ]

    elapsed = run_step(cmd, "TinyML Export")
    _ok(f"Exported artefacts → {EXPORT_DIR}")
    return elapsed


# ══════════════════════════════════════════════════════════════════════════════
#  PHASES 6-9 — LEGACY PIPELINE  (original DS-CNN+LSTM single-model flow)
# ══════════════════════════════════════════════════════════════════════════════

def phase6_legacy_train(
    epochs:     int,
    batch_size: int,
    no_gpu:     bool,
    no_mixed:   bool,
    seed:       int,
) -> float:
    """Train the original single-architecture DS-CNN+LSTM model."""
    _step_banner(6, "[Legacy] DS-CNN+LSTM Model Training")
    cmd = [
        PYTHON, "model/train.py",
        "--epochs",     str(epochs),
        "--batch-size", str(batch_size),
        "--seed",       str(seed),
    ]
    if no_gpu:  cmd.append("--no-gpu")
    if no_mixed: cmd.append("--no-mixed-precision")
    elapsed = run_step(cmd, "Legacy Training")
    _ok(f"Best model → {BEST_MODEL}")
    return elapsed


def phase7_legacy_evaluate(no_gpu: bool) -> float:
    """Evaluate the legacy DS-CNN+LSTM model on the test set."""
    _step_banner(7, "[Legacy] DS-CNN+LSTM Evaluation")
    cmd = [PYTHON, "model/evaluate.py", "--model", BEST_MODEL]
    if no_gpu: cmd.append("--no-gpu")
    elapsed = run_step(cmd, "Legacy Evaluation")
    _ok(f"Reports → {OUTPUTS_DIR}")
    return elapsed


def phase8_legacy_quantize(num_rep_samples: int) -> float:
    """Quantize the legacy model to INT8 TFLite."""
    _step_banner(8, "[Legacy] INT8 TFLite Quantization")
    cmd = [PYTHON, "deployment/quantize.py", "--num-rep-samples", str(num_rep_samples)]
    elapsed = run_step(cmd, "Legacy Quantization")
    _ok(f"TFLite model → {TFLITE_OUT}")
    return elapsed


def phase9_legacy_c_header() -> float:
    """Convert the legacy TFLite model to a C header for ESP32-S3."""
    _step_banner(9, "[Legacy] C Header Generation (ESP32-S3 Firmware)")
    cmd = [
        PYTHON, "deployment/convert_to_c_array.py",
        "--input",  TFLITE_OUT,
        "--output", C_HEADER,
    ]
    elapsed = run_step(cmd, "Legacy C Header Generation")
    _ok(f"C header → {C_HEADER}")
    return elapsed


# ══════════════════════════════════════════════════════════════════════════════
#  SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(timings: dict[str, float], total: float) -> None:
    """Print a colour-coded pipeline summary table."""
    _banner("PIPELINE COMPLETE — Summary", width=70)

    print(f"\n  {'Phase':<40} {'Time':>10}")
    print(f"  {'-'*40} {'-'*10}")
    for phase, secs in timings.items():
        td = str(timedelta(seconds=int(secs)))
        print(f"  {phase:<40} {td:>10}")
    print(f"  {'─'*50}")
    total_td = str(timedelta(seconds=int(total)))
    print(f"  {'TOTAL':<40} {total_td:>10}\n")

    # ── Output file manifest ──────────────────────────────────────────────────
    print("  Output files:")
    outputs = [
        # Core data
        ("Dataset (NPZ)",              DATASET_NPZ),
        ("Features - train",           FEATURES_TRAIN),
        ("Features - val",             os.path.join(OUTPUTS_DIR, "features_val.parquet")),
        ("Features - test",            os.path.join(OUTPUTS_DIR, "features_test.parquet")),
        # Multi-model outputs
        ("XGBoost model",              os.path.join(OUTPUTS_DIR, "xgb", "xgb_model.ubj")),
        ("XGBoost report",             os.path.join(OUTPUTS_DIR, "xgb", "class_report_xgb.txt")),
        ("Edge CNN checkpoint",        os.path.join(OUTPUTS_DIR, "edge_cnn", "best_edge_cnn.h5")),
        ("Edge CNN report",            os.path.join(OUTPUTS_DIR, "edge_cnn", "class_report_edge_cnn.txt")),
        ("BiLSTM checkpoint",          os.path.join(OUTPUTS_DIR, "bilstm",   "best_bilstm.h5")),
        ("BiLSTM report",              os.path.join(OUTPUTS_DIR, "bilstm",   "class_report_bilstm.txt")),
        ("TCN checkpoint",             os.path.join(OUTPUTS_DIR, "tcn",      "best_tcn.h5")),
        ("TCN report",                 os.path.join(OUTPUTS_DIR, "tcn",      "class_report_tcn.txt")),
        # Benchmark
        ("Benchmark table (CSV)",      BENCHMARK_CSV),
        ("Benchmark table (TXT)",      os.path.join(OUTPUTS_DIR, "benchmark_results", "benchmark_table.txt")),
        ("Radar chart",                os.path.join(OUTPUTS_DIR, "benchmark_results", "benchmark_radar.png")),
        ("Per-class F1 chart",         os.path.join(OUTPUTS_DIR, "benchmark_results", "per_class_f1.png")),
        # TinyML exports
        ("Edge CNN TFLite (INT8)",     os.path.join(EXPORT_DIR, "edge_cnn", "model_int8.tflite")),
        ("Edge CNN C header (TFLM)",   os.path.join(EXPORT_DIR, "edge_cnn", "model_data_edge_cnn.h")),
        ("BiLSTM TFLite (INT8)",       os.path.join(EXPORT_DIR, "bilstm",   "model_int8.tflite")),
        ("BiLSTM C header (TFLM)",     os.path.join(EXPORT_DIR, "bilstm",   "model_data_bilstm.h")),
        ("TCN TFLite (INT8)",          os.path.join(EXPORT_DIR, "tcn",      "model_int8.tflite")),
        ("TCN C header (TFLM)",        os.path.join(EXPORT_DIR, "tcn",      "model_data_tcn.h")),
        ("XGBoost C code",             os.path.join(EXPORT_DIR, "xgb",      "xgb_predict.h")),
        ("XGBoost ESP32 wrapper",      os.path.join(EXPORT_DIR, "xgb",      "xgb_esp32_usage.cpp")),
        # Legacy (only if --legacy)
        ("[Legacy] Best model (Keras)", BEST_MODEL),
        ("[Legacy] TFLite INT8",        TFLITE_OUT),
        ("[Legacy] Confusion matrix",   os.path.join(OUTPUTS_DIR, "confusion_matrix.png")),
        ("[Legacy] C header (ESP32)",   C_HEADER),
    ]
    for label, path in outputs:
        exists = "✅" if os.path.exists(path) else "❌"
        print(f"    {exists}  {label:<35} {path}")

    print(f"\n  TensorBoard: tensorboard --logdir {OUTPUTS_DIR}/logs")
    print(f"  Benchmark results: {os.path.join(OUTPUTS_DIR, 'benchmark_results')}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end PPG Arrhythmia Detection pipeline (multi-model)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Examples:
          python run_pipeline.py                          # full run (Phases 0-5)
          python run_pipeline.py --quick                  # 3 epochs, 10 MIMIC patients
          python run_pipeline.py --skip-preprocess        # skip Phase 1
          python run_pipeline.py --skip-features          # skip Phase 2
          python run_pipeline.py --skip-train             # skip Phase 3
          python run_pipeline.py --skip-benchmark         # skip Phase 4
          python run_pipeline.py --skip-export            # skip Phase 5
          python run_pipeline.py --models xgb cnn        # only XGBoost + Edge CNN
          python run_pipeline.py --legacy                 # also run original DS-CNN+LSTM
          python run_pipeline.py --no-gpu                 # force CPU
        """),
    )

    # ── Phase control ─────────────────────────────────────────────────────────
    parser.add_argument("--skip-preprocess", action="store_true",
                        help="Skip Phase 1: preprocessing (use existing dataset.npz)")
    parser.add_argument("--skip-features",   action="store_true",
                        help="Skip Phase 2: feature extraction (use existing parquet files)")
    parser.add_argument("--skip-train",      action="store_true",
                        help="Skip Phase 3: training (use existing checkpoints)")
    parser.add_argument("--skip-benchmark",  action="store_true",
                        help="Skip Phase 4: benchmarking")
    parser.add_argument("--skip-export",     action="store_true",
                        help="Skip Phase 5: TinyML export")
    parser.add_argument("--legacy",          action="store_true",
                        help="Also run the original DS-CNN+LSTM pipeline (Phases 6-9)")
    parser.add_argument("--skip-legacy-eval",    action="store_true",
                        help="[--legacy] Skip legacy evaluation Phase 7")
    parser.add_argument("--skip-legacy-deploy",  action="store_true",
                        help="[--legacy] Skip legacy quantize + C header (Phases 8-9)")

    # ── Model selection ───────────────────────────────────────────────────────
    parser.add_argument(
        "--models", nargs="+",
        choices=["xgb", "cnn", "bilstm", "tcn"],
        default=["xgb", "cnn", "bilstm", "tcn"],
        help="Which models to train, benchmark, and export (default: all four)",
    )

    # ── Training params ───────────────────────────────────────────────────────
    parser.add_argument("--epochs",     type=int, default=100,
                        help="Training epochs (default: 100)")
    parser.add_argument("--batch",      type=int, default=64,
                        help="Per-replica batch size (default: 64)")
    parser.add_argument("--seed",       type=int, default=42)

    # ── Preprocessing params ──────────────────────────────────────────────────
    parser.add_argument("--max-mimic",  type=int, default=None,
                        help="Max MIMIC patient dirs to load (default: all)")
    parser.add_argument("--max-ptt",    type=int, default=None,
                        help="Max PTT CSV files to load (default: all)")
    parser.add_argument("--no-smote",   action="store_true",
                        help="Disable SMOTE oversampling")
    parser.add_argument("--fs",         type=float, default=125.0,
                        help="Sampling frequency in Hz for feature extraction (default: 125)")

    # ── Export params ─────────────────────────────────────────────────────────
    parser.add_argument("--rep-samples", type=int, default=200,
                        help="INT8 PTQ calibration samples (default: 200)")

    # ── GPU / precision ───────────────────────────────────────────────────────
    parser.add_argument("--no-gpu",             action="store_true",
                        help="Force CPU-only execution")
    parser.add_argument("--no-mixed-precision", action="store_true",
                        help="Disable float16 mixed precision")

    # ── Quick mode ────────────────────────────────────────────────────────────
    parser.add_argument("--quick", action="store_true",
                        help="Smoke test: 3 epochs, 10 MIMIC patients, 5 PTT files")

    args = parser.parse_args()

    # ── Apply quick-mode overrides ────────────────────────────────────────────
    if args.quick:
        logger.info("Quick mode ON — limiting data and epochs for smoke test")
        args.epochs    = 3
        args.max_mimic = args.max_mimic or 10
        args.max_ptt   = args.max_ptt   or 5

    # ── Pre-flight checks ─────────────────────────────────────────────────────
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    if args.skip_train and not any(
        os.path.isfile(os.path.join(OUTPUTS_DIR, d, f"best_{m}.h5"))
        for d, m in [("edge_cnn", "edge_cnn"), ("bilstm", "bilstm"), ("tcn", "tcn")]
    ) and not os.path.isfile(os.path.join(OUTPUTS_DIR, "xgb", "xgb_model.ubj")):
        logger.warning(
            "--skip-train specified but no model checkpoints found in %s.\n"
            "  Run without --skip-train to train models first.",
            OUTPUTS_DIR,
        )

    if (args.skip_preprocess or args.skip_train) and not os.path.isfile(DATASET_NPZ):
        logger.error(
            "--skip-preprocess/--skip-train specified but no dataset found at %s.\n"
            "  Remove the flag to generate it first.",
            DATASET_NPZ,
        )
        sys.exit(1)

    if args.skip_features and not os.path.isfile(FEATURES_TRAIN):
        if "xgb" in args.models:
            logger.warning(
                "--skip-features specified but no feature files found. "
                "XGBoost training will fail without them."
            )

    # ── Run phases ────────────────────────────────────────────────────────────
    pipeline_start = time.monotonic()
    timings: dict[str, float] = {}

    try:
        # Phase 0 — GPU check (always runs)
        timings["Phase 0: GPU Verification"] = phase0_verify_gpu()

        # Phase 1 — Preprocessing
        if args.skip_preprocess or args.skip_train:
            _skip("Phase 1: Preprocessing (--skip-preprocess or --skip-train)")
        else:
            timings["Phase 1: Preprocessing"] = phase1_preprocess(
                max_mimic=args.max_mimic,
                max_ptt=args.max_ptt,
                no_smote=args.no_smote,
                seed=args.seed,
            )

        # Phase 2 — Feature Extraction
        if args.skip_features or args.skip_train:
            _skip("Phase 2: Feature Extraction (--skip-features or --skip-train)")
        else:
            timings["Phase 2: Feature Extraction"] = phase2_feature_extraction(
                fs=args.fs,
            )

        # Phase 3 — Multi-Model Training
        if args.skip_train:
            _skip("Phase 3: Multi-Model Training (--skip-train)")
        else:
            timings["Phase 3: Multi-Model Training"] = phase3_train_models(
                epochs=args.epochs,
                batch_size=args.batch,
                no_gpu=args.no_gpu,
                no_mixed=args.no_mixed_precision,
                seed=args.seed,
                models=args.models,
            )

        # Phase 4 — Benchmarking
        if args.skip_benchmark:
            _skip("Phase 4: Benchmarking (--skip-benchmark)")
        else:
            timings["Phase 4: Benchmarking"] = phase4_benchmark(
                no_gpu=args.no_gpu,
                models=args.models,
            )

        # Phase 5 — TinyML Export
        if args.skip_export:
            _skip("Phase 5: TinyML Export (--skip-export)")
        else:
            timings["Phase 5: TinyML Export"] = phase5_export(
                rep_samples=args.rep_samples,
                models=args.models,
            )

        # ── Legacy pipeline (Phases 6-9) ──────────────────────────────────────
        if args.legacy:
            _banner("LEGACY PIPELINE — Original DS-CNN+LSTM Flow", width=70)

            if args.skip_train:
                _skip("Phase 6: Legacy Training (--skip-train)")
            else:
                timings["Phase 6: [Legacy] Training"] = phase6_legacy_train(
                    epochs=args.epochs,
                    batch_size=args.batch,
                    no_gpu=args.no_gpu,
                    no_mixed=args.no_mixed_precision,
                    seed=args.seed,
                )

            if args.skip_legacy_eval:
                _skip("Phase 7: Legacy Evaluation (--skip-legacy-eval)")
            else:
                timings["Phase 7: [Legacy] Evaluation"] = phase7_legacy_evaluate(
                    no_gpu=args.no_gpu,
                )

            if args.skip_legacy_deploy:
                _skip("Phases 8-9: Legacy Deploy (--skip-legacy-deploy)")
            else:
                timings["Phase 8: [Legacy] Quantization"] = phase8_legacy_quantize(
                    num_rep_samples=args.rep_samples,
                )
                timings["Phase 9: [Legacy] C Header"] = phase9_legacy_c_header()

    except PipelineError as exc:
        _banner("PIPELINE FAILED", width=70)
        logger.error("%s", exc)
        sys.exit(1)
    except KeyboardInterrupt:
        _banner("PIPELINE INTERRUPTED BY USER", width=70)
        sys.exit(130)

    total_elapsed = time.monotonic() - pipeline_start
    print_summary(timings, total_elapsed)


if __name__ == "__main__":
    main()
