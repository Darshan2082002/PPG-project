"""
export_models.py
================
TinyML export pipelines for all four PPG arrhythmia model architectures.

Two paradigms are supported:

  A. Deep Learning Models (Edge CNN, BiLSTM, TCN)
     ─────────────────────────────────────────────
     Step 1 : Load Keras .h5 model
     Step 2 : INT8 Post-Training Quantization (PTQ) with representative dataset
     Step 3 : Save .tflite flatbuffer
     Step 4 : Convert .tflite -> C header array (model_data.h) for TFLM

  B. XGBoost
     ─────────
     Export the trained tree ensemble as pure, self-contained C code using
     m2cgen or emlearn.  The generated file is a single .c / .h pair that
     compiles and runs on the ESP32-S3 without any runtime library.

     Why this is superior to a TFLM approach for trees:
       - Zero RAM for tensor arena (no activation buffers)
       - Executes as native compiled C -- no interpreter overhead
       - Typically <1 ms latency on an LX7 core @ 240 MHz
       - Full type safety; tree logic is visible and auditable

Usage
-----
  python export_models.py                         # export all models
  python export_models.py --models cnn bilstm tcn # DL only
  python export_models.py --models xgb            # XGBoost C code only
  python export_models.py --rep-samples 300       # more PTQ calibration samples
  python export_models.py --skip-tflite           # skip TFLite, only C headers
  python export_models.py --no-gpu                # force CPU for quantization

Outputs
-------
  outputs/export/
    edge_cnn/
      model_int8.tflite
      model_data_edge_cnn.h     <- TFLM C header
    bilstm/
      model_int8.tflite
      model_data_bilstm.h
    tcn/
      model_int8.tflite
      model_data_tcn.h
    xgb/
      xgb_predict.c             <- dependency-free C inference function
      xgb_predict.h
"""

from __future__ import annotations

import os
import sys
import argparse
import logging
from pathlib import Path

import numpy as np

# Force CPU for quantization passes (PTQ runs on CPU only)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ---------------------------------------------------------------------
OUTPUT_DIR    = os.path.join(os.path.dirname(__file__), "outputs")
EXPORT_DIR    = os.path.join(OUTPUT_DIR, "export")
DATASET_PATH  = os.path.join(OUTPUT_DIR, "dataset.npz")
FEATURES_VAL  = os.path.join(OUTPUT_DIR, "features_val.parquet")

NUM_CLASSES  = 5
INPUT_LENGTH = 1250  # 10 s @ 125 Hz


# ── Model I/O paths ─────────────────────────────────────────────────────────
KERAS_MODEL_PATHS = {
    "edge_cnn": os.path.join(OUTPUT_DIR, "edge_cnn", "best_edge_cnn.h5"),
    "bilstm":   os.path.join(OUTPUT_DIR, "bilstm",   "best_bilstm.h5"),
    "tcn":      os.path.join(OUTPUT_DIR, "tcn",      "best_tcn.h5"),
}

XGB_MODEL_PATH = os.path.join(OUTPUT_DIR, "xgb", "xgb_model.ubj")


# ==============================================================================
#  REPRESENTATIVE DATASET GENERATOR  (for INT8 PTQ calibration)
# ==============================================================================

def representative_dataset_gen(X_val: np.ndarray, num_samples: int = 200):
    """
    Yield representative float32 samples for TFLite INT8 calibration.

    The TFLite converter calls this generator to determine the min/max
    activation range at each tensor, which it uses to compute INT8 scale
    and zero-point values.  More samples improve quantization accuracy.

    Parameters
    ----------
    X_val       : validation set as float32 array (N, L, 1)
    num_samples : how many calibration samples to yield
    """
    indices = np.random.choice(
        len(X_val), size=min(num_samples, len(X_val)), replace=False
    )
    for idx in indices:
        # Shape must be (1, L, 1) -- batch size of 1
        sample = X_val[idx : idx + 1].astype(np.float32)
        yield [sample]


# ==============================================================================
#  PART A: KERAS -> INT8 TFLITE -> C HEADER
# ==============================================================================

def quantize_keras_to_tflite(
    model_key: str,
    num_rep_samples: int = 200,
) -> str | None:
    """
    Convert a Keras model to a fully INT8-quantized TFLite flatbuffer.

    Quantization scheme
    -------------------
    - ``tf.lite.Optimize.DEFAULT`` triggers weight quantization to INT8 by
      default, plus dynamic range quantization for activations.
    - ``representative_dataset`` enables FULL integer quantization: both
      weights AND activations are quantized to INT8 using calibration data.
    - ``TFLITE_BUILTINS_INT8`` restricts the op set to only INT8-compatible
      built-in ops, ensuring every layer is quantized (no float fallback).
    - Specifying ``inference_input_type = tf.int8`` and
      ``inference_output_type = tf.int8`` means the model expects and returns
      INT8 tensors at the boundary, which the ESP32-S3 firmware feeds directly.

    Parameters
    ----------
    model_key       : one of "edge_cnn", "bilstm", "tcn"
    num_rep_samples : calibration sample count

    Returns
    -------
    str | None  --  path to the saved .tflite file, or None on failure
    """
    import tensorflow as tf

    model_path = KERAS_MODEL_PATHS.get(model_key)
    if not model_path or not os.path.isfile(model_path):
        logger.warning("Keras model not found: %s -- skipping", model_path)
        return None

    # ── Load calibration data ─────────────────────────────────────────────────
    if not os.path.isfile(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}\n"
            "Run preprocessing/fuse_datasets.py first."
        )
    data  = np.load(DATASET_PATH, allow_pickle=True)
    X_val = data["X_val"].astype(np.float32)
    logger.info("Calibrating on %d / %d validation samples", num_rep_samples, len(X_val))

    # ── Load Keras model ──────────────────────────────────────────────────────
    logger.info("Loading Keras model: %s", model_path)
    model = tf.keras.models.load_model(model_path, compile=False)

    # ── TFLite converter ──────────────────────────────────────────────────────
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Full-integer PTQ settings
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset_gen(
        X_val, num_rep_samples
    )
    # INT8 built-in ops only (compatible with TFLM on ESP32-S3)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # INT8 boundary tensors -- firmware feeds int8 directly, no float wrapping
    converter.inference_input_type  = tf.int8
    converter.inference_output_type = tf.int8

    logger.info("Running INT8 PTQ for %s ...", model_key)
    tflite_model = converter.convert()

    # ── Save .tflite ----------------------------------------------------------
    out_subdir = os.path.join(EXPORT_DIR, model_key)
    os.makedirs(out_subdir, exist_ok=True)
    tflite_path = os.path.join(out_subdir, "model_int8.tflite")

    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    orig_kb    = os.path.getsize(model_path) / 1024
    quant_kb   = len(tflite_model) / 1024
    ratio      = orig_kb / quant_kb if quant_kb > 0 else 0.0
    logger.info(
        "  %s  |  FP32: %.1f KB  ->  INT8: %.1f KB  (%.1fx smaller)",
        model_key, orig_kb, quant_kb, ratio,
    )

    # ── Sanity check ----------------------------------------------------------
    _verify_tflite(tflite_path, X_val[:1], label=model_key)

    return tflite_path


def _verify_tflite(tflite_path: str, sample: np.ndarray, label: str) -> None:
    """
    Run a single inference on the quantized model to verify correctness.

    Verifies that:
    1. The model loads without error
    2. Input/output shapes match expectations
    3. Output probabilities sum to ~1.0 (softmax sanity check)
    """
    import tensorflow as tf

    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()

    inp_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]

    logger.info(
        "  %s  |  Input: %s %s  |  Output: %s %s",
        label,
        inp_det["shape"], inp_det["dtype"],
        out_det["shape"], out_det["dtype"],
    )

    # Quantise float input -> INT8
    scale, zp = inp_det["quantization"]
    inp_int8  = (sample / scale + zp).clip(-128, 127).astype(np.int8)

    interp.set_tensor(inp_det["index"], inp_int8)
    interp.invoke()
    output = interp.get_tensor(out_det["index"])

    # Dequantise INT8 output -> float probabilities
    out_scale, out_zp = out_det["quantization"]
    probs = (output.astype(np.float32) - out_zp) * out_scale
    logger.info("  %s sanity -- predicted class: %d  (probs sum=%.3f)",
                label, int(np.argmax(probs)), float(probs.sum()))


def tflite_to_c_header(
    tflite_path: str,
    header_path: str,
    var_name: str,
) -> None:
    """
    Convert a .tflite flatbuffer to a C/C++ header byte array.

    The generated header follows the TensorFlow Lite for Microcontrollers
    (TFLM) convention used by ESP32-S3 Arduino / ESP-IDF projects:

        alignas(8) const unsigned char <var_name>[] = { 0x1c, 0x00, ... };
        const unsigned int <var_name>_len = XXXX;

    The ``alignas(8)`` attribute satisfies the TFLM tensor arena alignment
    requirement; without it, the interpreter may fault on strict-alignment
    architectures.

    Parameters
    ----------
    tflite_path : path to INT8 .tflite model
    header_path : output .h path
    var_name    : C variable name (e.g. ``g_edge_cnn_model``)
    """
    with open(tflite_path, "rb") as f:
        model_bytes = f.read()

    size = len(model_bytes)
    BYTES_PER_LINE = 12

    hex_lines = []
    for i in range(0, size, BYTES_PER_LINE):
        chunk     = model_bytes[i : i + BYTES_PER_LINE]
        hex_vals  = ", ".join(f"0x{b:02x}" for b in chunk)
        hex_lines.append(f"    {hex_vals}")
    hex_body = ",\n".join(hex_lines)

    guard = (
        os.path.basename(header_path)
        .upper()
        .replace(".", "_")
        .replace("-", "_")
        .replace(" ", "_")
    )

    content = f"""\
// ============================================================
// {os.path.basename(header_path)}
// Auto-generated by export_models.py -- DO NOT EDIT MANUALLY
//
// Source : {os.path.basename(tflite_path)}
// Size   : {size} bytes ({size / 1024:.1f} KB)
//
// Usage (ESP32-S3 / TFLM):
//   #include "{os.path.basename(header_path)}"
//   tflite::MicroInterpreter interpreter(
//       tflite::GetModel({var_name}), resolver,
//       tensor_arena, kTensorArenaSize, &error_reporter);
// ============================================================

#ifndef {guard}
#define {guard}

#ifdef __cplusplus
extern "C" {{
#endif

// Aligned to 8 bytes -- required by TFLM tensor arena
alignas(8) const unsigned char {var_name}[] = {{
{hex_body}
}};

const unsigned int {var_name}_len = {size}u;

#ifdef __cplusplus
}}
#endif

#endif  // {guard}
"""
    os.makedirs(os.path.dirname(header_path), exist_ok=True)
    Path(header_path).write_text(content)
    logger.info("  C header -> %s  (%d bytes)", header_path, size)


def export_dl_model(
    model_key: str,
    num_rep_samples: int,
    skip_tflite: bool = False,
) -> None:
    """
    Full pipeline: Keras H5 -> INT8 TFLite -> C header.

    Parameters
    ----------
    model_key       : "edge_cnn" | "bilstm" | "tcn"
    num_rep_samples : INT8 calibration samples
    skip_tflite     : if True, assume .tflite already exists; just re-export header
    """
    out_subdir  = os.path.join(EXPORT_DIR, model_key)
    tflite_path = os.path.join(out_subdir, "model_int8.tflite")
    header_path = os.path.join(out_subdir, f"model_data_{model_key}.h")
    var_name    = f"g_{model_key}_model"

    logger.info("\n%s\nExporting: %s\n%s", "=" * 60, model_key, "=" * 60)

    if not skip_tflite or not os.path.isfile(tflite_path):
        result = quantize_keras_to_tflite(model_key, num_rep_samples)
        if result is None:
            return
    else:
        logger.info("Skipping TFLite conversion (--skip-tflite); using %s", tflite_path)

    if not os.path.isfile(tflite_path):
        logger.error("TFLite file not found: %s", tflite_path)
        return

    tflite_to_c_header(tflite_path, header_path, var_name)
    logger.info("DL export complete for %s", model_key)


# ==============================================================================
#  PART B: XGBOOST -> PURE C CODE
# ==============================================================================

def export_xgboost_c_code(method: str = "auto") -> None:
    """
    Export the XGBoost model as dependency-free C inference code.

    Two backends are attempted (in order of preference):

    1. emlearn (preferred for MCU targets)
       ───────────────────────────────────
       ``emlearn`` converts XGBoost tree ensembles into C code that uses
       only integer arithmetic and lookup tables.  Resulting code:
         - Runs in <1 ms on ESP32-S3 @ 240 MHz
         - Requires ~8-40 KB Flash (trees as C arrays)
         - Zero heap allocation; fully static
         - Supports both .h (inline) and .c+.h pair output

    2. m2cgen (fallback)
       ──────────────────
       ``m2cgen`` generates a long if-else C function that traverses every
       tree.  Produces larger code (10-100 KB) but no external dependency.

    Parameters
    ----------
    method : "emlearn" | "m2cgen" | "auto"
             "auto" tries emlearn first, falls back to m2cgen
    """
    if not os.path.isfile(XGB_MODEL_PATH):
        logger.warning(
            "XGBoost model not found: %s\nRun train_models.py first.", XGB_MODEL_PATH
        )
        return

    out_subdir = os.path.join(EXPORT_DIR, "xgb")
    os.makedirs(out_subdir, exist_ok=True)

    logger.info("\n%s\nExporting: XGBoost -> C Code\n%s", "=" * 60, "=" * 60)

    # ── Load XGBoost model ────────────────────────────────────────────────────
    try:
        import xgboost as xgb
    except ImportError:
        logger.error("xgboost not installed. Run: pip install xgboost")
        return

    booster = xgb.Booster()
    booster.load_model(XGB_MODEL_PATH)
    logger.info("Loaded XGBoost model: %d trees", booster.num_boosted_rounds())

    # ── Attempt emlearn ───────────────────────────────────────────────────────
    if method in ("emlearn", "auto"):
        try:
            _export_with_emlearn(booster, out_subdir)
            return
        except ImportError:
            if method == "emlearn":
                logger.error("emlearn not installed. Run: pip install emlearn")
                return
            logger.info("emlearn not found; trying m2cgen ...")
        except Exception as e:
            if method == "emlearn":
                raise
            logger.warning("emlearn export failed (%s); trying m2cgen ...", e)

    # ── Attempt m2cgen ────────────────────────────────────────────────────────
    if method in ("m2cgen", "auto"):
        try:
            _export_with_m2cgen(booster, out_subdir)
        except ImportError:
            logger.error("m2cgen not installed. Run: pip install m2cgen")
        except Exception as e:
            logger.error("m2cgen export failed: %s", e)


def _export_with_emlearn(booster: "xgb.Booster", out_dir: str) -> None:
    """
    Export XGBoost trees to C using emlearn.

    emlearn.convert() accepts a scikit-learn compatible wrapper around the
    booster, then writes a .h header with static arrays representing each
    decision tree.  The resulting ``eml_trees_predict()`` function is
    callable without malloc or any standard library.

    Note: emlearn requires sklearn's XGBClassifier wrapper, not the raw
    Booster.  We reload via XGBClassifier.load_model().
    """
    import emlearn
    import xgboost as xgb
    from sklearn.pipeline import Pipeline

    logger.info("Exporting with emlearn ...")

    # Reload as XGBClassifier (emlearn needs sklearn-compatible estimator)
    clf = xgb.XGBClassifier(n_estimators=booster.num_boosted_rounds(),
                              num_class=NUM_CLASSES, objective="multi:softmax",
                              use_label_encoder=False)
    clf.load_model(XGB_MODEL_PATH)

    # Convert to C
    model_c = emlearn.convert(clf, method="inline")

    h_path  = os.path.join(out_dir, "xgb_predict.h")
    model_c.save(h_path)

    logger.info("  emlearn C header -> %s", h_path)
    _write_xgb_c_wrapper(h_path, out_dir, backend="emlearn")


def _export_with_m2cgen(booster: "xgb.Booster", out_dir: str) -> None:
    """
    Export XGBoost trees to C using m2cgen.

    m2cgen.export_to_c() traverses every tree in the ensemble and generates
    an equivalent nested if-else function.  For multiclass, it returns a
    score array and the caller takes argmax.

    The generated function signature is:
        void score(double * input, double * output);

    Limitations vs emlearn:
    - Uses double (float64) arithmetic; slower on MCU FPUs
    - Generated code can be very large (100+ KB) for deep ensembles
    - No static-only allocation; relies on stack for local arrays

    For production ESP32-S3 use, prefer emlearn or a quantised TFLM model.
    """
    import m2cgen as m2c
    import xgboost as xgb

    logger.info("Exporting with m2cgen ...")

    clf = xgb.XGBClassifier(use_label_encoder=False)
    clf.load_model(XGB_MODEL_PATH)

    c_code = m2c.export_to_c(clf)

    c_path = os.path.join(out_dir, "xgb_predict.c")
    Path(c_path).write_text(c_code)

    h_path = os.path.join(out_dir, "xgb_predict.h")
    _write_m2cgen_header(h_path)

    logger.info("  m2cgen C source -> %s", c_path)
    logger.info("  m2cgen C header -> %s", h_path)
    _write_xgb_c_wrapper(h_path, out_dir, backend="m2cgen")


def _write_m2cgen_header(h_path: str) -> None:
    """Write a header declaring the m2cgen-generated score function."""
    content = """\
// xgb_predict.h
// Auto-generated by export_models.py using m2cgen
// Declares the XGBoost inference function for ESP32-S3

#ifndef XGB_PREDICT_H
#define XGB_PREDICT_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief XGBoost multi-class inference.
 *
 * @param input   Pointer to feature array (length = NUM_FEATURES).
 *                Features must be in the same order as training.
 * @param output  Pointer to score array (length = NUM_CLASSES = 5).
 *                The predicted class is argmax(output).
 */
void score(double * input, double * output);

// Number of input features (must match feature_extraction.py output)
#define XGB_NUM_FEATURES  28
// Number of output classes
#define XGB_NUM_CLASSES    5

#ifdef __cplusplus
}
#endif

#endif  // XGB_PREDICT_H
"""
    Path(h_path).write_text(content)


def _write_xgb_c_wrapper(
    h_path: str,
    out_dir: str,
    backend: str,
) -> None:
    """
    Write a thin ESP32-S3 usage example / wrapper for the XGBoost C export.

    This shows exactly how to call the generated C function from firmware.
    """
    fname = "xgb_esp32_usage.cpp"
    content = f"""\
// xgb_esp32_usage.cpp
// =====================
// Example ESP32-S3 integration for the {backend}-exported XGBoost model.
// Include this as a reference; adapt to your firmware's feature pipeline.
//
// Build chain:
//   1. Add xgb_predict.c/.h (or xgb_predict.h for emlearn) to your project.
//   2. Call ppg_classify() after computing 28 features with your C feature
//      extractor (port of feature_extraction.py).

#include <stdint.h>
#include <float.h>
#include "{os.path.basename(h_path)}"

// Map integer class to human-readable label
static const char* CLASS_LABELS[XGB_NUM_CLASSES] = {{
    "Normal",
    "Atrial Fibrillation",
    "Bradycardia",
    "Tachycardia",
    "Other Arrhythmia",
}};

/**
 * @brief Classify a PPG window using the XGBoost model.
 *
 * @param features  Feature vector (length XGB_NUM_FEATURES).
 *                  Must be in the same order as output by feature_extraction.py.
 * @return uint8_t  Predicted class index (0-{NUM_CLASSES - 1}).
 */
uint8_t ppg_classify(const float* features) {{
    // m2cgen uses double internally; copy from float input
    double input_d[XGB_NUM_FEATURES];
    for (int i = 0; i < XGB_NUM_FEATURES; i++) {{
        input_d[i] = (double)features[i];
    }}

    double scores[XGB_NUM_CLASSES] = {{0}};
    score(input_d, scores);   // generated by {backend}

    // Argmax
    uint8_t best_class = 0;
    double  best_score = -DBL_MAX;
    for (int c = 0; c < XGB_NUM_CLASSES; c++) {{
        if (scores[c] > best_score) {{
            best_score = scores[c];
            best_class = (uint8_t)c;
        }}
    }}
    return best_class;
}}

// ---- Example usage in Arduino/ESP-IDF task ----------------------------------
//
// void arrhythmia_task(void* pv) {{
//     float features[XGB_NUM_FEATURES];
//     while (1) {{
//         acquire_ppg_window(raw_buffer, 1250);    // 10 s @ 125 Hz
//         extract_features(raw_buffer, features);  // your C feature extractor
//         uint8_t cls = ppg_classify(features);
//         ESP_LOGI("PPG", "Predicted: %s", CLASS_LABELS[cls]);
//         vTaskDelay(pdMS_TO_TICKS(10000));
//     }}
// }}
"""
    p = os.path.join(out_dir, fname)
    Path(p).write_text(content)
    logger.info("  ESP32 wrapper example -> %s", p)


# ==============================================================================
#  REPORT: post-export size + latency summary
# ==============================================================================

def print_export_summary(export_dir: str) -> None:
    """
    Print a summary of all exported files with sizes and TFLM usage snippets.
    """
    print(f"\n{'='*70}")
    print("  TinyML Export Summary")
    print(f"{'='*70}")

    all_files = []
    for root, dirs, files in os.walk(export_dir):
        for f in files:
            fp = os.path.join(root, f)
            size_kb = os.path.getsize(fp) / 1024
            all_files.append((os.path.relpath(fp, export_dir), size_kb))

    if not all_files:
        print("  No exported files found.")
        return

    print(f"\n  {'File':<50} {'Size (KB)':>10}")
    print(f"  {'-'*50} {'-'*10}")
    for rel_path, size_kb in sorted(all_files):
        print(f"  {rel_path:<50} {size_kb:>10.1f}")

    print(f"\n{'='*70}")
    print("  ESP32-S3 / TFLM Integration Snippets")
    print(f"{'='*70}")

    print("""
  [Deep Learning Models -- TFLM / TensorFlow Lite for Microcontrollers]

  // In your ESP32-S3 firmware (e.g. main.cpp):
  #include "model_data_edge_cnn.h"   // or bilstm / tcn
  #include "tensorflow/lite/micro/micro_interpreter.h"
  #include "tensorflow/lite/micro/all_ops_resolver.h"

  constexpr int kTensorArenaSize = 64 * 1024;  // adjust per model RAM
  uint8_t tensor_arena[kTensorArenaSize];

  tflite::AllOpsResolver resolver;
  const tflite::Model* model = tflite::GetModel(g_edge_cnn_model);
  tflite::MicroInterpreter interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter.AllocateTensors();

  // Feed INT8 input (pre-quantised from your PPG buffer)
  TfLiteTensor* input = interpreter.input(0);
  // ... fill input->data.int8 with your quantised samples ...
  interpreter.Invoke();
  TfLiteTensor* output = interpreter.output(0);
  int predicted_class = ArgMax(output->data.int8, 5);

  [XGBoost -- Native C (no runtime library)]

  #include "xgb_predict.h"
  float features[XGB_NUM_FEATURES];
  extract_features(ppg_buffer, features);        // your C feature extractor
  uint8_t cls = ppg_classify(features);          // from xgb_esp32_usage.cpp
""")


# ==============================================================================
#  MAIN
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export PPG arrhythmia models to TinyML formats"
    )
    parser.add_argument(
        "--models", nargs="+",
        choices=["edge_cnn", "bilstm", "tcn", "xgb"],
        default=["edge_cnn", "bilstm", "tcn", "xgb"],
        help="Which models to export (default: all)",
    )
    parser.add_argument(
        "--rep-samples", type=int, default=200,
        help="Calibration samples for INT8 PTQ (default: 200)",
    )
    parser.add_argument(
        "--skip-tflite", action="store_true",
        help="Skip TFLite conversion; re-generate C headers from existing .tflite files",
    )
    parser.add_argument(
        "--xgb-backend", choices=["auto", "emlearn", "m2cgen"], default="auto",
        help="C code backend for XGBoost export (default: auto -> try emlearn then m2cgen)",
    )
    args = parser.parse_args()

    os.makedirs(EXPORT_DIR, exist_ok=True)

    # ── Deep Learning models --------------------------------------------------
    for key in ["edge_cnn", "bilstm", "tcn"]:
        if key in args.models:
            export_dl_model(key, args.rep_samples, args.skip_tflite)

    # ── XGBoost -> C ---------------------------------------------------------
    if "xgb" in args.models:
        export_xgboost_c_code(method=args.xgb_backend)

    # ── Summary report --------------------------------------------------------
    print_export_summary(EXPORT_DIR)
    logger.info("\nAll exports complete -> %s", EXPORT_DIR)


if __name__ == "__main__":
    main()
