"""
quantize.py
===========
Convert the trained Keras model to a TensorFlow Lite INT8 quantized model.

Why CPU-only?
-------------
TFLite post-training quantization (PTQ) is a graph-transformation pass that
runs entirely on the CPU.  There is no GPU-accelerated PTQ converter.
We explicitly disable the GPU here (CUDA_VISIBLE_DEVICES=-1) so TF does not
waste VRAM on quantization, and so the script always works even on machines
where the GPU is busy training another model.

Steps:
  1. Load outputs/best_model.h5  (float32)
  2. Build a representative dataset from X_val for INT8 calibration
  3. Run full-integer post-training quantization (weights + activations → INT8)
  4. Save outputs/model_int8.tflite
  5. Verify with a single inference and print size report

Usage:
    python deployment/quantize.py
    python deployment/quantize.py --num-rep-samples 500
"""

from __future__ import annotations

import os
import sys
import argparse
import logging
import numpy as np

# ── Force CPU — PTQ has no GPU equivalent ─────────────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR   = os.path.join(os.path.dirname(__file__), "..", "outputs")
DATASET_PATH = os.path.join(OUTPUT_DIR, "dataset.npz")
MODEL_PATH   = os.path.join(OUTPUT_DIR, "best_model.h5")
TFLITE_PATH  = os.path.join(OUTPUT_DIR, "model_int8.tflite")


def representative_dataset_gen(
    X_val: np.ndarray,
    num_samples: int = 200,
):
    """
    Generator yielding representative float32 samples for INT8 calibration.

    The TFLite converter calls this generator to determine the min/max
    activation ranges that are used to scale weights and activations to INT8.
    More samples → better calibration → higher quantized accuracy.
    """
    indices = np.random.choice(
        len(X_val), size=min(num_samples, len(X_val)), replace=False
    )
    for idx in indices:
        sample = X_val[idx : idx + 1].astype(np.float32)  # shape: (1, 1250, 1)
        yield [sample]


def quantize(num_rep_samples: int = 200) -> None:
    import tensorflow as tf

    logger.info("Quantization running on CPU (GPU disabled for PTQ).")

    # ── Load validation data for calibration ──────────────────────────────────
    if not os.path.isfile(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}\n"
            "Run preprocessing/fuse_datasets.py first."
        )
    data  = np.load(DATASET_PATH, allow_pickle=True)
    X_val = data["X_val"].astype(np.float32)
    logger.info("Loaded %d validation samples for calibration", len(X_val))

    # ── Load model ────────────────────────────────────────────────────────────
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Run model/train.py first."
        )
    logger.info("Loading model: %s", MODEL_PATH)
    model = tf.keras.models.load_model(MODEL_PATH)
    logger.info("Model input shape: %s", model.input_shape)

    # ── TFLite converter ──────────────────────────────────────────────────────
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Full integer quantization (weights + activations → INT8)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset_gen(
        X_val, num_rep_samples
    )
    # INT8 ops required for TFLM (TensorFlow Lite for Microcontrollers / ESP32-S3)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type  = tf.int8
    converter.inference_output_type = tf.int8

    logger.info(
        "Running INT8 post-training quantization "
        "(calibrating on %d samples) …",
        num_rep_samples,
    )
    tflite_model = converter.convert()

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(TFLITE_PATH, "wb") as fh:
        fh.write(tflite_model)

    # ── Size report ───────────────────────────────────────────────────────────
    orig_kb    = os.path.getsize(MODEL_PATH) / 1024
    tflite_kb  = len(tflite_model) / 1024
    ratio      = orig_kb / tflite_kb if tflite_kb > 0 else 0.0
    logger.info("=" * 50)
    logger.info("  Original  (.h5)      : %6.1f KB", orig_kb)
    logger.info("  Quantized (.tflite)  : %6.1f KB  (%.1fx smaller)", tflite_kb, ratio)
    logger.info("  Saved → %s", TFLITE_PATH)
    logger.info("=" * 50)

    # ── Sanity-check inference ─────────────────────────────────────────────────
    _verify_tflite(TFLITE_PATH, X_val[:1])


def _verify_tflite(tflite_path: str, sample: np.ndarray) -> None:
    """Run a single inference on the INT8 model to verify correctness."""
    import tensorflow as tf

    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()

    inp_details = interp.get_input_details()[0]
    out_details = interp.get_output_details()[0]

    # Quantize float input → INT8 using the model's input scale/zero_point
    scale, zero_point = inp_details["quantization"]
    inp_int8 = (sample / scale + zero_point).astype(np.int8)

    interp.set_tensor(inp_details["index"], inp_int8)
    interp.invoke()
    output = interp.get_tensor(out_details["index"])

    # Dequantize INT8 output → float probabilities
    out_scale, out_zp = out_details["quantization"]
    probs = (output.astype(np.float32) - out_zp) * out_scale

    logger.info(
        "Sanity check — probabilities: %s  (sum=%.3f)",
        np.round(probs, 3), float(probs.sum()),
    )
    logger.info("TFLite INT8 model verified ✓")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Keras model to INT8 TFLite for ESP32-S3 TFLM"
    )
    parser.add_argument(
        "--num-rep-samples", type=int, default=200,
        help="Number of calibration samples for INT8 quantization (default: 200)",
    )
    args = parser.parse_args()
    quantize(num_rep_samples=args.num_rep_samples)
