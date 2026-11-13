"""
quantize.py
===========
Convert the trained Keras model to a TensorFlow Lite INT8 quantized model.

Steps:
  1. Load outputs/best_model.h5
  2. Build a representative dataset from X_val
  3. Run post-training integer quantization (weights + activations → INT8)
  4. Save outputs/model_int8.tflite
  5. Print size comparison

Usage:
    python deployment/quantize.py
    python deployment/quantize.py --num-rep-samples 500
"""

import os
import sys
import argparse
import logging
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR    = os.path.join(os.path.dirname(__file__), "..", "outputs")
DATASET_PATH  = os.path.join(OUTPUT_DIR, "dataset.npz")
MODEL_PATH    = os.path.join(OUTPUT_DIR, "best_model.h5")
TFLITE_PATH   = os.path.join(OUTPUT_DIR, "model_int8.tflite")


def representative_dataset_gen(X_val: np.ndarray, num_samples: int = 200):
    """
    Generator that yields representative input tensors for INT8 calibration.
    TFLite converter calls this to determine activation quantization ranges.
    """
    indices = np.random.choice(len(X_val), size=min(num_samples, len(X_val)), replace=False)
    for idx in indices:
        sample = X_val[idx:idx + 1].astype(np.float32)  # shape: (1, 1250, 1)
        yield [sample]


def quantize(num_rep_samples: int = 200) -> None:
    import tensorflow as tf

    # ── Load dataset for representative samples ────────────────────────────────
    if not os.path.isfile(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}\n"
            "Run preprocessing/fuse_datasets.py first."
        )
    data = np.load(DATASET_PATH, allow_pickle=True)
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

    # ── Create TFLite converter ───────────────────────────────────────────────
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Full integer quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset_gen(
        X_val, num_rep_samples
    )
    # Force INT8 for both inputs and outputs (required for TFLM)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type  = tf.int8
    converter.inference_output_type = tf.int8

    logger.info("Running INT8 post-training quantization (calibrating on %d samples) …",
                num_rep_samples)
    tflite_model = converter.convert()

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(TFLITE_PATH, "wb") as f:
        f.write(tflite_model)

    # ── Size report ───────────────────────────────────────────────────────────
    orig_size = os.path.getsize(MODEL_PATH) / 1024
    tflite_size = len(tflite_model) / 1024
    logger.info("=" * 50)
    logger.info("  Original (.h5)     : %.1f KB", orig_size)
    logger.info("  Quantized (.tflite): %.1f KB  (%.1fx smaller)",
                tflite_size, orig_size / tflite_size if tflite_size > 0 else 0)
    logger.info("  Saved → %s", TFLITE_PATH)
    logger.info("=" * 50)

    # ── Sanity-check inference ─────────────────────────────────────────────────
    _verify_tflite(TFLITE_PATH, X_val[:1])


def _verify_tflite(tflite_path: str, sample: np.ndarray) -> None:
    """Run a single inference on the quantized model to verify it works."""
    import tensorflow as tf

    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()

    inp_details = interp.get_input_details()[0]
    out_details = interp.get_output_details()[0]

    # Scale float input to INT8
    scale, zero_point = inp_details["quantization"]
    inp_int8 = (sample / scale + zero_point).astype(np.int8)

    interp.set_tensor(inp_details["index"], inp_int8)
    interp.invoke()
    output = interp.get_tensor(out_details["index"])

    # Dequantize output
    out_scale, out_zp = out_details["quantization"]
    probs = (output.astype(np.float32) - out_zp) * out_scale

    logger.info("Sanity check – output probabilities: %s  (sum=%.3f)",
                np.round(probs, 3), probs.sum())
    logger.info("TFLite INT8 model verified ✓")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-rep-samples", type=int, default=200)
    args = parser.parse_args()
    quantize(num_rep_samples=args.num_rep_samples)
