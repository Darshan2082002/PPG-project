"""
visualize_signals.py
====================
Loads the first available PTT CSV and plots:
  Panel 1 – Raw PPG (pleth_2)
  Panel 2 – Accelerometer magnitude (motion reference)
  Panel 3 – Filtered PPG (after LMS MA removal + bandpass)

Saves output to outputs/signal_visualization.png
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal as sp_signal

from preprocessing.motion_artifact   import remove_motion_artifacts, acc_magnitude_reference
from preprocessing.signal_processing import bandpass_filter, resample_to_target, TARGET_FS

PTT_CSV_DIR = os.path.join(
    os.path.dirname(__file__), "..",
    "Pulse Transit Time PPG Dataset",
    "physionet.org", "files", "pulse-transit-time-ppg", "1.1.0", "csv"
)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load first CSV ─────────────────────────────────────────────────────────────
csv_files = sorted(glob.glob(os.path.join(PTT_CSV_DIR, "s*.csv")))
assert csv_files, "No PTT CSV files found!"
csv_path = csv_files[0]
print(f"Loading: {os.path.basename(csv_path)}")

df = pd.read_csv(csv_path, on_bad_lines="skip")
print(f"Columns: {list(df.columns)}")

# Extract 15-second window (500Hz × 15s = 7500 samples) for display
DISPLAY_SEC = 15
N_DISPLAY   = 500 * DISPLAY_SEC

ppg_raw = df["pleth_2"].values[:N_DISPLAY].astype(np.float32)
acc     = df[["a_x","a_y","a_z"]].values[:N_DISPLAY].astype(np.float32)

# Drop NaNs
valid = ~(np.isnan(ppg_raw) | np.any(np.isnan(acc), axis=1))
ppg_raw = ppg_raw[valid]
acc     = acc[valid]

# ── Downsample 500 → 125 Hz ────────────────────────────────────────────────────
factor  = 500 // TARGET_FS
ppg_ds  = ppg_raw[::factor]
acc_ds  = acc[::factor]

# ── ACC magnitude reference ────────────────────────────────────────────────────
acc_mag = acc_magnitude_reference(acc_ds)

# ── LMS adaptive filter ────────────────────────────────────────────────────────
ppg_ma_removed = remove_motion_artifacts(ppg_ds, acc_ds)

# ── Bandpass filter ────────────────────────────────────────────────────────────
ppg_filtered = bandpass_filter(ppg_ma_removed, TARGET_FS)

# ── Time axis ─────────────────────────────────────────────────────────────────
t = np.arange(len(ppg_ds)) / TARGET_FS

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
fig.suptitle(
    f"PPG Signal Processing Pipeline\n{os.path.basename(csv_path)} — Subject 1, Sitting",
    fontsize=14, fontweight="bold", y=1.01
)

# Panel 1 – Raw PPG
axes[0].plot(t, ppg_ds, color="#2196F3", linewidth=0.8, alpha=0.9)
axes[0].set_ylabel("Amplitude (NU)", fontsize=11)
axes[0].set_title("① Raw PPG (pleth_2 @ 125 Hz)", fontsize=11, loc="left")
axes[0].grid(True, alpha=0.3)
axes[0].set_facecolor("#f8f9fa")

# Panel 2 – ACC magnitude (motion reference)
axes[1].plot(t, acc_mag, color="#FF5722", linewidth=0.8, alpha=0.9)
axes[1].set_ylabel("Magnitude (g)", fontsize=11)
axes[1].set_title("② Accelerometer Motion Reference |√(ax²+ay²+az²)|", fontsize=11, loc="left")
axes[1].grid(True, alpha=0.3)
axes[1].set_facecolor("#f8f9fa")

# Panel 3 – Filtered PPG
axes[2].plot(t, ppg_filtered, color="#4CAF50", linewidth=0.9)
axes[2].set_ylabel("Amplitude (NU)", fontsize=11)
axes[2].set_xlabel("Time (s)", fontsize=11)
axes[2].set_title("③ Clean PPG (after LMS MA removal + 0.5–8 Hz bandpass)", fontsize=11, loc="left")
axes[2].grid(True, alpha=0.3)
axes[2].set_facecolor("#f8f9fa")

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "signal_visualization.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nPlot saved -> {out_path}")

# ── Stats ────────────────────────────────────────────────────────────────────
print(f"\nSignal statistics (@ {TARGET_FS} Hz, {len(ppg_ds)/TARGET_FS:.1f}s):")
print(f"  Raw PPG      – mean={ppg_ds.mean():.1f}  std={ppg_ds.std():.1f}")
print(f"  ACC magnitude– mean={acc_mag.mean():.4f}  std={acc_mag.std():.4f}")
print(f"  Filtered PPG – mean={ppg_filtered.mean():.4f}  std={ppg_filtered.std():.4f}")
