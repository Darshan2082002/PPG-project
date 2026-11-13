# PPG Arrhythmia Detection

End-to-end TinyML pipeline for detecting cardiac arrhythmias from PPG signals, deployed on an **ESP32-S3** using TensorFlow Lite for Microcontrollers (TFLM).

```
Data (MIMIC-IV + PTT) → Preprocessing → DS-CNN+LSTM Model → INT8 TFLite → ESP32-S3
```

---

## Project Structure

```
PPG project using antigravity/
├── MIMIC_2/                          ← MIMIC-IV Waveform Dataset (WFDB)
├── Pulse Transit Time PPG Dataset/   ← PTT PPG Dataset (CSV + WFDB)
│
├── preprocessing/
│   ├── load_mimic.py         – Load MIMIC-IV segments + rhythm labels
│   ├── load_ptt.py           – Load PTT CSVs (pleth_2 + accelerometer)
│   ├── motion_artifact.py    – LMS adaptive filter for MA removal
│   ├── signal_processing.py  – Bandpass, resample, segment, normalize
│   ├── label_mapping.py      – Rhythm string → integer class
│   └── fuse_datasets.py      – Master orchestrator → outputs/dataset.npz
│
├── model/
│   ├── architecture.py  – Depthwise-Separable 1D-CNN + LSTM (~180K params)
│   ├── train.py         – Training script with callbacks
│   └── evaluate.py      – Metrics + confusion matrix
│
├── deployment/
│   ├── quantize.py            – Float32 → INT8 TFLite (PTQ)
│   └── convert_to_c_array.py  – TFLite → C byte array (model.h)
│
├── firmware/
│   ├── src/
│   │   ├── main.cpp   – TFLM inference firmware (Arduino)
│   │   └── model.h    – Auto-generated C byte array
│   └── platformio.ini – PlatformIO build config for ESP32-S3
│
├── outputs/                    – Auto-created during pipeline
│   ├── dataset.npz             – Pre-processed windows
│   ├── best_model.h5           – Best Keras checkpoint
│   ├── final_model.h5          – Final epoch weights
│   ├── model_int8.tflite       – Quantized TFLite model
│   ├── training_history.png    – Loss/accuracy curves
│   ├── confusion_matrix.png    – Evaluation confusion matrix
│   └── classification_report.txt
│
├── requirements.txt    – Python dependencies
├── setup_env.bat       – One-click venv setup (Windows)
└── README.md
```

---

## Label Classes

| ID | Class | Source |
|----|-------|--------|
| 0 | Normal Sinus Rhythm | MIMIC-IV + all PTT data |
| 1 | Atrial Fibrillation | MIMIC-IV |
| 2 | Bradycardia | MIMIC-IV |
| 3 | Tachycardia | MIMIC-IV |
| 4 | Other Arrhythmia | MIMIC-IV |

---

## Quick Start

### 1. Environment Setup

```bat
cd "D:\PPG project using antigravity"
setup_env.bat
```
This creates `venv\`, installs all Python packages from `requirements.txt`, and activates the environment.

For future sessions:
```bat
venv\Scripts\activate.bat
```

### 2. Phase 1 – Data Preprocessing & Fusion

```bat
python preprocessing/fuse_datasets.py
```

**Optional flags:**
```bat
# Limit data for quick testing
python preprocessing/fuse_datasets.py --max-mimic-patients 10 --max-ptt-files 5

# Skip SMOTE (if imbalanced-learn has issues)
python preprocessing/fuse_datasets.py --no-smote
```

**Output:** `outputs/dataset.npz`  
Prints class distribution before and after SMOTE.

### 3. Phase 2 – Model Training

```bat
python model/train.py
```

**Optional flags:**
```bat
python model/train.py --epochs 50 --batch-size 64
python model/train.py --epochs 3          # quick smoke-test
```

**Outputs:**
- `outputs/best_model.h5` — best validation checkpoint  
- `outputs/final_model.h5` — final epoch weights  
- `outputs/training_history.png` — loss/accuracy curves

### 4. Phase 2b – Evaluation

```bat
python model/evaluate.py
```

**Outputs:**
- Prints Accuracy, Precision, Recall, F1 (macro average)  
- `outputs/confusion_matrix.png`  
- `outputs/classification_report.txt`

### 5. Phase 3 – TinyML Quantization

```bat
# Step 1: Convert to INT8 TFLite
python deployment/quantize.py

# Step 2: Convert to C header for firmware
python deployment/convert_to_c_array.py
```

**Outputs:**
- `outputs/model_int8.tflite` — INT8 quantized model  
- `firmware/src/model.h` — C byte array ready to compile

### 6. Phase 4 – ESP32-S3 Firmware Flash

**Prerequisites:** [PlatformIO](https://platformio.org/) (VSCode extension recommended)

```bat
cd firmware
pio run --target upload
pio device monitor --baud 115200
```

**Expected Serial output:**
```
========================================
  PPG Arrhythmia Inference – ESP32-S3
========================================

[Inference #1 | 47ms]
  Prediction : Normal                 (class 0)
  Confidence : 0.9412 (94.1%)
  All classes:
    [0] Normal                 : 0.9412
    [1] Atrial Fibrillation    : 0.0231
    [2] Bradycardia            : 0.0178
    [3] Tachycardia            : 0.0143
    [4] Other Arrhythmia       : 0.0036
```

---

## Replacing the Dummy PPG with a Real Sensor

In `firmware/src/main.cpp`, the function `fill_ppg_window()` generates a synthetic waveform. To use a real **MAX30102/MAX30101** sensor:

1. Install the sensor library in `platformio.ini`:
   ```ini
   lib_deps =
       sparkfun/SparkFun MAX3010x Pulse and Proximity Sensor Library
       https://github.com/tensorflow/tflite-micro-arduino-examples#main
   ```
2. Replace the body of `fill_ppg_window()`:
   ```cpp
   void fill_ppg_window() {
       for (int i = 0; i < kInputLength; i++) {
           while (!particleSensor.available()) particleSensor.check();
           ppg_window[i] = (float)particleSensor.getRed();
           particleSensor.nextSample();
           delayMicroseconds(8000);  // 125 Hz = 8ms per sample
       }
       // Apply min-max normalisation (same as training)
       float mn = *std::min_element(ppg_window, ppg_window + kInputLength);
       float mx = *std::max_element(ppg_window, ppg_window + kInputLength);
       float rng = (mx - mn) < 1e-6f ? 1.0f : (mx - mn);
       for (int i = 0; i < kInputLength; i++)
           ppg_window[i] = (ppg_window[i] - mn) / rng;
   }
   ```

---

## Model Architecture

```
Input  (1250, 1)  – 10 s @ 125 Hz
  │
  ├─ Conv1D(32, k=7) + BN + ReLU
  ├─ DS-Conv1D(64,  k=7) + BN + ReLU + MaxPool(4)  → (312, 64)
  ├─ DS-Conv1D(128, k=5) + BN + ReLU + MaxPool(4)  → (78,  128)
  ├─ DS-Conv1D(128, k=5) + BN + ReLU + MaxPool(4)  → (19,  128)
  ├─ LSTM(64)
  ├─ Dense(64, ReLU) + Dropout(0.3)
  └─ Dense(5, Softmax)

Parameters: ~180K  |  Float32: ~720 KB  |  INT8: ~180 KB
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `FileNotFoundError: dataset.npz` | Run `preprocessing/fuse_datasets.py` first |
| `AllocateTensors() failed` | Increase `kTensorArenaSize` in `main.cpp` |
| SMOTE fails (too few samples) | Run with `--no-smote` flag |
| `wfdb` import error | Activate venv: `venv\Scripts\activate.bat` |
| PlatformIO: flash not found | Set `upload_port = COMX` in `platformio.ini` |
| Out of flash on ESP32-S3 | Ensure `board_build.partitions = huge_app.csv` |

---

## Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| tensorflow | ≥2.13.0 | Model training + TFLite conversion |
| wfdb | ≥4.1.0 | WFDB physiological signal I/O |
| scipy | ≥1.10.0 | Signal filtering + resampling |
| scikit-learn | ≥1.3.0 | Metrics, class weights |
| imbalanced-learn | ≥0.11.0 | SMOTE oversampling |
| numpy, pandas | latest | Data manipulation |
| matplotlib, seaborn | latest | Plotting |
