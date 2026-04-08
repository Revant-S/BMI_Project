# 🫀 Deep Learning-Based Arrhythmia Detection from ECG Signals

A PyTorch implementation of the **Small CNN** architecture for automated cardiac arrhythmia classification from ECG signals, based on the paper:

> **"Deep-Learning-Based Arrhythmia Detection Using ECG Signals: A Comparative Study and Performance Evaluation"**
> *Diagnostics* 2023, 13(24), 3605 — [DOI: 10.3390/diagnostics13243605](https://doi.org/10.3390/diagnostics13243605)

---

## 📋 Overview

This project trains a 1D Convolutional Neural Network (CNN) to classify individual heartbeats from ECG recordings into three clinically relevant categories (AAMI standard):

| Class | Type | Description |
|-------|------|-------------|
| `N` | Normal | Normal sinus rhythm beats |
| `S` | Supraventricular | Supraventricular ectopic beats |
| `V` | Ventricular | Ventricular ectopic beats (PVCs) |

**Dataset:** [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/) (PhysioNet) — 38 half-hour ECG recordings, 80,617 heartbeat segments total.

---

## 🏗️ Model Architecture

```
Input: (batch, 1, 200)   — 200-sample heartbeat window @ 360 Hz
  │
  ├─ Conv1D(1→32, k=11) → BatchNorm → ReLU → MaxPool(2)     [200 → 100]
  ├─ Conv1D(32→64, k=9) → BatchNorm → ReLU → MaxPool(2)     [100 → 50]
  ├─ Conv1D(64→128, k=7) → BatchNorm → ReLU → MaxPool(2)    [50 → 25]
  │
  ├─ Flatten
  ├─ Dropout(0.5)
  └─ Linear → 3 classes
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download & preprocess the MIT-BIH dataset
```bash
python preprocess.py
```
This will:
- Automatically download 38 MIT-BIH records from PhysioNet (~75 MB)
- Apply a Butterworth bandpass filter (0.5–50 Hz)
- Segment signals into 200-sample windows around each R-peak
- Save `processed/X.npy` (80k+ segments) and `processed/y.npy` (labels)

### 3. Train the model
```bash
python train.py
```
Training outputs:
- `checkpoints/best_model.pth` — best model weights (by validation accuracy)
- `results/training_curves.png` — loss & accuracy per epoch
- `results/confusion_matrix.png` — test set confusion matrix

---

## 📁 Project Structure

```
.
├── preprocess.py       # Dataset download & ECG segmentation
├── model.py            # Small 1D-CNN architecture
├── train.py            # Training loop, evaluation & plotting
├── requirements.txt    # Python dependencies
└── README.md
```

---

## 📦 Requirements

- Python 3.9+
- PyTorch (MPS/CUDA/CPU supported automatically)
- See `requirements.txt` for full list

---

## 📊 Results

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **99.42%** |
| Best Validation Accuracy | 99.16% |
| Epochs | 30 |
| Batch Size | 256 |

### Per-Class Performance

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| N (Normal) | 1.00 | 1.00 | 1.00 |
| S (Supraventricular) | 0.93 | 0.94 | 0.94 |
| V (Ventricular) | 0.98 | 0.99 | 0.99 |

> Results may vary slightly across runs due to random seed effects.
