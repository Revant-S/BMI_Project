# ECG Arrhythmia Detection using Deep Learning

Binary classification of heartbeats as **Normal** or **Abnormal** using CWT scalograms and CNNs.

## Pipeline Overview

```
MIT-BIH ECG Signals → Beat Extraction → CWT Scalograms → CNN Classification
                         (1D)              (2D images)       (SmallNet / GoogLeNet)
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download MIT-BIH data
```bash
python -m src.download_data
```

### 3. Extract heartbeats
```bash
python -m src.extract_beats
```

### 4. Generate scalograms
```bash
python -m src.build_scalograms --workers 4
```

### 5. Train a model
```bash
# SmallNet (lightweight, ~5K params)
python -m src.train --model smallnet --epochs 50

# GoogLeNet (transfer learning, ~6.8M params)
python -m src.train --model googlenet --epochs 50
```

### 6. Evaluate
```bash
python -m src.evaluate --model smallnet
python -m src.evaluate --model googlenet
python -m src.evaluate --compare
```

## Project Structure
```
BMI/
├── data/
│   ├── raw/                  # MIT-BIH .dat, .hea, .atr files
│   ├── processed/            # NumPy arrays (beats.npy, labels.npy)
│   └── scalograms/           # 224×224 RGB PNG images
│       ├── Normal/
│       └── Abnormal/
├── notebooks/
│   └── 01_data_exploration.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py             # Central configuration
│   ├── download_data.py      # MIT-BIH downloader
│   ├── extract_beats.py      # Beat extraction & balancing
│   ├── build_scalograms.py   # CWT scalogram generation
│   ├── dataset.py            # PyTorch DataLoaders
│   ├── model.py              # SmallNet + GoogLeNet
│   ├── train.py              # Training loop
│   └── evaluate.py           # Evaluation & metrics
├── results/                  # Checkpoints, plots, metrics
├── requirements.txt
└── README.md
```

## Models

| Model     | Parameters | Architecture                          |
|-----------|-----------|---------------------------------------|
| SmallNet  | ~5K       | 4× (Conv→BN→ReLU→MaxPool) + FC head  |
| GoogLeNet | ~6.8M     | Pre-trained ImageNet, fine-tuned FC   |

## Dataset

**MIT-BIH Arrhythmia Database** — 48 half-hour dual-channel ECG recordings at 360 Hz from PhysioNet.  
Records with paced beats (102, 104, 107, 217) are excluded per AAMI standard.

## Metrics

- Accuracy, Sensitivity, Specificity, Precision, F1-Score
- ROC Curve + AUC
- Confusion Matrix
