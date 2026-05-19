"""
Central configuration for the ECG Arrhythmia Detection pipeline.
All paths, hyperparameters, and label mappings live here.
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────
# Directory paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_PROCESSED = DATA_DIR / "processed"
DATA_SCALOGRAMS = DATA_DIR / "scalograms"
RESULTS_DIR = PROJECT_ROOT / "results"

# Ensure directories exist
for d in [DATA_RAW, DATA_PROCESSED, DATA_SCALOGRAMS, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# Signal parameters
# ──────────────────────────────────────────────
SAMPLING_RATE = 360        # MIT-BIH sampling rate (Hz)
WINDOW_BEFORE = 90         # Samples before R-peak
WINDOW_AFTER = 160         # Samples after R-peak
BEAT_LENGTH = WINDOW_BEFORE + WINDOW_AFTER  # 250 samples per beat

# ──────────────────────────────────────────────
# Label mapping (MIT-BIH annotation symbols → binary)
# ──────────────────────────────────────────────
# AAMI class N  —  Normal beats
NORMAL_LABELS = {'N', 'L', 'R', 'e', 'j'}

# AAMI classes S, V, F  —  Abnormal beats
ABNORMAL_LABELS = {'A', 'a', 'S', 'V', 'E', 'F', 'f', 'J', '/'}

# Class indices
LABEL_NORMAL = 0
LABEL_ABNORMAL = 1
CLASS_NAMES = ["Normal", "Abnormal"]

# ──────────────────────────────────────────────
# Records to exclude (paced-beat records per AAMI)
# ──────────────────────────────────────────────
EXCLUDE_RECORDS = ['102', '104', '107', '217']

# Full list of MIT-BIH record numbers
ALL_RECORDS = [
    '100', '101', '103', '105', '106', '108', '109',
    '111', '112', '113', '114', '115', '116', '117', '118', '119',
    '121', '122', '123', '124',
    '200', '201', '202', '203', '205', '207', '208', '209',
    '210', '212', '213', '214', '215', '219',
    '220', '221', '222', '223', '228', '230', '231', '232', '233', '234',
]

# ──────────────────────────────────────────────
# CWT / Scalogram parameters
# ──────────────────────────────────────────────
CWT_WAVELET = 'morl'                       # Morlet wavelet
CWT_SCALES_MAX = 128                       # np.arange(1, CWT_SCALES_MAX)
IMG_SIZE = 224                              # Scalogram image size (px)

# ──────────────────────────────────────────────
# Training hyperparameters
# ──────────────────────────────────────────────
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
MOMENTUM = 0.9
EPOCHS = 50
EARLY_STOP_PATIENCE = 10

# ──────────────────────────────────────────────
# Dataset split ratios
# ──────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ──────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────
RANDOM_SEED = 42
