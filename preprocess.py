"""
preprocess.py
-------------
Downloads a subset of the MIT-BIH Arrhythmia Database from PhysioNet,
extracts individual heartbeat segments, and saves them as NumPy arrays
ready for training.

Arrhythmia classes (AAMI standard – same grouping as the paper):
  N  → Normal / bundle-branch block beats
  S  → Supraventricular ectopic beats
  V  → Ventricular ectopic beats (PVC)
  F  → Fusion beats
  Q  → Unknown / paced beats

We keep only N, S, and V (the most clinically relevant) to keep things
manageable for a first run.
"""

import os
import numpy as np
import wfdb
from scipy.signal import butter, filtfilt
from tqdm import tqdm

# ── Configuration ────────────────────────────────────────────────────────────
DATA_DIR    = "./mitbih_data"   # raw .dat/.hea files go here
OUTPUT_DIR  = "./processed"    # preprocessed numpy arrays go here
WINDOW_BEFORE = 90             # samples before each R-peak
WINDOW_AFTER  = 110            # samples after  each R-peak
FS = 360                       # MIT-BIH sampling rate (Hz)

# MIT-BIH record IDs we want to download
RECORD_IDS = [
    100, 101, 103, 105, 106, 108, 109, 111,
    113, 114, 115, 117, 119, 121, 122,
    201, 202, 203, 205, 207, 208, 209, 210,
    212, 213, 214, 215, 217, 219, 220,
    221, 222, 228, 231, 232, 233, 234,
]

# AAMI grouping: map raw MIT-BIH beat symbols → class labels
BEAT_SYMBOL_MAP = {
    # Normal
    'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',
    # Supraventricular ectopic
    'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',
    # Ventricular ectopic
    'V': 'V', 'E': 'V',
    # Fusion
    'F': 'F',
    # Unknown / paced
    '/': 'Q', 'f': 'Q', 'Q': 'Q',
}

# Classes we actually train on (drop F and Q for now)
CLASSES = ['N', 'S', 'V']
LABEL_MAP = {c: i for i, c in enumerate(CLASSES)}

# ── Signal processing helpers ─────────────────────────────────────────────────

def bandpass_filter(signal, lowcut=0.5, highcut=50.0, fs=360, order=2):
    """Apply a Butterworth bandpass filter to remove baseline wander and HF noise."""
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, signal)


def normalize(segment):
    """Zero-mean, unit-variance normalisation applied per segment."""
    std = segment.std()
    if std < 1e-6:
        return segment - segment.mean()
    return (segment - segment.mean()) / std

# ── Download & extract ────────────────────────────────────────────────────────

def download_record(record_id):
    """Download a single MIT-BIH record from PhysioNet if not already cached."""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, str(record_id))
    if not os.path.exists(path + '.hea'):
        wfdb.dl_database('mitdb', dl_dir=DATA_DIR,
                         records=[str(record_id)], overwrite=False)


def extract_heartbeats(record_id):
    """
    Returns a list of (segment, label_int) tuples for a single MIT-BIH record.
    Uses only Lead II (channel 0).
    """
    try:
        download_record(record_id)
        record = wfdb.rdrecord(os.path.join(DATA_DIR, str(record_id)))
        ann    = wfdb.rdann(os.path.join(DATA_DIR, str(record_id)), 'atr')
    except Exception as e:
        print(f"  ⚠  Could not load record {record_id}: {e}")
        return []

    signal = record.p_signal[:, 0].astype(np.float32)
    signal = bandpass_filter(signal, fs=FS)

    r_peaks  = ann.sample
    symbols  = ann.symbol
    n_signal = len(signal)

    heartbeats = []
    for peak, sym in zip(r_peaks, symbols):
        if sym not in BEAT_SYMBOL_MAP:
            continue
        label_str = BEAT_SYMBOL_MAP[sym]
        if label_str not in LABEL_MAP:
            continue

        start = peak - WINDOW_BEFORE
        end   = peak + WINDOW_AFTER
        if start < 0 or end > n_signal:
            continue           # skip edge beats

        segment = signal[start:end]
        segment = normalize(segment)
        heartbeats.append((segment, LABEL_MAP[label_str]))

    return heartbeats

# ── Main ──────────────────────────────────────────────────────────────────────

def build_dataset():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_X, all_y = [], []
    print(f"Processing {len(RECORD_IDS)} MIT-BIH records …")
    for rec_id in tqdm(RECORD_IDS, unit="record"):
        beats = extract_heartbeats(rec_id)
        for seg, lbl in beats:
            all_X.append(seg)
            all_y.append(lbl)

    X = np.array(all_X, dtype=np.float32)   # shape (N, 200)
    y = np.array(all_y, dtype=np.int64)      # shape (N,)

    # Class distribution
    print("\nClass distribution:")
    for cls, idx in LABEL_MAP.items():
        count = (y == idx).sum()
        print(f"  {cls}: {count:>6,} samples")
    print(f"  TOTAL: {len(y):>6,} samples")

    # Save
    np.save(os.path.join(OUTPUT_DIR, "X.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, "y.npy"), y)
    print(f"\n✅ Saved to {OUTPUT_DIR}/X.npy and {OUTPUT_DIR}/y.npy")

    return X, y


if __name__ == "__main__":
    build_dataset()
