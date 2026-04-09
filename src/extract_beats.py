"""
Extract individual heartbeat windows from MIT-BIH recordings.

Reads raw .dat/.atr files, segments beats around R-peaks,
maps annotations to binary labels, and balances the classes.

Usage:
    python -m src.extract_beats
"""

import numpy as np
import wfdb
from tqdm import tqdm

from src.config import (
    DATA_RAW, DATA_PROCESSED,
    SAMPLING_RATE, WINDOW_BEFORE, WINDOW_AFTER, BEAT_LENGTH,
    NORMAL_LABELS, ABNORMAL_LABELS,
    LABEL_NORMAL, LABEL_ABNORMAL,
    ALL_RECORDS, RANDOM_SEED,
)


def extract_beats_from_record(record_name: str, data_dir=DATA_RAW):
    """Extract beat windows and labels from a single MIT-BIH record.

    Parameters
    ----------
    record_name : str
        Record identifier (e.g. '100').
    data_dir : Path
        Directory containing the raw .dat/.hea/.atr files.

    Returns
    -------
    beats : list[np.ndarray]
        Each element is a 1-D array of length BEAT_LENGTH.
    labels : list[int]
        Corresponding binary label (0 = Normal, 1 = Abnormal).
    """
    record_path = str(data_dir / record_name)

    # Read signal (channel 0 — MLII lead) and annotations
    record = wfdb.rdrecord(record_path, channels=[0])
    annotation = wfdb.rdann(record_path, 'atr')

    signal = record.p_signal.flatten()  # shape (n_samples,)
    sig_len = len(signal)

    beats, labels = [], []

    for idx, (sample, symbol) in enumerate(
        zip(annotation.sample, annotation.symbol)
    ):
        # Determine label
        if symbol in NORMAL_LABELS:
            label = LABEL_NORMAL
        elif symbol in ABNORMAL_LABELS:
            label = LABEL_ABNORMAL
        else:
            continue  # skip non-beat annotations (+, ~, |, etc.)

        # Window boundaries
        start = sample - WINDOW_BEFORE
        end = sample + WINDOW_AFTER

        # Skip beats too close to edges
        if start < 0 or end > sig_len:
            continue

        beat = signal[start:end].astype(np.float32)

        # Sanity check
        if len(beat) != BEAT_LENGTH:
            continue

        beats.append(beat)
        labels.append(label)

    return beats, labels


def balance_classes(beats, labels, seed=RANDOM_SEED):
    """Undersample the majority class to achieve a 1:1 ratio.

    Parameters
    ----------
    beats : np.ndarray, shape (N, BEAT_LENGTH)
    labels : np.ndarray, shape (N,)
    seed : int

    Returns
    -------
    beats_balanced : np.ndarray
    labels_balanced : np.ndarray
    """
    rng = np.random.RandomState(seed)

    normal_idx = np.where(labels == LABEL_NORMAL)[0]
    abnormal_idx = np.where(labels == LABEL_ABNORMAL)[0]

    n_normal = len(normal_idx)
    n_abnormal = len(abnormal_idx)
    minority_count = min(n_normal, n_abnormal)

    print(f"  Before balancing  -> Normal: {n_normal:,}  |  Abnormal: {n_abnormal:,}")
    print(f"  Undersampling to  -> {minority_count:,} per class")

    # Undersample majority
    if n_normal > n_abnormal:
        chosen = rng.choice(normal_idx, size=minority_count, replace=False)
        keep_idx = np.concatenate([chosen, abnormal_idx])
    else:
        chosen = rng.choice(abnormal_idx, size=minority_count, replace=False)
        keep_idx = np.concatenate([normal_idx, chosen])

    rng.shuffle(keep_idx)
    return beats[keep_idx], labels[keep_idx]


def main():
    all_beats, all_labels = [], []

    print("=" * 60)
    print("  Phase 1 - Extracting heartbeats from MIT-BIH records")
    print("=" * 60)

    for rec in tqdm(ALL_RECORDS, desc="Processing records"):
        beats, labels = extract_beats_from_record(rec)
        all_beats.extend(beats)
        all_labels.extend(labels)

    beats_arr = np.array(all_beats, dtype=np.float32)
    labels_arr = np.array(all_labels, dtype=np.int64)

    print(f"\n  Total beats extracted: {len(beats_arr):,}")

    # Balance classes
    beats_arr, labels_arr = balance_classes(beats_arr, labels_arr)

    # Save
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    np.save(DATA_PROCESSED / "beats.npy", beats_arr)
    np.save(DATA_PROCESSED / "labels.npy", labels_arr)

    print(f"\n  [OK] Saved {len(beats_arr):,} beats to {DATA_PROCESSED}")
    print(f"       beats.npy  shape: {beats_arr.shape}")
    print(f"       labels.npy shape: {labels_arr.shape}")
    print(f"       Normal: {(labels_arr == 0).sum():,}  |  Abnormal: {(labels_arr == 1).sum():,}")


if __name__ == "__main__":
    main()
