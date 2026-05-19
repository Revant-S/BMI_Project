"""
Build 2-D scalogram images from 1-D heartbeat arrays using CWT.

Applies the Continuous Wavelet Transform (Morlet wavelet) to each beat
and saves the resulting scalogram as a 224×224 PNG in class sub-directories.

Usage:
    python -m src.build_scalograms
    python -m src.build_scalograms --workers 4
"""

import argparse
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pywt
from PIL import Image
from tqdm import tqdm

from src.config import (
    DATA_PROCESSED, DATA_SCALOGRAMS,
    CWT_WAVELET, CWT_SCALES_MAX, IMG_SIZE,
    CLASS_NAMES, SAMPLING_RATE,
)


def beat_to_scalogram(beat: np.ndarray,
                      scales: np.ndarray,
                      wavelet: str = CWT_WAVELET) -> np.ndarray:
    """Convert a 1-D beat signal to a 2-D scalogram (uint8 image).

    Parameters
    ----------
    beat : np.ndarray, shape (BEAT_LENGTH,)
    scales : np.ndarray
    wavelet : str

    Returns
    -------
    img : np.ndarray, shape (IMG_SIZE, IMG_SIZE), dtype uint8
    """
    # Perform CWT
    coefficients, _ = pywt.cwt(beat, scales, wavelet,
                               sampling_period=1.0 / SAMPLING_RATE)
    scalogram = np.abs(coefficients)

    # Normalize to 0-255
    smin, smax = scalogram.min(), scalogram.max()
    if smax - smin > 0:
        scalogram = (scalogram - smin) / (smax - smin) * 255.0
    else:
        scalogram = np.zeros_like(scalogram)

    scalogram = scalogram.astype(np.uint8)

    # Resize to IMG_SIZE × IMG_SIZE
    img = Image.fromarray(scalogram, mode='L')
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)

    return np.array(img)


def _process_one(args):
    """Worker function for multiprocessing."""
    idx, beat, label, scales, output_dir = args
    class_name = CLASS_NAMES[label]
    out_path = output_dir / class_name / f"beat_{idx:06d}.png"

    if out_path.exists():
        return  # skip already-generated

    img_arr = beat_to_scalogram(beat, scales)
    img = Image.fromarray(img_arr, mode='L')

    # Convert to RGB (3-channel) for compatibility with pre-trained models
    img_rgb = img.convert('RGB')
    img_rgb.save(out_path)


def main():
    parser = argparse.ArgumentParser(description="Generate CWT scalograms")
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of parallel workers (default: 1)')
    args = parser.parse_args()

    print("=" * 60)
    print("  Phase 2 - Generating CWT scalogram images")
    print("=" * 60)

    # Load data
    beats = np.load(DATA_PROCESSED / "beats.npy")
    labels = np.load(DATA_PROCESSED / "labels.npy")
    print(f"  Loaded {len(beats):,} beats  ({beats.shape})")

    # Ensure output directories
    for cls in CLASS_NAMES:
        (DATA_SCALOGRAMS / cls).mkdir(parents=True, exist_ok=True)

    scales = np.arange(1, CWT_SCALES_MAX)

    # Prepare task list
    tasks = [
        (i, beats[i], labels[i], scales, DATA_SCALOGRAMS)
        for i in range(len(beats))
    ]

    n_workers = min(args.workers, cpu_count())
    print(f"  Wavelet: {CWT_WAVELET}  |  Scales: 1-{CWT_SCALES_MAX - 1}")
    print(f"  Output size: {IMG_SIZE}x{IMG_SIZE} RGB PNG")
    print(f"  Workers: {n_workers}\n")

    if n_workers <= 1:
        for t in tqdm(tasks, desc="Generating scalograms"):
            _process_one(t)
    else:
        with Pool(n_workers) as pool:
            list(tqdm(
                pool.imap_unordered(_process_one, tasks),
                total=len(tasks),
                desc="Generating scalograms",
            ))

    # Summary
    for cls in CLASS_NAMES:
        count = len(list((DATA_SCALOGRAMS / cls).glob("*.png")))
        print(f"  {cls:>10s}: {count:,} images")

    print("\n  [OK] Scalograms saved to", DATA_SCALOGRAMS)


if __name__ == "__main__":
    main()
