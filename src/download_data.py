"""
Download the MIT-BIH Arrhythmia Database from PhysioNet.

Usage:
    python -m src.download_data
"""

import sys
from pathlib import Path

import wfdb

from src.config import DATA_RAW


def download_mitbih(target_dir: Path = DATA_RAW) -> None:
    """Download all MIT-BIH Arrhythmia Database records to *target_dir*.

    Skips the download if the directory already contains .dat files.
    """
    target_dir.mkdir(parents=True, exist_ok=True)

    # Check if data already exists
    existing_dat = list(target_dir.glob("*.dat"))
    if len(existing_dat) >= 40:
        print(f"[INFO] Found {len(existing_dat)} .dat files in {target_dir}. "
              "Skipping download.")
        return

    print(f"[INFO] Downloading MIT-BIH Arrhythmia Database to {target_dir} ...")
    print("       This may take a few minutes depending on your connection.\n")

    wfdb.dl_database('mitdb', str(target_dir))

    # Validate
    downloaded = list(target_dir.glob("*.dat"))
    print(f"\n[OK] Downloaded {len(downloaded)} records to {target_dir}")


def main() -> None:
    download_mitbih()


if __name__ == "__main__":
    main()
