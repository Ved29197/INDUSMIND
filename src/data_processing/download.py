"""Download the UCI AI4I 2020 dataset into data/raw/.

Run from the project root:
    python -m src.data_processing.download
"""
from pathlib import Path
from urllib.request import urlretrieve
import zipfile

URL = (
    "https://archive.ics.uci.edu/static/public/601/"
    "ai4i+2020+predictive+maintenance+dataset.zip"
)

# Project root = two folders above this file (src/data_processing/download.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
ZIP_PATH = RAW_DIR / "ai4i2020.zip"
CSV_PATH = RAW_DIR / "ai4i2020.csv"


def download_ai4i(force: bool = False) -> Path:
    """Download and unzip the dataset. Skip if already present."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if CSV_PATH.exists() and not force:
        print(f"Already downloaded: {CSV_PATH}")
        return CSV_PATH

    print("Downloading AI4I 2020 dataset from UCI...")
    urlretrieve(URL, ZIP_PATH)

    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(RAW_DIR)
    ZIP_PATH.unlink()  # remove the zip, keep the CSV

    if not CSV_PATH.exists():
        found = [p.name for p in RAW_DIR.iterdir()]
        raise FileNotFoundError(
            f"Expected {CSV_PATH.name} after unzipping. Found: {found}"
        )

    print(f"Saved: {CSV_PATH}")
    return CSV_PATH


if __name__ == "__main__":
    download_ai4i()