"""Project settings: paths and parameters used by every module."""
from pathlib import Path

# Project root = two levels up from this file (src/config.py -> neurosignal-ai/)
ROOT = Path(__file__).resolve().parents[1]

# Data and output folders (raw EDFs and features are git-ignored)
RAW_DIR = ROOT / "data" / "raw"
FEATURES_DIR = ROOT / "data" / "features"
SAMPLE_DIR = ROOT / "data" / "sample"
DOCS_DIR = ROOT / "docs"

# CHB-MIT patients used in this project
PATIENTS = ["chb01", "chb02", "chb03", "chb05"]

# Length of each analysis window in seconds
WINDOW_SEC = 4

# Band-pass filter cut-offs in Hz (see README: Preprocessing)
L_FREQ, H_FREQ = 0.5, 40.0

# Redundant channels: second copy of T8-P8, and P7-T7 (inverse of T7-P7)
DROP_CHANNELS = ["T8-P8-1", "P7-T7"]

# Standard EEG frequency bands in Hz, used for band-power features
BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
}

# Fixed random seed so results are reproducible
RANDOM_STATE = 42