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

# The 21 channels used, in a fixed order, so every file gives the same feature columns
CHANNELS = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",      # left temporal chain
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",      # left parasagittal chain
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",      # right parasagittal chain
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",      # right temporal chain
    "FZ-CZ", "CZ-PZ",                         # midline
    "T7-FT9", "FT9-FT10", "FT10-T8",          # CHB-MIT extra temporal channels
]
