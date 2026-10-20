from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
FEATURES_DIR = ROOT / "data" / "features"
SAMPLE_DIR = ROOT / "data" / "sample"
DOCS_DIR = ROOT / "docs"

PATIENTS = ["chb01", "chb02", "chb03", "chb05"]
WINDOW_SEC = 4
L_FREQ, H_FREQ = 0.5, 40.0
DROP_CHANNELS = ["T8-P8-1", "P7-T7"]
BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
}
RANDOM_STATE = 42