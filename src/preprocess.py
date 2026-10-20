"""Load EDF recordings, remove redundant channels and band-pass filter."""
import warnings
import mne
from src import config

# Every CHB-MIT file has a duplicate T8-P8 channel; we handle it below,
# so silence MNE's repeated warning about it.
warnings.filterwarnings("ignore", message="Channel names are not unique")


def load_raw(file_name):
    """Load one EDF file, remove redundant channels and band-pass filter it."""
    # preload=True loads the signal into memory, which filtering requires
    raw = mne.io.read_raw_edf(config.RAW_DIR / file_name, preload=True, verbose=False)

    # Drop redundant channels, but only those present in this file
    to_drop = [ch for ch in config.DROP_CHANNELS if ch in raw.ch_names]
    raw.drop_channels(to_drop)

    # MNE renamed the kept duplicate to T8-P8-0; restore the clean name
    if "T8-P8-0" in raw.ch_names:
        raw.rename_channels({"T8-P8-0": "T8-P8"})

    # 0.5-40 Hz band-pass: removes slow drift and high-frequency noise (incl. the 48 Hz peak)
    raw.filter(config.L_FREQ, config.H_FREQ, verbose=False)
    return raw