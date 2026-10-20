"""Load EDF recordings, remove redundant channels and band-pass filter."""
import warnings
import mne
import numpy as np
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

def make_windows(raw, file_name, labels):
    """Cut a recording into non-overlapping windows and label each one.

    Returns windows (n_windows, n_channels, n_samples), labels y (0/1),
    and each window's start time in seconds.
    """
    sfreq = int(raw.info["sfreq"])                   # 256 samples per second
    win = config.WINDOW_SEC * sfreq                  # 4 s x 256 = 1024 samples

    # Signal in microvolts, channels in the fixed order from config
    data = raw.get_data(picks=config.CHANNELS) * 1e6

    # Keep whole windows only; drop the leftover samples at the end
    n_win = data.shape[1] // win
    data = data[:, : n_win * win]

    # Reshape (channels, time) -> (windows, channels, samples) without a loop
    windows = data.reshape(len(config.CHANNELS), n_win, win).transpose(1, 0, 2)

    # Start time of each window in seconds: 0, 4, 8, ...
    starts = np.arange(n_win) * config.WINDOW_SEC

    # Label 1 if at least half the window overlaps a labelled seizure
    y = np.zeros(n_win, dtype=int)
    for _, sz in labels[labels["file"] == file_name].iterrows():
        overlap_start = np.maximum(starts, sz["start"])
        overlap_end = np.minimum(starts + config.WINDOW_SEC, sz["end"])
        overlap = np.clip(overlap_end - overlap_start, 0, None)
        y[overlap >= config.WINDOW_SEC / 2] = 1
    return windows, y, starts