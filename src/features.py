"""Band-power and line-length features for every EEG window."""
import numpy as np
import pandas as pd
from scipy.signal import welch
from src import config
from src.preprocess import load_raw, make_windows


def window_features(windows, sfreq=256):
    """Return an array (n_windows, 105): 4 band powers + line length per channel."""
    # Welch power spectrum for every window and channel at once (1 Hz resolution)
    freqs, psd = welch(windows, fs=sfreq, nperseg=sfreq, axis=-1)  # psd: (win, ch, freq)

    feats = []
    # Mean power inside each EEG band, on a log scale
    for low, high in config.BANDS.values():
        in_band = (freqs >= low) & (freqs < high)
        feats.append(np.log10(psd[:, :, in_band].mean(axis=-1) + 1e-12))

    # Line length: total absolute change between neighbouring samples (log scale)
    line_length = np.abs(np.diff(windows, axis=-1)).sum(axis=-1)
    feats.append(np.log10(line_length + 1e-12))

    # Join the five (win, 21) blocks side by side -> (win, 105)
    return np.concatenate(feats, axis=1)


def feature_names():
    """Column names in the same order as window_features builds them."""
    names = [f"{band}_{ch}" for band in config.BANDS for ch in config.CHANNELS]
    names += [f"linelen_{ch}" for ch in config.CHANNELS]
    return names


def build_feature_table(labels):
    """Compute features for all downloaded files and save one table to disk."""
    tables = []
    for f in sorted(config.RAW_DIR.glob("*.edf")):
        # One file at a time keeps memory low
        raw = load_raw(f.name)
        windows, y, starts = make_windows(raw, f.name, labels)

        df = pd.DataFrame(window_features(windows), columns=feature_names())
        # Identifier columns first: which file/patient, when, and the label
        df.insert(0, "file", f.name)
        df.insert(1, "patient", f.name.split("_")[0])
        df.insert(2, "start", starts)
        df.insert(3, "label", y)
        tables.append(df)
        print(f"{f.name}: {len(df)} windows, {y.sum()} seizure")

    table = pd.concat(tables, ignore_index=True)

    # Save so later steps load in seconds instead of recomputing
    config.FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_pickle(config.FEATURES_DIR / "features.pkl")
    return table