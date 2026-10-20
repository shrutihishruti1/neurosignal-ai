"""Export small demo clips and draw the waveform / spectrogram / timeline figure."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from src import config
from src.preprocess import load_raw
from src.events import detect_events

# 6 channels from different head regions, enough to see the pattern
DEMO_CHANNELS = ["FP1-F7", "T7-P7", "C3-P3", "FT9-FT10", "T8-P8", "CZ-PZ"]


def export_clip(name, file_name, t0, duration, pred, labels):
    """Save a short EEG clip and its out-of-fold probabilities to data/sample/."""
    raw = load_raw(file_name)
    eeg = raw.copy().crop(t0, t0 + duration).get_data(picks=DEMO_CHANNELS) * 1e6

    # Window probabilities inside the clip (from the model that never saw this patient)
    p = pred[(pred["file"] == file_name) &
             (pred["start"] >= t0) & (pred["start"] < t0 + duration)].sort_values("start")
    seizures = labels.loc[labels["file"] == file_name, ["start", "end"]].values

    config.SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        config.SAMPLE_DIR / f"{name}.npz",
        eeg=eeg.astype(np.float32), channels=np.array(DEMO_CHANNELS),
        sfreq=256, t0=t0, source_file=file_name,
        win_starts=p["start"].values, probs=p["prob"].values, seizures=seizures,
    )


def load_clip(path):
    """Load a saved clip into a plain dictionary."""
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def demo_figure(clip, threshold=0.3, review=(0.3, 0.7)):
    """Four stacked panels: EEG, spectrogram, probability, and event tracks."""
    eeg, sfreq, t0 = clip["eeg"], int(clip["sfreq"]), float(clip["t0"])
    t = t0 + np.arange(eeg.shape[1]) / sfreq                  # time of every sample (s)
    starts, probs = clip["win_starts"], clip["probs"]
    events = detect_events(pd.DataFrame({"start": starts, "prob": probs}), threshold)

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True,
                             gridspec_kw={"height_ratios": [3, 2, 2, 0.8]})

    # Panel 1: EEG waveform, channels stacked with an offset
    for i, ch in enumerate(clip["channels"]):
        axes[0].plot(t, eeg[i] - i * 500, linewidth=0.5, color="black")
        axes[0].text(t[0] - 1, -i * 500, ch, ha="right", va="center", fontsize=8)
    axes[0].set_yticks([])
    axes[0].set_title(f"EEG: {clip['source_file']} (6 of 21 channels) — red = labelled seizure")

    # Panel 2: spectrogram of one temporal channel (power per frequency over time)
    f, ts, power = spectrogram(eeg[1], fs=sfreq, nperseg=sfreq, noverlap=sfreq // 2)
    keep = f <= 40
    axes[1].pcolormesh(t0 + ts, f[keep], 10 * np.log10(power[keep] + 1e-12),
                       shading="auto", cmap="magma")
    axes[1].set_ylabel("Frequency (Hz)")
    axes[1].set_title(f"Spectrogram ({clip['channels'][1]})")

    # Red seizure shading on the two signal panels only (nothing else drawn there)
    for s_start, s_end in clip["seizures"]:
        for ax in axes[:2]:
            ax.axvspan(s_start, s_end, color="red", alpha=0.15)

    # Panel 3: probability line, threshold and review zone (grey), no other shading
    axes[2].axhspan(*review, color="lightgrey", alpha=0.6, label="Needs expert review (0.3–0.7)")
    axes[2].axhline(threshold, color="black", linestyle="--", label=f"Alarm threshold {threshold}")
    axes[2].plot(starts + config.WINDOW_SEC / 2, probs, marker=".", color="tab:blue",
                 label="Seizure probability")
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("Probability")
    axes[2].legend(loc="upper left", fontsize=8)

    # Panel 4: two separate rows so colours never overlap
    seizure_bars = [(s, e - s) for s, e in clip["seizures"]]       # (start, width)
    alarm_bars = [(s, e - s) for s, e, _ in events]
    axes[3].broken_barh(seizure_bars, (1.1, 0.8), color="red")
    axes[3].broken_barh(alarm_bars, (0.1, 0.8), color="green")
    axes[3].set_yticks([0.5, 1.5], ["Model alarms", "Labelled seizure"])
    axes[3].set_ylim(0, 2)
    axes[3].set_xlabel("Time in recording (s)")

    axes[0].set_xlim(t[0], t[-1])
    fig.tight_layout()
    return fig