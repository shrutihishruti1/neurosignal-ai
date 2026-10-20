"""NeuroSignal AI: interactive demo of patient-independent EEG seizure detection."""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make the project root importable when Streamlit runs app/app.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config
from src.demo import load_clip, demo_figure
from src.events import detect_events

# Page set-up and honest framing
st.set_page_config(page_title="NeuroSignal AI", layout="wide")
st.title("NeuroSignal AI: EEG seizure detection")
st.caption("Research prototype only, not for clinical diagnosis. "
           "Data: CHB-MIT Scalp EEG Database (PhysioNet), ODC-By 1.0.")
st.markdown(
    "Each clip comes from a patient the model **never saw during training** "
    "(leave-one-patient-out), so what you see here is the honest, patient-independent performance."
)

# Sidebar controls: which clip, and the alarm threshold
CLIPS = {
    "Detected seizure (chb05_06)": "detected_chb05_06",
    "Detected seizure (chb03_01)": "detected_chb03_01",
    "Missed seizure (chb01_03)": "missed_chb01_03",
    "False alarm (chb05_22)": "false_alarm_chb05_22",
}
choice = st.sidebar.selectbox("Choose a recording", list(CLIPS))
threshold = st.sidebar.slider("Alarm threshold", 0.1, 0.9, 0.3, 0.05)
st.sidebar.markdown(
    "**Overall results (22 seizures, 4 unseen patients)**  \n"
    "15/22 seizures detected · 8.6 false alarms/hour · 18 s median delay"
)

# Load the chosen clip and find alarms at the chosen threshold
clip = load_clip(config.SAMPLE_DIR / f"{CLIPS[choice]}.npz")
clip_end = float(clip["t0"]) + clip["eeg"].shape[1] / int(clip["sfreq"])
events = detect_events(pd.DataFrame({"start": clip["win_starts"], "prob": clip["probs"]}), threshold)
seizures_in_clip = [s for s in clip["seizures"] if s[0] < clip_end and s[1] > float(clip["t0"])]

# Three headline numbers above the figure
col1, col2, col3 = st.columns(3)
col1.metric("Labelled seizures in clip", len(seizures_in_clip))
col2.metric("Model alarms", len(events))
col3.metric("Peak seizure probability", f"{clip['probs'].max():.2f}")

# The four-panel figure
st.pyplot(demo_figure(clip, threshold=threshold))

with st.expander("How to read this figure"):
    st.markdown(
        "1. **EEG**: 6 of 21 channels; the red band is the clinician-labelled seizure.  \n"
        "2. **Spectrogram**: power per frequency over time; seizures brighten the low frequencies.  \n"
        "3. **Probability**: the model's seizure probability every 4 s; grey = needs expert review.  \n"
        "4. **Events**: red = labelled seizure, green = model alarms (2+ windows above threshold)."
    )