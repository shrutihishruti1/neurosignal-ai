"""Turn window probabilities into seizure alarms and score them at event level."""
import numpy as np
import pandas as pd
from src import config


def detect_events(file_pred, threshold, min_consecutive=2):
    """Alarms for one file: runs of >= min_consecutive windows with prob >= threshold.

    Returns a list of (start_s, end_s, alarm_s) tuples, where alarm_s is the time
    the alarm would fire (end of the min_consecutive-th window).
    """
    file_pred = file_pred.sort_values("start")
    flags = file_pred["prob"].values >= threshold
    starts = file_pred["start"].values

    events, run_start = [], None
    # Append a False so a run that lasts to the end of the file is closed too
    for i, flag in enumerate(np.append(flags, False)):
        if flag and run_start is None:
            run_start = i                                  # a run begins
        elif not flag and run_start is not None:
            if i - run_start >= min_consecutive:           # long enough to alarm
                start_s = starts[run_start]
                end_s = starts[i - 1] + config.WINDOW_SEC
                alarm_s = start_s + min_consecutive * config.WINDOW_SEC
                events.append((start_s, end_s, alarm_s))
            run_start = None                               # the run ends
    return events


def overlaps(event, seizure):
    """True if an alarm (start, end, alarm) overlaps a seizure row in time."""
    return event[0] < seizure["end"] and event[1] > seizure["start"]


def evaluate_events(pred, labels, threshold, min_consecutive=2):
    """Event-level sensitivity, false alarms per hour and median detection latency."""
    detected, false_alarms, latencies = 0, 0, []
    per_patient = {}
    total_hours = len(pred) * config.WINDOW_SEC / 3600

    for file_name, file_pred in pred.groupby("file"):
        events = detect_events(file_pred, threshold, min_consecutive)
        seizures = labels[labels["file"] == file_name]
        patient = file_pred["patient"].iloc[0]
        stats = per_patient.setdefault(patient, {"seizures": 0, "detected": 0, "false_alarms": 0})

        # A true seizure is detected if any alarm overlaps it
        for _, sz in seizures.iterrows():
            hits = [e for e in events if overlaps(e, sz)]
            stats["seizures"] += 1
            if hits:
                detected += 1
                stats["detected"] += 1
                latencies.append(min(e[2] for e in hits) - sz["start"])

        # An alarm that overlaps no seizure is a false alarm
        for e in events:
            if not any(overlaps(e, sz) for _, sz in seizures.iterrows()):
                false_alarms += 1
                stats["false_alarms"] += 1

    summary = {
        "threshold": threshold,
        "seizures_detected": f"{detected}/{len(labels)}",
        "sensitivity": round(detected / len(labels), 3),
        "false_alarms": false_alarms,
        "false_alarms_per_hour": round(false_alarms / total_hours, 2),
        "median_latency_s": float(np.median(latencies)) if latencies else None,
    }
    return summary, pd.DataFrame(per_patient).T
def seizure_outcomes(pred, labels, threshold, min_consecutive=2):
    """One row per true seizure: was it detected, and how confident was the model?"""
    rows = []
    for _, sz in labels.iterrows():
        file_pred = pred[pred["file"] == sz["file"]]
        events = detect_events(file_pred, threshold, min_consecutive)
        # Windows that overlap this seizure
        during = file_pred[(file_pred["start"] < sz["end"]) &
                           (file_pred["start"] + config.WINDOW_SEC > sz["start"])]
        rows.append({
            "patient": sz["patient"],
            "file": sz["file"],
            "duration_s": sz["duration"],
            "detected": any(overlaps(e, sz) for e in events),
            "max_prob": round(during["prob"].max(), 3),   # model's highest confidence
        })
    return pd.DataFrame(rows)


def false_alarm_list(pred, labels, threshold, min_consecutive=2):
    """One row per false alarm (an alarm that overlaps no seizure)."""
    rows = []
    for file_name, file_pred in pred.groupby("file"):
        seizures = labels[labels["file"] == file_name]
        for start_s, end_s, alarm_s in detect_events(file_pred, threshold, min_consecutive):
            event = (start_s, end_s, alarm_s)
            if not any(overlaps(event, sz) for _, sz in seizures.iterrows()):
                rows.append({
                    "patient": file_pred["patient"].iloc[0],
                    "file": file_name,
                    "start_s": start_s,
                    "duration_s": end_s - start_s,
                })
    return pd.DataFrame(rows)