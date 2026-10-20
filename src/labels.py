import re
import pandas as pd
from src import config

FILE_RE = re.compile(r"File Name:\s*(\S+\.edf)")
START_RE = re.compile(r"Seizure(?:\s+\d+)?\s+Start Time:\s*(\d+)\s*seconds")
END_RE = re.compile(r"Seizure(?:\s+\d+)?\s+End Time:\s*(\d+)\s*seconds")


def parse_summary(patient):
    """Return one row per seizure (patient, file, start, end) for one patient."""
    text = (config.RAW_DIR / f"{patient}-summary.txt").read_text()
    rows = []
    current_file = None
    start = None
    for line in text.splitlines():
        file_match = FILE_RE.search(line)
        if file_match:
            current_file = file_match.group(1)
            continue
        start_match = START_RE.search(line)
        if start_match:
            start = int(start_match.group(1))
            continue
        end_match = END_RE.search(line)
        if end_match:
            rows.append({
                "patient": patient,
                "file": current_file,
                "start": start,
                "end": int(end_match.group(1)),
            })
    return pd.DataFrame(rows)


def load_all_labels():
    """Seizure table for all patients, kept to the files we downloaded."""
    labels = pd.concat([parse_summary(p) for p in config.PATIENTS], ignore_index=True)
    downloaded = {f.name for f in config.RAW_DIR.glob("*.edf")}
    labels = labels[labels["file"].isin(downloaded)].reset_index(drop=True)
    labels["duration"] = labels["end"] - labels["start"]
    return labels