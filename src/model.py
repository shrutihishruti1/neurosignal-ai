"""Train LightGBM with patient-held-out evaluation and compute metrics."""
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from src import config
from src.features import feature_names

# Columns that identify a window (everything else is a feature)
ID_COLS = ["file", "patient", "start", "label"]


def make_model():
    """LightGBM classifier; class_weight='balanced' upweights the rare seizure class."""
    return LGBMClassifier(
        n_estimators=300,        # number of trees built in sequence
        learning_rate=0.05,      # how much each tree corrects the previous ones
        num_leaves=31,           # size of each tree
        class_weight="balanced", # seizure windows count ~65x more than normal ones
        random_state=config.RANDOM_STATE,
        verbose=-1,              # silence training logs
    )


def cross_validate(features, split="patient"):
    """Return every window with an out-of-fold seizure probability.

    split="patient": train on 3 patients, test on the 4th, rotate (honest).
    split="random": 4 random folds mixing all patients (leaky, for comparison).
    """
    X = features[feature_names()].values
    y = features["label"].values

    if split == "patient":
        folds = LeaveOneGroupOut().split(X, y, groups=features["patient"])
    else:
        folds = StratifiedKFold(n_splits=4, shuffle=True,
                                random_state=config.RANDOM_STATE).split(X, y)

    probs = np.zeros(len(y))
    for train_idx, test_idx in folds:
        model = make_model()
        model.fit(X[train_idx], y[train_idx])
        # Column 1 of predict_proba = probability of class 1 (seizure)
        probs[test_idx] = model.predict_proba(X[test_idx])[:, 1]

    pred = features[ID_COLS].copy()
    pred["prob"] = probs
    return pred


def window_metrics(pred, threshold=0.5):
    """Window-level metrics for all windows and for each held-out patient."""
    rows = []
    for name, group in [("All", pred)] + list(pred.groupby("patient")):
        y, p = group["label"], group["prob"]
        y_hat = (p >= threshold).astype(int)   # probability -> 0/1 decision
        rows.append({
            "patient": name,
            "sensitivity": recall_score(y, y_hat, zero_division=0),
            "precision": precision_score(y, y_hat, zero_division=0),
            "f1": f1_score(y, y_hat, zero_division=0),
            "auroc": roc_auc_score(y, p),
        })
    return pd.DataFrame(rows).round(3)


def fit_final_model(features):
    """Train one model on all patients (used for the demo and feature importance)."""
    model = make_model()
    model.fit(features[feature_names()].values, features["label"].values)
    return model