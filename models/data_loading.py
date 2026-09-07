"""
data_loading.py

Shared EEG loading utilities for the BCI Competition IV 2a dataset.

IMPORTANT — event code handling:
MNE assigns numeric event codes per file, based on which annotation
description strings are present in that specific recording and in what
order. The same description (e.g. "769" = Left Hand) is NOT guaranteed
to get the same numeric code across different subjects' files. This
was confirmed directly for subject A04T (see notebooks/debug/
A04T_investigation.ipynb), whose file has fewer distinct annotation
types than most other subjects, shifting its true motor-imagery codes
down by two positions.

All loading functions in this module therefore look codes up by their
description string, never assume a fixed numeric code, and translate
to a fixed, consistent OUTPUT label defined in DESC_TO_LABEL below.
"""

import os
import numpy as np
import mne
from scipy.io import loadmat

# Fixed OUTPUT labels used across this project.
# 7 = Left Hand, 8 = Right Hand, 9 = Foot, 10 = Tongue
DESC_TO_LABEL = {
    "769": 7,   # LEFT
    "770": 8,   # RIGHT
    "771": 9,   # FOOT
    "772": 10,  # TONGUE
}

SUBJECTS = [
    "A01", "A02", "A03", "A04", "A05", "A06", "A07", "A08", "A09",
]


def load_training_session(data_folder, subject, tmin=0.0, tmax=4.0, verbose=False):
    """
    Load one subject's training-session (T) recording as epoched,
    22-channel EEG data with correctly, consistently labeled classes.

    Parameters
    ----------
    data_folder : str
        Path to the folder containing the .gdf files.
    subject : str
        Subject ID without suffix, e.g. "A01" (the "T" is added here).
    tmin, tmax : float
        Epoch window in seconds relative to cue onset.

    Returns
    -------
    X : ndarray, shape (n_trials, 22, n_samples)
    y : ndarray, shape (n_trials,)
        Labels using the fixed DESC_TO_LABEL convention (7/8/9/10).
    """
    file_path = os.path.join(data_folder, subject + "T.gdf")

    raw = mne.io.read_raw_gdf(file_path, preload=True, verbose=verbose)
    raw.pick_types(eeg=True)

    events, event_dict = mne.events_from_annotations(raw, verbose=verbose)

    available_events = {
        desc: event_dict[desc]
        for desc in DESC_TO_LABEL
        if desc in event_dict
    }

    if len(available_events) != 4:
        missing = set(DESC_TO_LABEL) - set(available_events)
        print(f"WARNING: {subject}T missing motor-imagery events: {missing}")

    epochs = mne.Epochs(
        raw, events, event_id=available_events,
        tmin=tmin, tmax=tmax, baseline=None, preload=True, verbose=verbose,
    )

    X = epochs.get_data()[:, :22, :]

    code_to_label = {
        event_dict[desc]: DESC_TO_LABEL[desc]
        for desc in available_events
    }
    y = np.array([code_to_label[code] for code in epochs.events[:, -1]])

    return X, y


def load_evaluation_session(data_folder, label_folder, subject, tmin=0.0, tmax=4.0, verbose=False):
    """
    Load one subject's evaluation-session (E) recording. True labels
    come from the accompanying .mat file, since the GDF file itself
    only marks a generic "unknown cue" event (description "783").

    Returns
    -------
    X : ndarray, shape (288, 22, n_samples)
    y : ndarray, shape (288,)
        Labels using the fixed DESC_TO_LABEL convention (7/8/9/10),
        remapped from the .mat file's 1/2/3/4 convention.
    """
    file_path = os.path.join(data_folder, subject + "E.gdf")

    raw = mne.io.read_raw_gdf(file_path, preload=True, verbose=verbose)
    raw.pick_types(eeg=True)

    events, event_dict = mne.events_from_annotations(raw, verbose=verbose)

    if "783" not in event_dict:
        raise ValueError(
            f"{subject}E: '783' trial marker not found. "
            f"Available events: {event_dict}"
        )

    trial_code = event_dict["783"]
    target_events = events[events[:, 2] == trial_code]

    if len(target_events) != 288:
        raise ValueError(
            f"{subject}E: expected 288 trials, found {len(target_events)}"
        )

    epochs = mne.Epochs(
        raw, target_events, event_id={"MI": trial_code},
        tmin=tmin, tmax=tmax, baseline=None, preload=True, verbose=verbose,
    )

    X = epochs.get_data()[:, :22, :]

    mat_path = os.path.join(label_folder, subject + "E.mat")
    mat = loadmat(mat_path)
    y_raw = np.asarray(mat["classlabel"]).flatten().astype(int)

    if len(y_raw) != 288:
        raise ValueError(f"{subject}E: expected 288 labels, found {len(y_raw)}")

    # .mat file uses 1=Left, 2=Right, 3=Foot, 4=Tongue -> remap to 7/8/9/10
    mat_to_label = {1: 7, 2: 8, 3: 9, 4: 10}
    y = np.array([mat_to_label[v] for v in y_raw])

    return X, y


def load_pooled_training(data_folder, subjects=None, tmin=0.0, tmax=4.0, verbose=False):
    """
    Load and pool the training session for multiple subjects.

    Returns
    -------
    X : ndarray, shape (n_trials_total, 22, n_samples)
    y : ndarray, shape (n_trials_total,)
    groups : ndarray, shape (n_trials_total,)
        Subject ID string per trial (for LeaveOneGroupOut / LOSO).
    """
    subjects = subjects or SUBJECTS

    X_list, y_list, groups_list = [], [], []

    for subject in subjects:
        X, y = load_training_session(data_folder, subject, tmin=tmin, tmax=tmax, verbose=verbose)
        X_list.append(X)
        y_list.append(y)
        groups_list.extend([subject] * len(y))

        counts = dict(zip(*np.unique(y, return_counts=True)))
        print(f"{subject}T: {len(y)} trials, class counts {counts}")

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    groups = np.array(groups_list)

    print(f"\nPooled dataset: {X_all.shape}, classes: {dict(zip(*np.unique(y_all, return_counts=True)))}")

    return X_all, y_all, groups
