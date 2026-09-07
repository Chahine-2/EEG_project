"""
export_model.py

Trains the project's best-known configuration (single-subject A01,
12-component CSP + linear SVM, 83.0% mean 5-fold CV accuracy) on the
full training session and saves it as a reusable artifact.

This is the model referenced in the command-mapping demonstration and
is the natural starting point for any future real-time / object-control
work: load it once, call predict_class() on new epochs.

Usage:
    python export_model.py
Produces:
    models/single_subject_A01_csp12_svm.joblib
"""

import os
import sys
import numpy as np
import joblib
from mne.decoding import CSP
from sklearn.svm import SVC

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from data_loading import load_training_session  # noqa: E402

DATA_FOLDER = os.path.join(os.path.dirname(__file__), "..", "data", "BCI_IV_2a")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "single_subject_A01_csp12_svm.joblib")

SUBJECT = "A01"
N_CSP_COMPONENTS = 12
LABEL_NAMES = {7: "Left Hand", 8: "Right Hand", 9: "Foot", 10: "Tongue"}


def band_pass_filter(X, sfreq=250, l_freq=8, h_freq=30):
    import mne
    X_filtered = np.empty_like(X)
    for i in range(X.shape[0]):
        X_filtered[i] = mne.filter.filter_data(
            X[i], sfreq=sfreq, l_freq=l_freq, h_freq=h_freq, verbose=False
        )
    return X_filtered


def train_and_export():
    print(f"Loading {SUBJECT}T ...")
    X, y = load_training_session(DATA_FOLDER, SUBJECT, tmin=0.0, tmax=4.0)
    print(f"Loaded: {X.shape}, classes: {dict(zip(*np.unique(y, return_counts=True)))}")

    print("Band-pass filtering (8-30 Hz) ...")
    X_filtered = band_pass_filter(X)

    print(f"Fitting CSP ({N_CSP_COMPONENTS} components) ...")
    csp = CSP(n_components=N_CSP_COMPONENTS, reg=None, log=True, norm_trace=False)
    X_csp = csp.fit_transform(X_filtered, y)

    print("Fitting linear SVM ...")
    svm = SVC(kernel="linear", C=1)
    svm.fit(X_csp, y)

    train_acc = svm.score(X_csp, y)
    print(f"Training-set accuracy (sanity check, not a generalization estimate): {train_acc*100:.2f}%")

    artifact = {
        "csp": csp,
        "classifier": svm,
        "label_names": LABEL_NAMES,
        "subject": SUBJECT,
        "n_csp_components": N_CSP_COMPONENTS,
        "band_pass": (8, 30),
        "epoch_window": (0.0, 4.0),
        "expected_cv_accuracy": 0.83,
        "notes": (
            "Trained on the full A01T training session (all 288 trials). "
            "83.0% is the 5-fold cross-validated estimate from "
            "01_single_subject.ipynb; this final fit uses all available "
            "data and should NOT be re-evaluated on its own training set."
        ),
    }

    joblib.dump(artifact, MODEL_PATH)
    print(f"\nSaved model artifact to: {MODEL_PATH}")


def predict_class(epoch, model_path=MODEL_PATH):
    """
    Predict the motor imagery class for a single new epoch.

    Parameters
    ----------
    epoch : ndarray, shape (22, n_samples)
        A single band-pass-filtered (8-30 Hz), 0-4s EEG epoch, 22 channels.
    model_path : str
        Path to a .joblib artifact produced by train_and_export().

    Returns
    -------
    label : int        -- 7/8/9/10
    label_name : str    -- e.g. "Left Hand"
    command : str        -- mapped control command
    """
    artifact = joblib.load(model_path)

    command_map = {7: "LEFT", 8: "RIGHT", 9: "FORWARD", 10: "STOP"}

    X = epoch[np.newaxis, :, :]  # CSP expects (n_epochs, n_channels, n_samples)
    X_csp = artifact["csp"].transform(X)
    label = int(artifact["classifier"].predict(X_csp)[0])

    return label, artifact["label_names"][label], command_map[label]


if __name__ == "__main__":
    train_and_export()
