"""
stream_simulator.py

Simulates a LIVE EEG stream by replaying an existing recording
sample-by-sample, and runs sliding-window classification + decision
smoothing on it -- exactly the shape a real-time BCI pipeline would
have, just fed from a file instead of a live device.

This does NOT require any EEG hardware. It reads one of your existing
GDF recordings (e.g. a subject's E-session) through the same loader
used everywhere else in this project, then "pretends" the samples are
arriving in real time.

Usage (as a script):
    python stream_simulator.py

Usage (as a module, e.g. from pygame_demo.py):
    from stream_simulator import StreamSimulator
    sim = StreamSimulator(model_path=..., on_command=my_callback)
    sim.run(X_continuous)
"""

import os
import sys
import time
from collections import deque, Counter

import numpy as np
import joblib
import mne

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from data_loading import load_evaluation_session  # noqa: E402

SFREQ = 250
COMMAND_MAP = {7: "LEFT", 8: "RIGHT", 9: "FORWARD", 10: "STOP"}
LABEL_NAMES = {7: "Left Hand", 8: "Right Hand", 9: "Foot", 10: "Tongue"}


def band_pass_filter(window, sfreq=SFREQ, l_freq=8, h_freq=30):
    """Filter a single (channels, samples) window."""
    return mne.filter.filter_data(
        window.astype(np.float64), sfreq=sfreq, l_freq=l_freq, h_freq=h_freq, verbose=False
    )


class StreamSimulator:
    """
    Feeds pre-recorded EEG through a fake real-time loop:
    buffer incoming samples -> every `step_seconds`, classify the last
    `window_seconds` -> smooth the decision over the last
    `smoothing_window` predictions -> fire on_command() when a new,
    confirmed decision is reached.
    """

    def __init__(
        self,
        model_path,
        window_seconds=4.0,
        step_seconds=0.5,
        smoothing_window=3,
        min_agreement=2,
        sfreq=SFREQ,
        on_command=None,
        on_raw_prediction=None,
    ):
        artifact = joblib.load(model_path)
        self.csp = artifact["csp"]
        self.classifier = artifact["classifier"]

        self.window_samples = int(window_seconds * sfreq)
        self.step_samples = int(step_seconds * sfreq)
        self.sfreq = sfreq

        self.smoothing_window = smoothing_window
        self.min_agreement = min_agreement
        self.recent_predictions = deque(maxlen=smoothing_window)

        self.on_command = on_command or (lambda label, name, cmd, t: print(
            f"[t={t:5.1f}s] CONFIRMED: {name} ({label}) -> command: {cmd}"
        ))
        self.on_raw_prediction = on_raw_prediction

        self.last_confirmed_command = None

    def _classify_window(self, window):
        """window: shape (22, window_samples) -> returns int label (7/8/9/10)."""
        filtered = band_pass_filter(window, sfreq=self.sfreq)
        X = filtered[np.newaxis, :, :]
        X_csp = self.csp.transform(X)
        return int(self.classifier.predict(X_csp)[0])

    def _update_smoothing(self, raw_label, t):
        self.recent_predictions.append(raw_label)

        if self.on_raw_prediction:
            self.on_raw_prediction(raw_label, LABEL_NAMES[raw_label], t)

        if len(self.recent_predictions) < self.smoothing_window:
            return  # not enough history yet

        most_common_label, count = Counter(self.recent_predictions).most_common(1)[0]

        if count >= self.min_agreement and most_common_label != self.last_confirmed_command:
            self.last_confirmed_command = most_common_label
            self.on_command(
                most_common_label,
                LABEL_NAMES[most_common_label],
                COMMAND_MAP[most_common_label],
                t,
            )

    def run(self, X_continuous, realtime=False, speed_factor=1.0):
        """
        X_continuous : ndarray, shape (22, n_total_samples)
            A single continuous "recording" to stream through.
        realtime : bool
            If True, actually sleep to match wall-clock time (useful for
            a live demo). If False, runs as fast as possible (useful for
            quick testing).
        speed_factor : float
            >1.0 replays faster than real time (only used if realtime=True).
        """
        n_samples = X_continuous.shape[1]
        cursor = self.window_samples  # need at least one full window buffered first

        print(f"Streaming {n_samples/self.sfreq:.1f}s of data "
              f"({self.window_samples/self.sfreq:.1f}s window, "
              f"{self.step_samples/self.sfreq:.1f}s step)...\n")

        while cursor <= n_samples:
            window = X_continuous[:, cursor - self.window_samples:cursor]
            t = cursor / self.sfreq

            raw_label = self._classify_window(window)
            self._update_smoothing(raw_label, t)

            if realtime:
                time.sleep(self.step_samples / self.sfreq / speed_factor)

            cursor += self.step_samples

        print("\nStream ended.")


def build_fake_continuous_stream(data_folder, label_folder, subject, n_trials=10):
    """
    Helper: since BCI IV 2a data comes pre-cut into 4s trials, this
    concatenates a handful of a subject's real evaluation trials
    end-to-end to make one longer "continuous" stream to replay --
    a reasonable stand-in for a live feed when no live device exists.

    NOTE: requires the subject's true-label .mat file (e.g. A01E.mat)
    in label_folder. If you don't have those downloaded yet, use
    build_fake_continuous_stream_from_training() instead, which only
    needs the .gdf file.
    """
    X, y = load_evaluation_session(data_folder, label_folder, subject)
    X_subset = X[:n_trials]
    y_subset = y[:n_trials]

    X_continuous = np.concatenate([trial for trial in X_subset], axis=1)  # (22, n_trials * n_samples)

    print(f"Built a fake continuous stream from {n_trials} real trials of {subject}E.")
    print(f"True labels of the concatenated trials, in order: "
          f"{[LABEL_NAMES[y] for y in y_subset]}")

    return X_continuous


def build_fake_continuous_stream_from_training(data_folder, subject, n_trials=10):
    """
    Same idea as build_fake_continuous_stream(), but uses the T-session
    file instead of E-session -- no .mat label file needed, since the
    training GDF's own annotations already carry the class labels.
    Use this if you haven't downloaded the true_labels .mat files yet.
    """
    from data_loading import load_training_session  # local import to avoid unused import above

    X, y = load_training_session(data_folder, subject)
    X_subset = X[:n_trials]
    y_subset = y[:n_trials]

    X_continuous = np.concatenate([trial for trial in X_subset], axis=1)

    print(f"Built a fake continuous stream from {n_trials} real trials of {subject}T.")
    print(f"True labels of the concatenated trials, in order: "
          f"{[LABEL_NAMES[y] for y in y_subset]}")

    return X_continuous


if __name__ == "__main__":
    DATA_FOLDER = os.path.join(os.path.dirname(__file__), "..", "data", "BCI_IV_2a")
    LABEL_FOLDER = os.path.join(os.path.dirname(__file__), "..", "data", "true_labels")
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "single_subject_A01_csp12_svm.joblib")
    SUBJECT = "A01"

    # Uses the E-session with real true_labels .mat files -- the fair
    # test, since the model never saw these trials during training.
    X_continuous = build_fake_continuous_stream(DATA_FOLDER, LABEL_FOLDER, SUBJECT, n_trials=6)

    sim = StreamSimulator(model_path=MODEL_PATH, window_seconds=4.0, step_seconds=0.5, smoothing_window=3, min_agreement=2)
    sim.run(X_continuous, realtime=False)
