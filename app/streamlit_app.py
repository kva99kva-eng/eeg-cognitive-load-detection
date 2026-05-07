from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.models.eeg_cnn import EEGSimpleCNN


CHANNEL_NAMES = [
    "AF3",
    "F7",
    "F3",
    "FC5",
    "T7",
    "P7",
    "O1",
    "O2",
    "P8",
    "T8",
    "FC6",
    "F4",
    "F8",
    "AF4",
]

SFREQ = 128


def normalize_window(x: np.ndarray) -> np.ndarray:
    """Normalize EEG window channel-wise."""
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True) + 1e-6

    return (x - mean) / std


@st.cache_data
def load_demo_data() -> dict:
    """Load prepared EEG demo samples."""
    path = PROJECT_ROOT / "app" / "demo_samples.npz"

    if not path.exists():
        raise FileNotFoundError(
            "app/demo_samples.npz not found. "
            "Run scripts/create_demo_samples.py first."
        )

    data = np.load(path)

    return {
        "X": data["X"],
        "y": data["y"],
        "groups": data["groups"],
        "source_indices": data["source_indices"],
    }


@st.cache_resource
def load_model():
    """Load trained CNN checkpoint."""
    checkpoint_path = PROJECT_ROOT / "models" / "eeg_cnn_subject_split_binary.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "models/eeg_cnn_subject_split_binary.pt not found. "
            "Run scripts/train_cnn_subject_split_binary.py first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EEGSimpleCNN(n_channels=14, n_times=256).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, device


def predict(model, device, x: np.ndarray) -> tuple[float, float]:
    """Predict low/high cognitive load probabilities for one EEG window."""
    x_norm = normalize_window(x)

    tensor = torch.tensor(
        x_norm[None, :, :],
        dtype=torch.float32,
    ).to(device)

    with torch.no_grad():
        logit = model(tensor)
        probability_high = torch.sigmoid(logit).item()
        probability_low = 1.0 - probability_high

    return probability_low, probability_high


def plot_eeg_window(x: np.ndarray):
    """Plot one 14-channel EEG window."""
    time = np.arange(x.shape[1]) / SFREQ

    fig, ax = plt.subplots(figsize=(12, 7))
    offset = 0

    for ch_idx, ch_name in enumerate(CHANNEL_NAMES):
        signal = x[ch_idx]
        signal = signal - np.mean(signal)
        signal = signal / (np.std(signal) + 1e-8)

        ax.plot(time, signal + offset, linewidth=0.8)
        ax.text(time[-1] + 0.03, offset, ch_name, va="center", fontsize=9)

        offset += 4

    ax.set_title("EEG window, 14 channels")
    ax.set_xlabel("Time, seconds")
    ax.set_ylabel("Normalized channels with vertical offset")
    ax.set_yticks([])
    ax.grid(alpha=0.2)
    fig.tight_layout()

    return fig


def plot_probability_bar(prob_low: float, prob_high: float):
    """Plot predicted probabilities for low/high load."""
    fig, ax = plt.subplots(figsize=(6, 3))

    labels = ["Low load", "High load"]
    values = [prob_low, prob_high]

    ax.bar(labels, values)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    ax.set_title("CNN predicted probabilities")

    for i, value in enumerate(values):
        ax.text(i, value + 0.02, f"{value:.3f}", ha="center")

    fig.tight_layout()

    return fig


def main() -> None:
    """Run Streamlit demo app."""
    st.set_page_config(
        page_title="EEG Cognitive Load Detection",
        layout="wide",
    )

    st.title("EEG Cognitive Load Detection")
    st.caption("Demo app for classifying cognitive load from EEG windows using a simple CNN.")

    try:
        demo_data = load_demo_data()
        model, device = load_model()
    except FileNotFoundError as error:
        st.error(str(error))
        st.stop()

    X = demo_data["X"]
    y = demo_data["y"]
    groups = demo_data["groups"]
    source_indices = demo_data["source_indices"]

    st.sidebar.header("Sample selection")

    sample_idx = st.sidebar.slider(
        "EEG sample index",
        min_value=0,
        max_value=len(X) - 1,
        value=0,
        step=1,
    )

    threshold = st.sidebar.slider(
        "Decision threshold for high load",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.01,
    )

    x = X[sample_idx]
    true_label = int(y[sample_idx])
    subject = int(groups[sample_idx])
    source_index = int(source_indices[sample_idx])

    prob_low, prob_high = predict(model, device, x)
    predicted_label = int(prob_high >= threshold)

    label_names = {
        0: "Low cognitive load",
        1: "High cognitive load",
    }

    st.subheader("Prediction")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("True label", label_names[true_label])
    col2.metric("Predicted label", label_names[predicted_label])
    col3.metric("P(high load)", f"{prob_high:.3f}")
    col4.metric("Threshold", f"{threshold:.2f}")

    st.write(
        {
            "sample_idx": sample_idx,
            "source_index": source_index,
            "subject_id": subject,
            "device": str(device),
            "window_shape": x.shape,
        }
    )

    left, right = st.columns([2, 1])

    with left:
        st.subheader("EEG signal")
        fig_signal = plot_eeg_window(x)
        st.pyplot(fig_signal)

    with right:
        st.subheader("Predicted probabilities")
        fig_proba = plot_probability_bar(prob_low, prob_high)
        st.pyplot(fig_proba)

    st.subheader("Model note")
    st.info(
        "This CNN was trained on raw EEG windows using a subject-independent split. "
        "The model achieved higher ROC-AUC than classical baselines on the selected split, "
        "but its threshold-based classification remained biased toward the high-load class."
    )

    st.subheader("Project context")
    st.markdown(
        """
        This demo is part of an EEG cognitive load detection project.

        The project compares:

        - classical ML on spectral bandpower features;
        - CNN on raw EEG windows;
        - window-level validation;
        - subject-independent validation.

        The key finding is that window-level validation can strongly overestimate performance
        because EEG windows from the same subject can appear in both train and test sets.
        """
    )

    st.subheader("Demo samples overview")

    df = pd.DataFrame(
        {
            "sample_idx": np.arange(len(X)),
            "true_label": y,
            "subject_id": groups,
            "source_index": source_indices,
        }
    )

    st.dataframe(df, use_container_width=True)


if __name__ == "__main__":
    main()
