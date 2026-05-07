import numpy as np
from scipy.signal import welch


FREQ_BANDS = {
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}


def extract_bandpower_features(window: np.ndarray, sfreq: int = 128) -> np.ndarray:
    """
    Extract bandpower features from one EEG window.

    Parameters
    ----------
    window:
        EEG window with shape (n_channels, n_times).
    sfreq:
        Sampling frequency.

    Returns
    -------
    np.ndarray
        Feature vector with shape (n_channels * n_bands,).
    """
    features = []

    for channel_signal in window:
        freqs, psd = welch(
            channel_signal,
            fs=sfreq,
            nperseg=min(256, len(channel_signal)),
        )

        for low, high in FREQ_BANDS.values():
            mask = (freqs >= low) & (freqs <= high)

            if not np.any(mask):
                power = 0.0
            else:
                power = np.trapezoid(psd[mask], freqs[mask])

            features.append(power)

    return np.array(features, dtype=np.float32)


def build_feature_matrix(X: np.ndarray, sfreq: int = 128) -> np.ndarray:
    """
    Convert EEG windows into a bandpower feature matrix.

    Parameters
    ----------
    X:
        EEG windows with shape (n_samples, n_channels, n_times).
    sfreq:
        Sampling frequency.

    Returns
    -------
    np.ndarray
        Feature matrix with shape (n_samples, n_features).
    """
    return np.vstack([
        extract_bandpower_features(window, sfreq=sfreq)
        for window in X
    ])
