import numpy as np
from scipy.signal import welch


FREQ_BANDS = {
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}


def extract_bandpower_features(window, sfreq=128):
    """
    Extract bandpower features from one EEG window.

    Parameters
    ----------
    window : np.ndarray
        Shape: (n_channels, n_times)

    Returns
    -------
    features : np.ndarray
        Shape: (n_channels * n_bands,)
    """
    features = []

    for channel_signal in window:
        freqs, psd = welch(
            channel_signal,
            fs=sfreq,
            nperseg=min(256, len(channel_signal))
        )

        for low, high in FREQ_BANDS.values():
            mask = (freqs >= low) & (freqs <= high)

            if not np.any(mask):
                power = 0.0
            else:
                power = np.trapezoid(psd[mask], freqs[mask])

            features.append(power)

    return np.array(features, dtype=np.float32)


def build_feature_matrix(X, sfreq=128):
    """
    Convert EEG windows into bandpower feature matrix.

    Parameters
    ----------
    X : np.ndarray
        Shape: (n_samples, n_channels, n_times)

    Returns
    -------
    X_features : np.ndarray
        Shape: (n_samples, n_features)
    """
    return np.vstack([
        extract_bandpower_features(window, sfreq=sfreq)
        for window in X
    ])
