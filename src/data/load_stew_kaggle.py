from pathlib import Path

import numpy as np
from scipy.io import loadmat


RAW_DIR = Path("data/raw/stew")


def _load_first_variable(mat_path):
    mat = loadmat(mat_path)
    keys = [key for key in mat.keys() if not key.startswith("__")]

    if len(keys) != 1:
        raise ValueError(f"Expected exactly one variable in {mat_path}, got {keys}")

    return mat[keys[0]]


def create_windows(signal, window_size=256, step=64):
    """
    signal shape: (n_channels, n_times)

    returns:
        windows shape: (n_windows, n_channels, window_size)
    """
    windows = []

    n_channels, n_times = signal.shape

    for start in range(0, n_times - window_size + 1, step):
        end = start + window_size
        window = signal[:, start:end]
        windows.append(window)

    return np.array(windows, dtype=np.float32)


def load_stew_kaggle_windows(
    window_size=256,
    step=64,
    binary=False,
):
    """
    Loads Kaggle STEW .mat files and creates EEG windows.

    dataset.mat shape: (14, 19200, 45)
    class_012.mat shape: (45, 1)

    Parameters
    ----------
    binary:
        If False: use 3 classes: 0, 1, 2
        If True: drop class 1 and map:
            0 -> 0
            2 -> 1

    Returns
    -------
    X:
        shape (n_windows_total, 14, window_size)
    y:
        shape (n_windows_total,)
    groups:
        shape (n_windows_total,), subject id for GroupKFold
    """
    dataset_path = RAW_DIR / "dataset.mat"
    labels_path = RAW_DIR / "class_012.mat"

    if not dataset_path.exists():
        raise FileNotFoundError(f"Not found: {dataset_path}")

    if not labels_path.exists():
        raise FileNotFoundError(f"Not found: {labels_path}")

    data = _load_first_variable(dataset_path)
    labels = _load_first_variable(labels_path).ravel().astype(int)

    print("Raw data shape:", data.shape)
    print("Labels shape:", labels.shape)
    print("Labels distribution:", np.unique(labels, return_counts=True))

    if data.shape[0] != 14:
        raise ValueError(f"Expected 14 EEG channels in axis 0, got {data.shape}")

    if data.shape[2] != len(labels):
        raise ValueError(
            f"Subject count mismatch: data has {data.shape[2]}, labels have {len(labels)}"
        )

    X_all = []
    y_all = []
    groups_all = []

    n_subjects = data.shape[2]

    for subject_idx in range(n_subjects):
        label = labels[subject_idx]

        if binary and label == 1:
            continue

        if binary:
            label = 0 if label == 0 else 1

        signal = data[:, :, subject_idx]

        windows = create_windows(
            signal,
            window_size=window_size,
            step=step,
        )

        X_all.append(windows)
        y_all.append(np.full(len(windows), label, dtype=np.int64))
        groups_all.append(np.full(len(windows), subject_idx, dtype=np.int64))

    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    groups = np.concatenate(groups_all, axis=0)

    return X, y, groups


if __name__ == "__main__":
    print("3-class version:")
    X, y, groups = load_stew_kaggle_windows(binary=False)
    print("X:", X.shape)
    print("y:", y.shape)
    print("groups:", groups.shape)
    print("classes:", np.unique(y, return_counts=True))
    print("subjects:", len(np.unique(groups)))

    print("\nBinary version:")
    X, y, groups = load_stew_kaggle_windows(binary=True)
    print("X:", X.shape)
    print("y:", y.shape)
    print("groups:", groups.shape)
    print("classes:", np.unique(y, return_counts=True))
    print("subjects:", len(np.unique(groups)))