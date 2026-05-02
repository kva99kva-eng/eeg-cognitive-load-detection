import numpy as np
from datasets import load_dataset


def load_stew():
    """
    Loads STEW EEG cognitive load dataset.

    Returns:
        X: EEG windows, shape (n_samples, n_channels, n_times)
        y: labels, shape (n_samples,)
        groups: subject ids, shape (n_samples,)
    """
    dataset = load_dataset(
        "monster-monash/STEW",
        trust_remote_code=True
    )

    data = dataset["train"]

    X = np.array(data["X"], dtype=np.float32)
    y = np.array(data["y"], dtype=np.int64)

    # В датасете 48 испытуемых.
    # Всего 28512 окон, значит на каждого приходится 594 окна.
    n_subjects = 48
    n_samples = len(X)

    if n_samples % n_subjects != 0:
        raise ValueError(
            f"Cannot split {n_samples} samples equally into {n_subjects} subjects"
        )

    samples_per_subject = n_samples // n_subjects
    groups = np.repeat(np.arange(n_subjects), samples_per_subject)

    return X, y, groups


if __name__ == "__main__":
    X, y, groups = load_stew()

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("groups shape:", groups.shape)
    print("X dtype:", X.dtype)
    print("y classes:", np.unique(y, return_counts=True))
    print("Number of subjects:", len(np.unique(groups)))
    print("Samples per subject:", len(X) // len(np.unique(groups)))