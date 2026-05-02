from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch


SFREQ = 128

CHANNEL_NAMES = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"
]


def plot_class_balance(y, output_dir):
    labels, counts = np.unique(y, return_counts=True)

    plt.figure(figsize=(6, 4))
    plt.bar([str(label) for label in labels], counts)
    plt.title("Class balance")
    plt.xlabel("Class")
    plt.ylabel("Number of windows")
    plt.tight_layout()
    plt.savefig(output_dir / "class_balance.png", dpi=150)
    plt.close()


def plot_eeg_window(X, y, class_id, output_dir):
    idx = np.where(y == class_id)[0][0]
    window = X[idx]

    time = np.arange(window.shape[1]) / SFREQ

    plt.figure(figsize=(12, 8))

    offset = 0
    for ch_idx, ch_name in enumerate(CHANNEL_NAMES):
        signal = window[ch_idx]
        signal = signal - np.mean(signal)
        signal = signal / (np.std(signal) + 1e-8)

        plt.plot(time, signal + offset, linewidth=0.8)
        plt.text(time[-1] + 0.02, offset, ch_name, va="center")

        offset += 4

    plt.title(f"Example EEG window, class {class_id}")
    plt.xlabel("Time, seconds")
    plt.ylabel("Channels with vertical offset")
    plt.tight_layout()
    plt.savefig(output_dir / f"eeg_window_class_{class_id}.png", dpi=150)
    plt.close()


def plot_psd_by_class(X, y, output_dir, max_samples_per_class=1000):
    plt.figure(figsize=(10, 5))

    for class_id in np.unique(y):
        idx = np.where(y == class_id)[0][:max_samples_per_class]
        X_class = X[idx]

        freqs, psd = welch(
            X_class,
            fs=SFREQ,
            nperseg=256,
            axis=-1
        )

        mean_psd = psd.mean(axis=(0, 1))

        mask = (freqs >= 1) & (freqs <= 45)
        plt.plot(freqs[mask], mean_psd[mask], label=f"class {class_id}")

    plt.title("Average PSD by class")
    plt.xlabel("Frequency, Hz")
    plt.ylabel("Power spectral density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "psd_by_class.png", dpi=150)
    plt.close()


def save_subject_class_balance(y, groups, output_dir):
    df = pd.DataFrame({
        "subject": groups,
        "label": y
    })

    table = (
        df
        .groupby(["subject", "label"])
        .size()
        .unstack(fill_value=0)
    )

    table.to_csv(output_dir / "subject_class_balance.csv")

    print("\nSubject/class balance:")
    print(table.head(10))


def main():
    output_dir = Path("reports/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load("data/processed/stew_windows.npz")

    X = data["X"]
    y = data["y"]
    groups = data["groups"]

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("groups shape:", groups.shape)

    print("Class distribution:", np.unique(y, return_counts=True))
    print("Number of subjects:", len(np.unique(groups)))

    plot_class_balance(y, output_dir)
    plot_eeg_window(X, y, class_id=0, output_dir=output_dir)
    plot_eeg_window(X, y, class_id=1, output_dir=output_dir)
    plot_psd_by_class(X, y, output_dir)
    save_subject_class_balance(y, groups, output_dir)

    print("\nSaved figures to:", output_dir)
    print("Created:")
    print("- class_balance.png")
    print("- eeg_window_class_0.png")
    print("- eeg_window_class_1.png")
    print("- psd_by_class.png")
    print("- subject_class_balance.csv")


if __name__ == "__main__":
    main()