from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.eeg_cnn import EEGSimpleCNN


def normalize_windows(X: np.ndarray) -> np.ndarray:
    mean = X.mean(axis=2, keepdims=True)
    std = X.std(axis=2, keepdims=True) + 1e-6
    return ((X - mean) / std).astype(np.float32)


def main() -> None:
    data_path = PROJECT_ROOT / "data" / "processed" / "stew_kaggle_windows_binary.npz"
    model_path = PROJECT_ROOT / "models" / "eeg_cnn_subject_split_binary.pt"
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError("Run scripts/prepare_stew_kaggle.py first.")

    if not model_path.exists():
        raise FileNotFoundError("Run scripts/train_cnn_subject_split_binary.py first.")

    data = np.load(data_path)
    X = normalize_windows(data["X"])
    y = data["y"].astype(int)
    groups = data["groups"]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    _, test_idx = next(splitter.split(X, y, groups=groups))

    X_test = X[test_idx]
    y_test = y[test_idx]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    model = EEGSimpleCNN(
        n_channels=int(checkpoint.get("n_channels", X.shape[1])),
        n_times=int(checkpoint.get("n_times", X.shape[2])),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        logits = model(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy()
        probabilities = 1 / (1 + np.exp(-logits))

    rows = []

    for threshold in np.arange(0.1, 0.91, 0.05):
        predictions = (probabilities >= threshold).astype(int)

        rows.append(
            {
                "threshold": round(float(threshold), 2),
                "accuracy": accuracy_score(y_test, predictions),
                "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
                "precision": precision_score(y_test, predictions, zero_division=0),
                "recall": recall_score(y_test, predictions, zero_division=0),
                "macro_f1": f1_score(y_test, predictions, average="macro", zero_division=0),
                "weighted_f1": f1_score(y_test, predictions, average="weighted", zero_division=0),
                "roc_auc": roc_auc_score(y_test, probabilities),
            }
        )

    threshold_metrics = pd.DataFrame(rows)
    output_path = reports_dir / "cnn_threshold_tuning.csv"
    threshold_metrics.to_csv(output_path, index=False)

    print(f"Saved threshold metrics to: {output_path.relative_to(PROJECT_ROOT)}")
    print(threshold_metrics.sort_values("macro_f1", ascending=False).head())


if __name__ == "__main__":
    main()
