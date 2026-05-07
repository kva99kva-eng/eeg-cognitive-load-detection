from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.eeg_cnn import EEGSimpleCNN


def normalize_windows(X: np.ndarray) -> np.ndarray:
    mean = X.mean(axis=2, keepdims=True)
    std = X.std(axis=2, keepdims=True) + 1e-6
    return ((X - mean) / std).astype(np.float32)


def main() -> None:
    data_path = PROJECT_ROOT / "data" / "processed" / "stew_kaggle_windows_binary.npz"
    reports_dir = PROJECT_ROOT / "reports"
    models_dir = PROJECT_ROOT / "models"

    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(
            "Missing data/processed/stew_kaggle_windows_binary.npz. "
            "Run scripts/prepare_stew_kaggle.py first."
        )

    data = np.load(data_path)
    X = normalize_windows(data["X"])
    y = data["y"].astype(np.float32)
    groups = data["groups"]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=42,
    )

    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
    )

    model = EEGSimpleCNN(n_channels=X.shape[1], n_times=X.shape[2]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    n_epochs = 8

    for epoch in range(1, n_epochs + 1):
        model.train()
        losses = []

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            losses.append(float(loss.detach().cpu()))

        print(f"Epoch {epoch}/{n_epochs} - loss: {np.mean(losses):.4f}")

    model.eval()

    with torch.no_grad():
        test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        logits = model(test_tensor).cpu().numpy()
        probabilities = 1 / (1 + np.exp(-logits))

    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "validation": "subject_independent_group_shuffle_split",
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro")),
        "weighted_f1": float(f1_score(y_test, predictions, average="weighted")),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "n_train_windows": int(len(train_idx)),
        "n_test_windows": int(len(test_idx)),
    }

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
            "n_channels": int(X.shape[1]),
            "n_times": int(X.shape[2]),
        },
        models_dir / "eeg_cnn_subject_split_binary.pt",
    )

    pd.DataFrame([metrics]).to_csv(
        reports_dir / "cnn_subject_split_binary_metrics.csv",
        index=False,
    )

    (reports_dir / "cnn_subject_split_binary_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print("CNN metrics:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
