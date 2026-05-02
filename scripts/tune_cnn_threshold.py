from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.models.eeg_cnn import EEGSimpleCNN


RANDOM_STATE = 42
BATCH_SIZE = 128


class EEGWindowDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]

        mean = x.mean(axis=1, keepdims=True)
        std = x.std(axis=1, keepdims=True) + 1e-6
        x = (x - mean) / std

        y = self.y[idx]

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def make_subject_split(X, y, groups):
    outer_cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    train_val_idx, test_idx = next(outer_cv.split(X, y, groups))

    X_train_val = X[train_val_idx]
    y_train_val = y[train_val_idx]
    groups_train_val = groups[train_val_idx]

    inner_cv = StratifiedGroupKFold(
        n_splits=4,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    inner_train_idx, val_idx = next(
        inner_cv.split(X_train_val, y_train_val, groups_train_val)
    )

    train_idx = train_val_idx[inner_train_idx]
    val_idx = train_val_idx[val_idx]

    return train_idx, val_idx, test_idx


def predict_proba(model, loader, device):
    model.eval()

    all_y_true = []
    all_y_proba = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)

            logits = model(X_batch)
            proba = torch.sigmoid(logits)

            all_y_true.extend(y_batch.numpy().tolist())
            all_y_proba.extend(proba.cpu().numpy().tolist())

    return np.array(all_y_true).astype(int), np.array(all_y_proba)


def metrics_at_threshold(y_true, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)

    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def find_best_threshold(y_true, y_proba, metric_name="balanced_accuracy"):
    rows = []

    for threshold in np.linspace(0.05, 0.95, 181):
        row = metrics_at_threshold(y_true, y_proba, threshold)
        rows.append(row)

    df = pd.DataFrame(rows)

    best_idx = df[metric_name].idxmax()
    best_row = df.loc[best_idx].to_dict()

    return best_row, df


def main():
    data = np.load("data/processed/stew_kaggle_windows_binary.npz")

    X = data["X"]
    y = data["y"]
    groups = data["groups"]

    _, val_idx, test_idx = make_subject_split(X, y, groups)

    print("Validation subjects:", np.unique(groups[val_idx]))
    print("Test subjects:", np.unique(groups[test_idx]))
    print("Validation classes:", np.unique(y[val_idx], return_counts=True))
    print("Test classes:", np.unique(y[test_idx], return_counts=True))

    val_dataset = EEGWindowDataset(X[val_idx], y[val_idx])
    test_dataset = EEGWindowDataset(X[test_idx], y[test_idx])

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EEGSimpleCNN(n_channels=14, n_times=256).to(device)

    checkpoint_path = Path("models/eeg_cnn_subject_split_binary.pt")

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Model checkpoint not found. Run scripts/train_cnn_subject_split_binary.py first."
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    y_val_true, y_val_proba = predict_proba(model, val_loader, device)
    y_test_true, y_test_proba = predict_proba(model, test_loader, device)

    best_threshold_row, threshold_df = find_best_threshold(
        y_val_true,
        y_val_proba,
        metric_name="balanced_accuracy",
    )

    best_threshold = best_threshold_row["threshold"]

    print("\nBest validation threshold:")
    print(json.dumps(best_threshold_row, indent=2))

    test_metrics_default = metrics_at_threshold(
        y_test_true,
        y_test_proba,
        threshold=0.5,
    )

    test_metrics_tuned = metrics_at_threshold(
        y_test_true,
        y_test_proba,
        threshold=best_threshold,
    )

    print("\nTest metrics with default threshold 0.5:")
    print(json.dumps(test_metrics_default, indent=2))

    print("\nTest metrics with tuned threshold:")
    print(json.dumps(test_metrics_tuned, indent=2))

    y_test_pred_default = (y_test_proba >= 0.5).astype(int)
    y_test_pred_tuned = (y_test_proba >= best_threshold).astype(int)

    print("\nDefault threshold classification report:")
    print(classification_report(y_test_true, y_test_pred_default))

    print("\nTuned threshold classification report:")
    print(classification_report(y_test_true, y_test_pred_tuned))

    print("\nDefault threshold confusion matrix:")
    print(confusion_matrix(y_test_true, y_test_pred_default))

    print("\nTuned threshold confusion matrix:")
    print(confusion_matrix(y_test_true, y_test_pred_tuned))

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    threshold_df.to_csv(
        reports_dir / "cnn_threshold_search_validation.csv",
        index=False,
    )

    pd.DataFrame([test_metrics_default]).to_csv(
        reports_dir / "cnn_test_metrics_default_threshold.csv",
        index=False,
    )

    pd.DataFrame([test_metrics_tuned]).to_csv(
        reports_dir / "cnn_test_metrics_tuned_threshold.csv",
        index=False,
    )

    pd.DataFrame({
        "y_true": y_test_true,
        "y_proba": y_test_proba,
        "y_pred_default": y_test_pred_default,
        "y_pred_tuned": y_test_pred_tuned,
    }).to_csv(
        reports_dir / "cnn_test_predictions_tuned_threshold.csv",
        index=False,
    )

    print("\nSaved:")
    print("- reports/cnn_threshold_search_validation.csv")
    print("- reports/cnn_test_metrics_default_threshold.csv")
    print("- reports/cnn_test_metrics_tuned_threshold.csv")
    print("- reports/cnn_test_predictions_tuned_threshold.csv")


if __name__ == "__main__":
    main()