from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    ConfusionMatrixDisplay,
    classification_report,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.models.eeg_cnn import EEGSimpleCNN


RANDOM_STATE = 42
BATCH_SIZE = 128
N_EPOCHS = 15
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4


class EEGWindowDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]

        # Per-window, per-channel normalization
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


def print_split_info(name, y, groups, idx):
    print(f"\n{name}")
    print("-" * 80)
    print("Samples:", len(idx))
    print("Subjects:", np.unique(groups[idx]))
    print("Number of subjects:", len(np.unique(groups[idx])))
    print("Class distribution:", np.unique(y[idx], return_counts=True))


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(X_batch)

    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0

    all_y_true = []
    all_y_proba = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            proba = torch.sigmoid(logits)

            total_loss += loss.item() * len(X_batch)

            all_y_true.extend(y_batch.cpu().numpy().tolist())
            all_y_proba.extend(proba.cpu().numpy().tolist())

    y_true = np.array(all_y_true).astype(int)
    y_proba = np.array(all_y_proba)
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
    }

    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
    else:
        metrics["roc_auc"] = np.nan

    return metrics, y_true, y_pred, y_proba


def plot_training_history(history, output_dir):
    df = pd.DataFrame(history)

    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_loss"], label="train_loss")
    plt.plot(df["epoch"], df["val_loss"], label="val_loss")
    plt.title("CNN training loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "cnn_subject_split_training_loss.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["val_balanced_accuracy"], label="val_balanced_accuracy")
    plt.plot(df["epoch"], df["val_roc_auc"], label="val_roc_auc")
    plt.title("CNN validation metrics")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "cnn_subject_split_validation_metrics.png", dpi=150)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, output_dir):
    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["low load", "high load"],
    )

    disp.plot(values_format="d")
    plt.title("CNN subject-independent test confusion matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "cnn_subject_split_confusion_matrix.png", dpi=150)
    plt.close()


def main():
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    output_dir = Path("reports/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    data = np.load("data/processed/stew_kaggle_windows_binary.npz")

    X = data["X"]
    y = data["y"]
    groups = data["groups"]

    print("Loaded data")
    print("X:", X.shape)
    print("y:", y.shape)
    print("groups:", groups.shape)
    print("Classes:", np.unique(y, return_counts=True))
    print("Subjects:", len(np.unique(groups)))

    train_idx, val_idx, test_idx = make_subject_split(X, y, groups)

    print_split_info("Train split", y, groups, train_idx)
    print_split_info("Validation split", y, groups, val_idx)
    print_split_info("Test split", y, groups, test_idx)

    train_dataset = EEGWindowDataset(X[train_idx], y[train_idx])
    val_dataset = EEGWindowDataset(X[val_idx], y[val_idx])
    test_dataset = EEGWindowDataset(X[test_idx], y[test_idx])

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

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
    print("\nDevice:", device)

    model = EEGSimpleCNN(n_channels=14, n_times=256).to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_balanced_accuracy = -1.0
    best_model_path = models_dir / "eeg_cnn_subject_split_binary.pt"

    history = []

    for epoch in range(1, N_EPOCHS + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        val_metrics, _, _, _ = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_roc_auc": val_metrics["roc_auc"],
        }

        history.append(row)

        print(json.dumps(row, indent=2))

        if val_metrics["balanced_accuracy"] > best_val_balanced_accuracy:
            best_val_balanced_accuracy = val_metrics["balanced_accuracy"]

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                best_model_path,
            )

    print("\nLoading best model:", best_model_path)

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics, y_true, y_pred, y_proba = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
    )

    print("\nTest metrics:")
    print(json.dumps(test_metrics, indent=2))

    print("\nClassification report:")
    print(classification_report(y_true, y_pred))

    print("\nConfusion matrix:")
    print(confusion_matrix(y_true, y_pred))

    history_df = pd.DataFrame(history)
    history_df.to_csv("reports/cnn_subject_split_history.csv", index=False)

    test_results_df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "y_proba": y_proba,
    })

    test_results_df.to_csv("reports/cnn_subject_split_test_predictions.csv", index=False)

    test_metrics_df = pd.DataFrame([test_metrics])
    test_metrics_df.to_csv("reports/cnn_subject_split_test_metrics.csv", index=False)

    plot_training_history(history, output_dir)
    plot_confusion_matrix(y_true, y_pred, output_dir)

    print("\nSaved:")
    print("- models/eeg_cnn_subject_split_binary.pt")
    print("- reports/cnn_subject_split_history.csv")
    print("- reports/cnn_subject_split_test_predictions.csv")
    print("- reports/cnn_subject_split_test_metrics.csv")
    print("- reports/figures/cnn_subject_split_training_loss.png")
    print("- reports/figures/cnn_subject_split_validation_metrics.png")
    print("- reports/figures/cnn_subject_split_confusion_matrix.png")


if __name__ == "__main__":
    main()