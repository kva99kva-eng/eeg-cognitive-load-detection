from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


RANDOM_STATE = 42


def plot_metric_comparison(results_df, output_dir):
    metrics = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "roc_auc",
    ]

    summary = results_df.groupby("model")[metrics].mean()

    ax = summary.plot(
        kind="bar",
        figsize=(10, 5),
        ylim=(0, 1),
        rot=0,
    )

    ax.set_title("Baseline model comparison")
    ax.set_ylabel("Score")
    ax.set_xlabel("Model")
    ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(output_dir / "baseline_metric_comparison.png", dpi=150)
    plt.close()


def plot_random_forest_confusion_matrix(output_dir):
    data = np.load("data/processed/stew_bandpower_features.npz")

    X_features = data["X_features"]
    y = data["y"]

    model = RandomForestClassifier(
        n_estimators=150,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    y_pred = cross_val_predict(
        model,
        X_features,
        y,
        cv=cv,
        n_jobs=-1,
    )

    cm = confusion_matrix(y, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["low load", "high load"],
    )

    disp.plot(values_format="d")
    plt.title("RandomForest confusion matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "random_forest_confusion_matrix.png", dpi=150)
    plt.close()


def write_results_markdown(results_df, output_path):
    metrics = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "roc_auc",
    ]

    summary = (
        results_df
        .groupby("model")[metrics]
        .agg(["mean", "std"])
    )

    lines = []

    lines.append("# Baseline results")
    lines.append("")
    lines.append("This report summarizes the first baseline for EEG cognitive load detection.")
    lines.append("")
    lines.append("## Validation setup")
    lines.append("")
    lines.append("- Features: spectral bandpower")
    lines.append("- Frequency bands: theta, alpha, beta, gamma")
    lines.append("- Validation: 5-fold stratified window-level cross-validation")
    lines.append("- Models: Logistic Regression, Random Forest")
    lines.append("")
    lines.append("> Note: this is a window-level baseline. Subject-independent validation should be added as the next stage when true subject IDs are available.")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| Model | Accuracy | Balanced Accuracy | Macro F1 | Weighted F1 | ROC-AUC |")
    lines.append("|---|---:|---:|---:|---:|---:|")

    for model_name in summary.index:
        row = summary.loc[model_name]

        lines.append(
            f"| {model_name} "
            f"| {row[('accuracy', 'mean')]:.3f} ± {row[('accuracy', 'std')]:.3f} "
            f"| {row[('balanced_accuracy', 'mean')]:.3f} ± {row[('balanced_accuracy', 'std')]:.3f} "
            f"| {row[('macro_f1', 'mean')]:.3f} ± {row[('macro_f1', 'std')]:.3f} "
            f"| {row[('weighted_f1', 'mean')]:.3f} ± {row[('weighted_f1', 'std')]:.3f} "
            f"| {row[('roc_auc', 'mean')]:.3f} ± {row[('roc_auc', 'std')]:.3f} |"
        )

    lines.append("")
    lines.append("## Current interpretation")
    lines.append("")
    lines.append("Random Forest clearly outperforms Logistic Regression on bandpower features.")
    lines.append("This suggests that the relationship between spectral EEG features and cognitive load is likely non-linear.")
    lines.append("")
    lines.append("## Generated figures")
    lines.append("")
    lines.append("- `reports/figures/baseline_metric_comparison.png`")
    lines.append("- `reports/figures/random_forest_confusion_matrix.png`")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    output_dir = Path("reports/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = Path("reports/baseline_fast_cv_results.csv")

    if not results_path.exists():
        raise FileNotFoundError(
            "reports/baseline_fast_cv_results.csv not found. "
            "Run scripts/train_baseline_fast.py first."
        )

    results_df = pd.read_csv(results_path)

    plot_metric_comparison(results_df, output_dir)
    plot_random_forest_confusion_matrix(output_dir)
    write_results_markdown(results_df, Path("reports/results.md"))

    print("Saved:")
    print("- reports/figures/baseline_metric_comparison.png")
    print("- reports/figures/random_forest_confusion_matrix.png")
    print("- reports/results.md")


if __name__ == "__main__":
    main()