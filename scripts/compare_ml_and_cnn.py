from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def summarize_group_ml():
    path = Path("reports/group_binary_cv_results.csv")

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    metrics = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "roc_auc",
    ]

    summary = (
        df
        .groupby("model")[metrics]
        .agg(["mean", "std"])
    )

    rows = []

    for model_name in summary.index:
        row = summary.loc[model_name]

        rows.append({
            "model": model_name,
            "input": "Bandpower features",
            "validation": "Subject-independent 5-fold CV",
            "accuracy_mean": row[("accuracy", "mean")],
            "accuracy_std": row[("accuracy", "std")],
            "balanced_accuracy_mean": row[("balanced_accuracy", "mean")],
            "balanced_accuracy_std": row[("balanced_accuracy", "std")],
            "macro_f1_mean": row[("macro_f1", "mean")],
            "macro_f1_std": row[("macro_f1", "std")],
            "weighted_f1_mean": row[("weighted_f1", "mean")],
            "weighted_f1_std": row[("weighted_f1", "std")],
            "roc_auc_mean": row[("roc_auc", "mean")],
            "roc_auc_std": row[("roc_auc", "std")],
        })

    return rows


def load_cnn_metrics():
    default_path = Path("reports/cnn_subject_split_test_metrics.csv")
    tuned_path = Path("reports/cnn_test_metrics_tuned_threshold.csv")

    if not default_path.exists():
        raise FileNotFoundError(default_path)

    default = pd.read_csv(default_path).iloc[0]

    rows = []

    rows.append({
        "model": "SimpleCNN",
        "input": "Raw EEG windows",
        "validation": "Subject-independent single split, threshold=0.5",
        "accuracy_mean": default["accuracy"],
        "accuracy_std": np.nan,
        "balanced_accuracy_mean": default["balanced_accuracy"],
        "balanced_accuracy_std": np.nan,
        "macro_f1_mean": default["macro_f1"],
        "macro_f1_std": np.nan,
        "weighted_f1_mean": default["weighted_f1"],
        "weighted_f1_std": np.nan,
        "roc_auc_mean": default["roc_auc"],
        "roc_auc_std": np.nan,
    })

    if tuned_path.exists():
        tuned = pd.read_csv(tuned_path).iloc[0]

        rows.append({
            "model": "SimpleCNN tuned threshold",
            "input": "Raw EEG windows",
            "validation": f"Subject-independent single split, threshold={tuned['threshold']:.2f}",
            "accuracy_mean": tuned["accuracy"],
            "accuracy_std": np.nan,
            "balanced_accuracy_mean": tuned["balanced_accuracy"],
            "balanced_accuracy_std": np.nan,
            "macro_f1_mean": tuned["macro_f1"],
            "macro_f1_std": np.nan,
            "weighted_f1_mean": tuned["weighted_f1"],
            "weighted_f1_std": np.nan,
            "roc_auc_mean": tuned["roc_auc"],
            "roc_auc_std": np.nan,
        })

    return rows


def fmt(mean, std):
    if pd.isna(std):
        return f"{mean:.3f}"
    return f"{mean:.3f} +/- {std:.3f}"


def plot_model_comparison(df, output_dir):
    plot_df = df.copy()

    plot_df["label"] = plot_df["model"] + "\n" + plot_df["input"]

    metrics = [
        "balanced_accuracy_mean",
        "macro_f1_mean",
        "roc_auc_mean",
    ]

    readable_names = {
        "balanced_accuracy_mean": "Balanced Accuracy",
        "macro_f1_mean": "Macro F1",
        "roc_auc_mean": "ROC-AUC",
    }

    for metric in metrics:
        plt.figure(figsize=(10, 5))
        plt.bar(plot_df["label"], plot_df[metric])
        plt.ylim(0, 1)
        plt.title(f"Model comparison: {readable_names[metric]}")
        plt.ylabel(readable_names[metric])
        plt.xlabel("Model")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()

        filename = metric.replace("_mean", "")
        plt.savefig(output_dir / f"ml_vs_cnn_{filename}.png", dpi=150)
        plt.close()


def write_markdown(df, output_path):
    lines = []

    lines.append("# ML vs CNN comparison")
    lines.append("")
    lines.append("This report compares classical ML models based on bandpower features with a simple CNN trained directly on raw EEG windows.")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| Model | Input | Validation | Accuracy | Balanced Accuracy | Macro F1 | ROC-AUC |")
    lines.append("|---|---|---|---:|---:|---:|---:|")

    for _, row in df.iterrows():
        lines.append(
            f"| {row['model']} "
            f"| {row['input']} "
            f"| {row['validation']} "
            f"| {fmt(row['accuracy_mean'], row['accuracy_std'])} "
            f"| {fmt(row['balanced_accuracy_mean'], row['balanced_accuracy_std'])} "
            f"| {fmt(row['macro_f1_mean'], row['macro_f1_std'])} "
            f"| {fmt(row['roc_auc_mean'], row['roc_auc_std'])} |"
        )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("The CNN achieved a higher ROC-AUC than the classical ML baselines on the selected subject-independent test split, which suggests that it can rank EEG windows by cognitive load probability reasonably well.")
    lines.append("")
    lines.append("However, its balanced accuracy and macro F1 remained low because the default decision threshold produced a strong bias toward the high-load class.")
    lines.append("")
    lines.append("Threshold tuning on the validation set did not improve test-set balanced accuracy, so the default threshold result is kept as the main CNN baseline.")
    lines.append("")
    lines.append("Overall, this shows that deep learning on raw EEG is promising but requires better calibration, regularization and validation across multiple subject-independent folds.")
    lines.append("")
    lines.append("## Generated figures")
    lines.append("")
    lines.append("- `reports/figures/ml_vs_cnn_balanced_accuracy.png`")
    lines.append("- `reports/figures/ml_vs_cnn_macro_f1.png`")
    lines.append("- `reports/figures/ml_vs_cnn_roc_auc.png`")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    output_dir = Path("reports/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    rows.extend(summarize_group_ml())
    rows.extend(load_cnn_metrics())

    df = pd.DataFrame(rows)

    output_csv = Path("reports/ml_vs_cnn_summary.csv")
    output_md = Path("reports/ml_vs_cnn_comparison.md")

    df.to_csv(output_csv, index=False)

    plot_model_comparison(df, output_dir)
    write_markdown(df, output_md)

    print("Saved:")
    print("- reports/ml_vs_cnn_summary.csv")
    print("- reports/ml_vs_cnn_comparison.md")
    print("- reports/figures/ml_vs_cnn_balanced_accuracy.png")
    print("- reports/figures/ml_vs_cnn_macro_f1.png")
    print("- reports/figures/ml_vs_cnn_roc_auc.png")

    print("\nSummary:")
    print(df)


if __name__ == "__main__":
    main()