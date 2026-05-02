from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_results():
    window_path = Path("reports/kaggle_window_binary_cv_results.csv")
    group_path = Path("reports/group_binary_cv_results.csv")

    if not window_path.exists():
        raise FileNotFoundError(window_path)

    if not group_path.exists():
        raise FileNotFoundError(group_path)

    window_df = pd.read_csv(window_path)
    group_df = pd.read_csv(group_path)

    window_df["validation_strategy"] = "Window-level CV"
    group_df["validation_strategy"] = "Subject-independent CV"

    if "n_overlapping_subjects" not in group_df.columns:
        group_df["n_overlapping_subjects"] = 0

    combined = pd.concat([window_df, group_df], ignore_index=True)

    return combined


def create_summary(df):
    metrics = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "roc_auc",
    ]

    summary = (
        df
        .groupby(["validation_strategy", "model"])[metrics]
        .agg(["mean", "std"])
        .reset_index()
    )

    return summary


def flatten_summary(summary):
    summary_flat = summary.copy()

    summary_flat.columns = [
        "_".join(col).strip("_")
        if isinstance(col, tuple)
        else col
        for col in summary_flat.columns
    ]

    return summary_flat


def plot_metric(df, metric, output_dir):
    summary = (
        df
        .groupby(["validation_strategy", "model"])[metric]
        .mean()
        .reset_index()
    )

    pivot = summary.pivot(
        index="model",
        columns="validation_strategy",
        values=metric,
    )

    ax = pivot.plot(
        kind="bar",
        figsize=(9, 5),
        ylim=(0, 1),
        rot=0,
    )

    ax.set_title(f"{metric} by validation strategy")
    ax.set_ylabel(metric)
    ax.set_xlabel("Model")
    ax.legend(title="Validation")

    plt.tight_layout()
    plt.savefig(output_dir / f"validation_comparison_{metric}.png", dpi=150)
    plt.close()


def write_markdown_report(df, summary_flat, output_path):
    lines = []

    lines.append("# Validation strategy comparison")
    lines.append("")
    lines.append("This report compares two validation strategies on the Kaggle STEW binary workload dataset.")
    lines.append("")
    lines.append("## Why this comparison matters")
    lines.append("")
    lines.append("EEG windows from the same subject are often highly correlated. If windows from one subject appear in both train and test sets, the model may learn subject-specific patterns instead of general cognitive load patterns.")
    lines.append("")
    lines.append("Therefore, subject-independent validation is more realistic for evaluating generalization to unseen people.")
    lines.append("")
    lines.append("## Compared strategies")
    lines.append("")
    lines.append("| Strategy | Description |")
    lines.append("|---|---|")
    lines.append("| Window-level CV | Random stratified split of EEG windows. Subjects can overlap between train and test. |")
    lines.append("| Subject-independent CV | Grouped split by subject. Test subjects are unseen during training. |")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| Validation | Model | Accuracy | Balanced Accuracy | Macro F1 | ROC-AUC |")
    lines.append("|---|---|---:|---:|---:|---:|")

    for _, row in summary_flat.iterrows():
        validation = row["validation_strategy"]
        model = row["model"]

        lines.append(
            f"| {validation} "
            f"| {model} "
            f"| {row['accuracy_mean']:.3f} +/- {row['accuracy_std']:.3f} "
            f"| {row['balanced_accuracy_mean']:.3f} +/- {row['balanced_accuracy_std']:.3f} "
            f"| {row['macro_f1_mean']:.3f} +/- {row['macro_f1_std']:.3f} "
            f"| {row['roc_auc_mean']:.3f} +/- {row['roc_auc_std']:.3f} |"
        )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("Window-level cross-validation gives much higher scores, especially for Random Forest.")
    lines.append("However, this setup allows subject overlap between training and test folds.")
    lines.append("")
    lines.append("Subject-independent validation is substantially harder and gives lower, more realistic performance.")
    lines.append("This indicates that a large part of the apparent window-level performance is likely driven by subject-specific EEG patterns.")
    lines.append("")
    lines.append("## Generated figures")
    lines.append("")
    lines.append("- `reports/figures/validation_comparison_accuracy.png`")
    lines.append("- `reports/figures/validation_comparison_balanced_accuracy.png`")
    lines.append("- `reports/figures/validation_comparison_macro_f1.png`")
    lines.append("- `reports/figures/validation_comparison_roc_auc.png`")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    output_dir = Path("reports/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_results()

    combined_path = Path("reports/validation_comparison_all_folds.csv")
    df.to_csv(combined_path, index=False)

    summary = create_summary(df)
    summary_flat = flatten_summary(summary)

    summary_path = Path("reports/validation_comparison_summary.csv")
    summary_flat.to_csv(summary_path, index=False)

    for metric in [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "roc_auc",
    ]:
        plot_metric(df, metric, output_dir)

    write_markdown_report(
        df=df,
        summary_flat=summary_flat,
        output_path=Path("reports/validation_comparison.md"),
    )

    print("Saved:")
    print("- reports/validation_comparison_all_folds.csv")
    print("- reports/validation_comparison_summary.csv")
    print("- reports/validation_comparison.md")
    print("- reports/figures/validation_comparison_accuracy.png")
    print("- reports/figures/validation_comparison_balanced_accuracy.png")
    print("- reports/figures/validation_comparison_macro_f1.png")
    print("- reports/figures/validation_comparison_roc_auc.png")

    print("\nSummary:")
    print(summary_flat)


if __name__ == "__main__":
    main()