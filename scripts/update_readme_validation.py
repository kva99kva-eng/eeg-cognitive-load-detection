from pathlib import Path

import pandas as pd


README_PATH = Path("README.md")

START_MARKER = "<!-- VALIDATION_COMPARISON_START -->"
END_MARKER = "<!-- VALIDATION_COMPARISON_END -->"


def summarize_results(path, validation_name):
    df = pd.read_csv(path)

    metrics = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
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
            "validation": validation_name,
            "model": model_name,
            "accuracy_mean": row[("accuracy", "mean")],
            "accuracy_std": row[("accuracy", "std")],
            "balanced_accuracy_mean": row[("balanced_accuracy", "mean")],
            "balanced_accuracy_std": row[("balanced_accuracy", "std")],
            "macro_f1_mean": row[("macro_f1", "mean")],
            "macro_f1_std": row[("macro_f1", "std")],
            "roc_auc_mean": row[("roc_auc", "mean")],
            "roc_auc_std": row[("roc_auc", "std")],
        })

    return rows


def fmt(mean, std):
    return f"{mean:.3f} +/- {std:.3f}"


def build_section():
    window_rows = summarize_results(
        "reports/kaggle_window_binary_cv_results.csv",
        "Window-level CV",
    )

    group_rows = summarize_results(
        "reports/group_binary_cv_results.csv",
        "Subject-independent CV",
    )

    rows = window_rows + group_rows

    lines = []

    lines.append(START_MARKER)
    lines.append("")
    lines.append("## Window-level vs subject-independent validation")
    lines.append("")
    lines.append("A key part of this project is comparing two validation strategies on the same Kaggle STEW binary dataset.")
    lines.append("")
    lines.append("EEG windows from the same subject are often highly correlated. If windows from the same person appear in both train and test sets, the model may learn subject-specific patterns instead of general cognitive load patterns.")
    lines.append("")
    lines.append("For this reason, subject-independent validation is a more realistic test of generalization to unseen people.")
    lines.append("")
    lines.append("### Validation strategies")
    lines.append("")
    lines.append("| Strategy | Description |")
    lines.append("|---|---|")
    lines.append("| Window-level CV | Random stratified split of EEG windows. The same subjects can appear in both train and test. |")
    lines.append("| Subject-independent CV | Grouped split by subject. Test subjects are unseen during training. |")
    lines.append("")
    lines.append("### Results")
    lines.append("")
    lines.append("| Validation | Model | Accuracy | Balanced Accuracy | Macro F1 | ROC-AUC |")
    lines.append("|---|---|---:|---:|---:|---:|")

    for row in rows:
        lines.append(
            f"| {row['validation']} "
            f"| {row['model']} "
            f"| {fmt(row['accuracy_mean'], row['accuracy_std'])} "
            f"| {fmt(row['balanced_accuracy_mean'], row['balanced_accuracy_std'])} "
            f"| {fmt(row['macro_f1_mean'], row['macro_f1_std'])} "
            f"| {fmt(row['roc_auc_mean'], row['roc_auc_std'])} |"
        )

    lines.append("")
    lines.append("### Interpretation")
    lines.append("")
    lines.append("Window-level cross-validation gives much higher performance, especially for Random Forest.")
    lines.append("However, this setup allows subject overlap between train and test folds.")
    lines.append("")
    lines.append("Subject-independent validation is substantially harder and gives lower, more realistic performance.")
    lines.append("This suggests that a large part of the window-level performance is likely driven by subject-specific EEG patterns rather than fully generalizable cognitive load markers.")
    lines.append("")
    lines.append("### Generated figures")
    lines.append("")
    lines.append("- `reports/figures/validation_comparison_accuracy.png`")
    lines.append("- `reports/figures/validation_comparison_balanced_accuracy.png`")
    lines.append("- `reports/figures/validation_comparison_macro_f1.png`")
    lines.append("- `reports/figures/validation_comparison_roc_auc.png`")
    lines.append("")
    lines.append(END_MARKER)
    lines.append("")

    return "\n".join(lines)


def update_file(path, section):
    if not path.exists():
        raise FileNotFoundError(path)

    text = path.read_text(encoding="utf-8")

    if START_MARKER in text and END_MARKER in text:
        before = text.split(START_MARKER)[0].rstrip()
        after = text.split(END_MARKER)[1].lstrip()
        new_text = before + "\n\n" + section + "\n" + after
    else:
        new_text = text.rstrip() + "\n\n" + section

    path.write_text(new_text, encoding="utf-8")


def main():
    section = build_section()

    update_file(README_PATH, section)

    results_path = Path("reports/results.md")
    if results_path.exists():
        update_file(results_path, section)

    print("Updated:")
    print("- README.md")
    print("- reports/results.md")


if __name__ == "__main__":
    main()