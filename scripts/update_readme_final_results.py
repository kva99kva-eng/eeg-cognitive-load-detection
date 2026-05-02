from pathlib import Path

import pandas as pd


README_PATH = Path("README.md")

START_MARKER = "<!-- FINAL_RESULTS_START -->"
END_MARKER = "<!-- FINAL_RESULTS_END -->"


def fmt(mean, std=None):
    if std is None or pd.isna(std):
        return f"{mean:.3f}"
    return f"{mean:.3f} +/- {std:.3f}"


def load_validation_comparison():
    path = Path("reports/validation_comparison_summary.csv")
    if not path.exists():
        raise FileNotFoundError(path)

    return pd.read_csv(path)


def load_ml_vs_cnn():
    path = Path("reports/ml_vs_cnn_summary.csv")
    if not path.exists():
        raise FileNotFoundError(path)

    return pd.read_csv(path)


def build_section():
    validation_df = load_validation_comparison()
    ml_cnn_df = load_ml_vs_cnn()

    lines = []

    lines.append(START_MARKER)
    lines.append("")
    lines.append("## Final results and key findings")
    lines.append("")
    lines.append("This project evaluates EEG cognitive load detection under both optimistic and realistic validation settings.")
    lines.append("")
    lines.append("The most important result is not only the model score, but the difference between window-level and subject-independent validation.")
    lines.append("")
    lines.append("### Window-level vs subject-independent validation")
    lines.append("")
    lines.append("| Validation | Model | Accuracy | Balanced Accuracy | Macro F1 | ROC-AUC |")
    lines.append("|---|---|---:|---:|---:|---:|")

    for _, row in validation_df.iterrows():
        lines.append(
            f"| {row['validation_strategy']} "
            f"| {row['model']} "
            f"| {fmt(row['accuracy_mean'], row['accuracy_std'])} "
            f"| {fmt(row['balanced_accuracy_mean'], row['balanced_accuracy_std'])} "
            f"| {fmt(row['macro_f1_mean'], row['macro_f1_std'])} "
            f"| {fmt(row['roc_auc_mean'], row['roc_auc_std'])} |"
        )

    lines.append("")
    lines.append("Window-level validation gives much higher scores because EEG windows from the same subject can appear in both train and test sets.")
    lines.append("")
    lines.append("Subject-independent validation is harder and more realistic because the model is tested on unseen subjects.")
    lines.append("")
    lines.append("![Validation comparison](reports/figures/validation_comparison_balanced_accuracy.png)")
    lines.append("")
    lines.append("### Classical ML vs CNN")
    lines.append("")
    lines.append("| Model | Input | Validation | Accuracy | Balanced Accuracy | Macro F1 | ROC-AUC |")
    lines.append("|---|---|---|---:|---:|---:|---:|")

    for _, row in ml_cnn_df.iterrows():
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
    lines.append("The CNN achieved the best ROC-AUC on the selected subject-independent split, but its balanced accuracy and macro F1 remained limited due to a strong bias toward the high-load class.")
    lines.append("")
    lines.append("Threshold tuning did not improve test-set balanced accuracy, so the default threshold result is kept as the main CNN baseline.")
    lines.append("")
    lines.append("![ML vs CNN ROC-AUC](reports/figures/ml_vs_cnn_roc_auc.png)")
    lines.append("")
    lines.append("### Feature importance")
    lines.append("")
    lines.append("Random Forest feature importance was used to estimate which EEG spectral features contributed most to the window-level classification.")
    lines.append("")
    lines.append("The most important features were mainly located in frontal and temporal channels, especially in the theta band.")
    lines.append("")
    lines.append("![Top feature importance](reports/figures/top20_feature_importance.png)")
    lines.append("")
    lines.append("![Channel-band importance](reports/figures/channel_band_importance_heatmap.png)")
    lines.append("")
    lines.append("### Main conclusion")
    lines.append("")
    lines.append("This project demonstrates that high EEG classification scores can be misleading when validation is performed at the window level.")
    lines.append("")
    lines.append("Subject-independent validation provides a more realistic estimate of generalization to unseen people and reveals that cognitive load detection from EEG remains a challenging problem.")
    lines.append("")
    lines.append("### Current limitations")
    lines.append("")
    lines.append("- The CNN was evaluated on a single subject-independent split, not full cross-validation.")
    lines.append("- The dataset is relatively small for deep learning.")
    lines.append("- The CNN requires better calibration and regularization.")
    lines.append("- The current project focuses on offline classification, not real-time inference.")
    lines.append("")
    lines.append("### Next steps")
    lines.append("")
    lines.append("- Train CNN with subject-independent cross-validation.")
    lines.append("- Add EEGNet-style architecture.")
    lines.append("- Add probability calibration.")
    lines.append("- Add Streamlit demo.")
    lines.append("- Package the project for GitHub and resume.")
    lines.append("")
    lines.append(END_MARKER)
    lines.append("")

    return "\n".join(lines)


def update_readme(section):
    if not README_PATH.exists():
        raise FileNotFoundError(README_PATH)

    text = README_PATH.read_text(encoding="utf-8")

    if START_MARKER in text and END_MARKER in text:
        before = text.split(START_MARKER)[0].rstrip()
        after = text.split(END_MARKER)[1].lstrip()
        new_text = before + "\n\n" + section + "\n" + after
    else:
        new_text = text.rstrip() + "\n\n" + section

    README_PATH.write_text(new_text, encoding="utf-8")


def main():
    section = build_section()
    update_readme(section)

    print("Updated README.md")
    print("Added final results section.")


if __name__ == "__main__":
    main()