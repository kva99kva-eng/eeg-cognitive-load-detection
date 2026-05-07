from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    ml_path = REPORTS_DIR / "group_baseline_binary_metrics.csv"
    cnn_path = REPORTS_DIR / "cnn_subject_split_binary_metrics.csv"

    if not ml_path.exists() or not cnn_path.exists():
        raise FileNotFoundError(
            "Missing ML or CNN metrics. Run training scripts first."
        )

    ml_metrics = pd.read_csv(ml_path).assign(model="RandomForest bandpower")
    cnn_metrics = pd.read_csv(cnn_path).assign(model="CNN raw EEG")

    comparison = pd.concat([ml_metrics, cnn_metrics], ignore_index=True)

    output_path = REPORTS_DIR / "ml_vs_cnn_comparison.csv"
    comparison.to_csv(output_path, index=False)

    plt.figure(figsize=(7, 5))
    plt.bar(comparison["model"], comparison["balanced_accuracy"])
    plt.ylabel("Balanced accuracy")
    plt.xlabel("Model")
    plt.title("ML vs CNN: Subject-Independent Balanced Accuracy")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()

    figure_path = FIGURES_DIR / "ml_vs_cnn_balanced_accuracy.png"
    plt.savefig(figure_path, dpi=150)
    plt.close()

    print(f"Saved comparison to: {output_path.relative_to(PROJECT_ROOT)}")
    print(f"Saved figure to: {figure_path.relative_to(PROJECT_ROOT)}")
    print(comparison)


if __name__ == "__main__":
    main()
