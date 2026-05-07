from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    window_path = REPORTS_DIR / "window_baseline_binary_metrics.csv"
    group_path = REPORTS_DIR / "group_baseline_binary_metrics.csv"

    if not window_path.exists() or not group_path.exists():
        raise FileNotFoundError(
            "Baseline metric files are missing. Run both baseline training scripts first."
        )

    window_metrics = pd.read_csv(window_path)
    group_metrics = pd.read_csv(group_path)

    comparison = pd.concat(
        [window_metrics, group_metrics],
        ignore_index=True,
    )

    output_path = REPORTS_DIR / "validation_strategy_comparison.csv"
    comparison.to_csv(output_path, index=False)

    plot_df = comparison[["validation", "balanced_accuracy"]].copy()

    plt.figure(figsize=(8, 5))
    plt.bar(plot_df["validation"], plot_df["balanced_accuracy"])
    plt.ylabel("Balanced accuracy")
    plt.xlabel("Validation strategy")
    plt.title("Validation Strategy Comparison")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    figure_path = FIGURES_DIR / "validation_comparison_balanced_accuracy.png"
    plt.savefig(figure_path, dpi=150)
    plt.close()

    print(f"Saved comparison to: {output_path.relative_to(PROJECT_ROOT)}")
    print(f"Saved figure to: {figure_path.relative_to(PROJECT_ROOT)}")
    print(comparison)


if __name__ == "__main__":
    main()
