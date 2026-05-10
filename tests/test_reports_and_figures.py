from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_key_report_files_exist_and_are_not_empty():
    required_reports = [
        "reports/baseline_fast_summary.csv",
        "reports/group_binary_summary.csv",
        "reports/kaggle_window_binary_summary.csv",
        "reports/validation_comparison_summary.csv",
        "reports/ml_vs_cnn_summary.csv",
        "reports/results.md",
        "reports/validation_comparison.md",
        "reports/ml_vs_cnn_comparison.md",
    ]

    for file_path in required_reports:
        path = PROJECT_ROOT / file_path
        assert path.exists(), f"Missing report file: {file_path}"
        assert path.stat().st_size > 0, f"Report file is empty: {file_path}"


def test_key_figures_exist_and_are_not_empty():
    required_figures = [
        "reports/figures/validation_comparison_accuracy.png",
        "reports/figures/validation_comparison_balanced_accuracy.png",
        "reports/figures/validation_comparison_macro_f1.png",
        "reports/figures/validation_comparison_roc_auc.png",
        "reports/figures/random_forest_confusion_matrix.png",
        "reports/figures/cnn_subject_split_confusion_matrix.png",
        "reports/figures/ml_vs_cnn_balanced_accuracy.png",
        "reports/figures/channel_band_importance_heatmap.png",
    ]

    for file_path in required_figures:
        path = PROJECT_ROOT / file_path
        assert path.exists(), f"Missing figure file: {file_path}"
        assert path.stat().st_size > 0, f"Figure file is empty: {file_path}"


def test_demo_assets_exist():
    required_files = [
        "app/demo_samples.npz",
        "models/eeg_cnn_subject_split_binary.pt",
    ]

    for file_path in required_files:
        path = PROJECT_ROOT / file_path
        assert path.exists(), f"Missing demo asset: {file_path}"
        assert path.stat().st_size > 0, f"Demo asset is empty: {file_path}"
