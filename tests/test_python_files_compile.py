import py_compile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_core_python_files_compile():
    python_files = [
        "src/data/load_stew.py",
        "src/data/load_stew_kaggle.py",
        "src/features/bandpower.py",
        "src/models/eeg_cnn.py",
        "app/streamlit_app.py",
        "scripts/compare_validation_strategies.py",
        "scripts/compare_ml_and_cnn.py",
        "scripts/analyze_feature_importance.py",
        "scripts/train_baseline_fast.py",
        "scripts/train_cnn_subject_split_binary.py",
        "scripts/tune_cnn_threshold.py",
    ]

    for file_path in python_files:
        py_compile.compile(
            str(PROJECT_ROOT / file_path),
            doraise=True,
        )
