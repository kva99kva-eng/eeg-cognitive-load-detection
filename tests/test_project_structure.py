from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_required_top_level_files_exist():
    required_files = [
        "README.md",
        "requirements.txt",
        "LICENSE",
        ".gitignore",
        ".gitattributes",
    ]

    for file_path in required_files:
        assert (PROJECT_ROOT / file_path).exists(), f"Missing required file: {file_path}"


def test_required_directories_exist():
    required_dirs = [
        "app",
        "models",
        "notebooks",
        "reports",
        "scripts",
        "src",
    ]

    for dir_path in required_dirs:
        assert (PROJECT_ROOT / dir_path).exists(), f"Missing required directory: {dir_path}"


def test_core_source_files_exist():
    required_files = [
        "src/data/load_stew.py",
        "src/data/load_stew_kaggle.py",
        "src/features/bandpower.py",
        "src/models/eeg_cnn.py",
        "app/streamlit_app.py",
    ]

    for file_path in required_files:
        assert (PROJECT_ROOT / file_path).exists(), f"Missing source file: {file_path}"
