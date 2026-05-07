from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_stew_kaggle import load_stew_kaggle_windows


def main() -> None:
    output_dir = PROJECT_ROOT / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Preparing 3-class EEG windows...")
    X, y, groups = load_stew_kaggle_windows(binary=False)

    np.savez_compressed(
        output_dir / "stew_kaggle_windows_3class.npz",
        X=X,
        y=y,
        groups=groups,
    )

    print("Preparing binary EEG windows...")
    X_binary, y_binary, groups_binary = load_stew_kaggle_windows(binary=True)

    np.savez_compressed(
        output_dir / "stew_kaggle_windows_binary.npz",
        X=X_binary,
        y=y_binary,
        groups=groups_binary,
    )

    print("Saved files:")
    print("- data/processed/stew_kaggle_windows_3class.npz")
    print("- data/processed/stew_kaggle_windows_binary.npz")
    print("Binary X:", X_binary.shape)
    print("Binary y:", y_binary.shape)
    print("Binary groups:", groups_binary.shape)


if __name__ == "__main__":
    main()
