from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data.load_stew_kaggle import load_stew_kaggle_windows


def save_dataset(binary):
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y, groups = load_stew_kaggle_windows(
        window_size=256,
        step=64,
        binary=binary,
    )

    if binary:
        output_path = output_dir / "stew_kaggle_windows_binary.npz"
    else:
        output_path = output_dir / "stew_kaggle_windows_3class.npz"

    np.savez_compressed(
        output_path,
        X=X,
        y=y,
        groups=groups,
    )

    print("\nSaved:", output_path)
    print("X:", X.shape)
    print("y:", y.shape)
    print("groups:", groups.shape)
    print("classes:", np.unique(y, return_counts=True))
    print("subjects:", len(np.unique(groups)))


def main():
    print("=" * 80)
    print("Saving 3-class dataset")
    print("=" * 80)
    save_dataset(binary=False)

    print("\n" + "=" * 80)
    print("Saving binary dataset")
    print("=" * 80)
    save_dataset(binary=True)


if __name__ == "__main__":
    main()