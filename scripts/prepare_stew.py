from pathlib import Path
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data.load_stew import load_stew


def main():
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y, groups = load_stew()

    output_path = output_dir / "stew_windows.npz"

    np.savez_compressed(
        output_path,
        X=X,
        y=y,
        groups=groups
    )

    print("Saved:", output_path)
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("groups shape:", groups.shape)


if __name__ == "__main__":
    main()