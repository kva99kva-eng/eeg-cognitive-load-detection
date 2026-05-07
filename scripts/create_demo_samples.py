from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    data_path = PROJECT_ROOT / "data" / "processed" / "stew_kaggle_windows_binary.npz"
    output_path = PROJECT_ROOT / "app" / "demo_samples.npz"

    if not data_path.exists():
        raise FileNotFoundError(
            "Missing data/processed/stew_kaggle_windows_binary.npz. "
            "Run scripts/prepare_stew_kaggle.py first."
        )

    data = np.load(data_path)
    X = data["X"]
    y = data["y"]
    groups = data["groups"]

    rng = np.random.default_rng(42)
    sample_size = min(250, len(X))
    indices = rng.choice(len(X), size=sample_size, replace=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X=X[indices],
        y=y[indices],
        groups=groups[indices],
        source_indices=indices,
    )

    print(f"Saved demo samples to: {output_path.relative_to(PROJECT_ROOT)}")
    print(f"Samples: {sample_size}")


if __name__ == "__main__":
    main()
