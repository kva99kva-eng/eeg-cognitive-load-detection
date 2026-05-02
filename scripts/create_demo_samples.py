from pathlib import Path

import numpy as np


def main():
    input_path = Path("data/processed/stew_kaggle_windows_binary.npz")
    output_path = Path("app/demo_samples.npz")

    if not input_path.exists():
        raise FileNotFoundError(
            "data/processed/stew_kaggle_windows_binary.npz not found. "
            "Run scripts/prepare_stew_kaggle.py first."
        )

    data = np.load(input_path)

    X = data["X"]
    y = data["y"]
    groups = data["groups"]

    rng = np.random.default_rng(42)

    idx_class_0 = np.where(y == 0)[0]
    idx_class_1 = np.where(y == 1)[0]

    selected_0 = rng.choice(idx_class_0, size=50, replace=False)
    selected_1 = rng.choice(idx_class_1, size=50, replace=False)

    selected_idx = np.concatenate([selected_0, selected_1])
    rng.shuffle(selected_idx)

    X_demo = X[selected_idx]
    y_demo = y[selected_idx]
    groups_demo = groups[selected_idx]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        X=X_demo,
        y=y_demo,
        groups=groups_demo,
        source_indices=selected_idx,
    )

    print("Saved:", output_path)
    print("X:", X_demo.shape)
    print("y:", y_demo.shape)
    print("groups:", groups_demo.shape)
    print("classes:", np.unique(y_demo, return_counts=True))
    print("subjects:", len(np.unique(groups_demo)))


if __name__ == "__main__":
    main()