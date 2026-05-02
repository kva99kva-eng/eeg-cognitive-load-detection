from pathlib import Path

import numpy as np
from scipy.io import loadmat


RAW_DIR = Path("data/raw/stew")


def describe_value(name, value):
    print(f"\nKey: {name}")
    print("Type:", type(value))

    if isinstance(value, np.ndarray):
        print("Shape:", value.shape)
        print("Dtype:", value.dtype)

        flat = value.ravel()

        if flat.size > 0:
            print("First values:", flat[:10])

        if np.issubdtype(value.dtype, np.number):
            print("Min:", np.nanmin(value))
            print("Max:", np.nanmax(value))
    else:
        print("Value:", value)


def inspect_mat_file(path):
    print("\n" + "=" * 100)
    print("FILE:", path.name)
    print("=" * 100)

    mat = loadmat(path)

    keys = [key for key in mat.keys() if not key.startswith("__")]
    print("Keys:", keys)

    for key in keys:
        describe_value(key, mat[key])


def main():
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Folder not found: {RAW_DIR}")

    mat_files = sorted(RAW_DIR.glob("*.mat"))

    print("Found .mat files:", len(mat_files))

    for path in mat_files:
        inspect_mat_file(path)


if __name__ == "__main__":
    main()