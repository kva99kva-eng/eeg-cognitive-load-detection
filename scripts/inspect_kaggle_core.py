from pathlib import Path

import numpy as np
from scipy.io import loadmat, whosmat


RAW_DIR = Path("data/raw/stew")


def inspect_file(filename):
    path = RAW_DIR / filename

    print("\n" + "=" * 100)
    print(filename)
    print("=" * 100)

    print("\nwhosmat:")
    for item in whosmat(path):
        print(item)

    mat = loadmat(path)
    keys = [k for k in mat.keys() if not k.startswith("__")]

    print("\nKeys:", keys)

    for key in keys:
        value = mat[key]

        print(f"\nVariable: {key}")
        print("type:", type(value))
        print("shape:", getattr(value, "shape", None))
        print("dtype:", getattr(value, "dtype", None))
        print("ndim:", getattr(value, "ndim", None))
        print("size:", getattr(value, "size", None))

        if isinstance(value, np.ndarray):
            if value.dtype == object:
                print("Object array detected")
                print("First element type:", type(value.flat[0]))
                print("First element shape:", getattr(value.flat[0], "shape", None))
                print("First element dtype:", getattr(value.flat[0], "dtype", None))

                first = value.flat[0]
                if isinstance(first, np.ndarray):
                    print("First element first values:", first.ravel()[:20])
            else:
                print("First 30 values:", value.ravel()[:30])

                if np.issubdtype(value.dtype, np.number):
                    print("min:", np.nanmin(value))
                    print("max:", np.nanmax(value))


def main():
    inspect_file("dataset.mat")
    inspect_file("class_012.mat")


if __name__ == "__main__":
    main()