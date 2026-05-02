from pathlib import Path

import numpy as np
from scipy.io import loadmat


RAW_DIR = Path("data/raw/stew")


def print_mat_summary(filename):
    path = RAW_DIR / filename

    print("\n" + "=" * 80)
    print(filename)
    print("=" * 80)

    mat = loadmat(path)
    keys = [k for k in mat.keys() if not k.startswith("__")]

    print("Keys:", keys)

    for key in keys:
        value = mat[key]

        print(f"\nVariable: {key}")
        print("type:", type(value))
        print("shape:", getattr(value, "shape", None))
        print("dtype:", getattr(value, "dtype", None))

        if isinstance(value, np.ndarray):
            print("ndim:", value.ndim)
            print("size:", value.size)

            if value.dtype == object:
                print("Object array detected")
                print("First element type:", type(value.flat[0]))
                print("First element shape:", getattr(value.flat[0], "shape", None))
                print("First element dtype:", getattr(value.flat[0], "dtype", None))
                print("First element first values:", value.flat[0].ravel()[:10])
            else:
                print("First values:", value.ravel()[:20])

                if np.issubdtype(value.dtype, np.number):
                    print("min:", np.nanmin(value))
                    print("max:", np.nanmax(value))


def main():
    for filename in [
        "dataset.mat",
        "class_012.mat",
        "rating.mat",
        "three_class_one_hot.mat",
    ]:
        print_mat_summary(filename)


if __name__ == "__main__":
    main()