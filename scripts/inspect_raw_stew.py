from pathlib import Path
import re

import pandas as pd


RAW_DIR = Path("data/raw/stew")


def main():
    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Folder not found: {RAW_DIR}. "
            "Download and extract raw STEW files into data/raw/stew/"
        )

    files = sorted(RAW_DIR.rglob("*.txt"))

    print("Found txt files:", len(files))

    pattern = re.compile(r"sub(\d+)_(lo|hi)\.txt", re.IGNORECASE)

    matched = []

    for path in files:
        match = pattern.search(path.name)

        if match:
            subject = int(match.group(1))
            condition = match.group(2).lower()
            matched.append((path, subject, condition))

    print("Matched STEW files:", len(matched))

    subjects = sorted(set(subject for _, subject, _ in matched))
    print("Subjects:", subjects)
    print("Number of subjects:", len(subjects))

    print("\nFirst matched files:")
    for item in matched[:10]:
        print(item)

    if matched:
        example_path = matched[0][0]
        print("\nReading example file:")
        print(example_path)

        df = pd.read_csv(
            example_path,
            sep=r"\s+|,|;",
            engine="python",
            header=None
        )

        print("Example shape:", df.shape)
        print(df.head())


if __name__ == "__main__":
    main()