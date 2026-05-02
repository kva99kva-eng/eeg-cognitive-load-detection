from datasets import load_dataset


def main():
    dataset = load_dataset(
        "monster-monash/STEW",
        trust_remote_code=True
    )

    print(dataset)
    print("Splits:", dataset.keys())

    split_name = list(dataset.keys())[0]
    data = dataset[split_name]

    print("Split:", split_name)
    print("Number of rows:", len(data))
    print("Columns:", data.column_names)

    first = data[0]
    print("First item keys:", first.keys())

    for key, value in first.items():
        try:
            print(key, type(value), "len:", len(value))
        except TypeError:
            print(key, type(value), value)


if __name__ == "__main__":
    main()