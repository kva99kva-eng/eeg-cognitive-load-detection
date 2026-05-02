from pathlib import Path

import numpy as np
import pandas as pd


N_SUBJECTS = 48


def summarize_groups(y, groups, name):
    df = pd.DataFrame({
        "group": groups,
        "label": y,
    })

    table = (
        df
        .groupby(["group", "label"])
        .size()
        .unstack(fill_value=0)
    )

    if 0 not in table.columns:
        table[0] = 0
    if 1 not in table.columns:
        table[1] = 0

    table = table[[0, 1]]

    valid_balanced = ((table[0] == 297) & (table[1] == 297)).sum()
    both_classes = ((table[0] > 0) & (table[1] > 0)).sum()

    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)
    print(table.head(15))
    print("...")
    print(table.tail(15))
    print()
    print("Groups:", len(table))
    print("Groups with both classes:", both_classes)
    print("Groups with exactly 297 low and 297 high:", valid_balanced)

    return table, valid_balanced


def print_label_runs(y):
    print("\nLabel order diagnostics")
    print("=" * 80)

    changes = np.where(np.diff(y) != 0)[0] + 1

    print("First 30 labels:", y[:30])
    print("Last 30 labels:", y[-30:])
    print("Number of label changes:", len(changes))
    print("First 20 change indices:", changes[:20])

    if len(changes) > 0:
        print("First change at index:", changes[0])

    unique_first_half = np.unique(y[: len(y) // 2], return_counts=True)
    unique_second_half = np.unique(y[len(y) // 2 :], return_counts=True)

    print("First half class distribution:", unique_first_half)
    print("Second half class distribution:", unique_second_half)


def main():
    data = np.load("data/processed/stew_windows.npz")

    y = data["y"]

    n_samples = len(y)
    windows_per_subject_per_class = n_samples // (N_SUBJECTS * 2)

    print("n_samples:", n_samples)
    print("n_subjects:", N_SUBJECTS)
    print("windows_per_subject_per_class:", windows_per_subject_per_class)
    print("class distribution:", np.unique(y, return_counts=True))

    print_label_runs(y)

    # Вариант A:
    # каждый участник идёт блоком: 297 low + 297 high
    groups_subject_blocks = np.repeat(
        np.arange(N_SUBJECTS),
        windows_per_subject_per_class * 2
    )

    # Вариант B:
    # сначала идут все low по участникам, потом все high по участникам:
    # low:  subject 0, subject 1, ..., subject 47
    # high: subject 0, subject 1, ..., subject 47
    groups_class_first_same_order = np.concatenate([
        np.repeat(np.arange(N_SUBJECTS), windows_per_subject_per_class),
        np.repeat(np.arange(N_SUBJECTS), windows_per_subject_per_class),
    ])

    # Вариант C:
    # сначала low по участникам, потом high в обратном порядке
    groups_class_first_reverse_order = np.concatenate([
        np.repeat(np.arange(N_SUBJECTS), windows_per_subject_per_class),
        np.repeat(np.arange(N_SUBJECTS)[::-1], windows_per_subject_per_class),
    ])

    candidates = {
        "A_subject_blocks_594_windows": groups_subject_blocks,
        "B_class_first_same_subject_order": groups_class_first_same_order,
        "C_class_first_reverse_subject_order": groups_class_first_reverse_order,
    }

    best_name = None
    best_groups = None
    best_score = -1

    for name, groups in candidates.items():
        _, score = summarize_groups(y, groups, name)

        if score > best_score:
            best_score = score
            best_name = name
            best_groups = groups

    print("\n" + "=" * 80)
    print("Best candidate:", best_name)
    print("Balanced groups:", best_score, "/", N_SUBJECTS)

    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_dir / "stew_inferred_groups.npz",
        groups=best_groups,
        candidate_name=best_name,
        balanced_groups=best_score,
    )

    print("Saved inferred groups to:")
    print(output_dir / "stew_inferred_groups.npz")

    if best_score == N_SUBJECTS:
        print("\nOK: candidate looks valid for GroupKFold.")
    else:
        print("\nWARNING: no candidate perfectly reconstructs subject groups.")
        print("In this case, raw STEW files should be used for true subject-independent validation.")


if __name__ == "__main__":
    main()