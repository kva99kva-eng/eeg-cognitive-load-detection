from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier


RANDOM_STATE = 42

CHANNEL_NAMES = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"
]

BANDS = ["theta", "alpha", "beta", "gamma"]


def build_feature_names():
    names = []

    for channel in CHANNEL_NAMES:
        for band in BANDS:
            names.append(f"{channel}_{band}")

    return names


def train_random_forest(X_features, y):
    model = RandomForestClassifier(
        n_estimators=150,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_features, y)

    return model


def save_feature_importance(importances, output_path):
    feature_names = build_feature_names()

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    })

    df["channel"] = df["feature"].apply(lambda x: x.split("_")[0])
    df["band"] = df["feature"].apply(lambda x: x.split("_")[1])

    df = df.sort_values("importance", ascending=False)

    df.to_csv(output_path, index=False)

    return df


def plot_top20_features(df, output_dir):
    top20 = df.head(20).sort_values("importance", ascending=True)

    plt.figure(figsize=(8, 6))
    plt.barh(top20["feature"], top20["importance"])
    plt.title("Top 20 most important EEG bandpower features")
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_dir / "top20_feature_importance.png", dpi=150)
    plt.close()


def plot_channel_importance(df, output_dir):
    channel_importance = (
        df
        .groupby("channel")["importance"]
        .sum()
        .reindex(CHANNEL_NAMES)
        .sort_values(ascending=True)
    )

    plt.figure(figsize=(8, 5))
    plt.barh(channel_importance.index, channel_importance.values)
    plt.title("Aggregated feature importance by EEG channel")
    plt.xlabel("Total importance")
    plt.ylabel("Channel")
    plt.tight_layout()
    plt.savefig(output_dir / "channel_importance.png", dpi=150)
    plt.close()


def plot_band_importance(df, output_dir):
    band_importance = (
        df
        .groupby("band")["importance"]
        .sum()
        .reindex(BANDS)
        .sort_values(ascending=True)
    )

    plt.figure(figsize=(7, 4))
    plt.barh(band_importance.index, band_importance.values)
    plt.title("Aggregated feature importance by frequency band")
    plt.xlabel("Total importance")
    plt.ylabel("Frequency band")
    plt.tight_layout()
    plt.savefig(output_dir / "band_importance.png", dpi=150)
    plt.close()


def plot_channel_band_heatmap(df, output_dir):
    pivot = (
        df
        .pivot_table(
            index="channel",
            columns="band",
            values="importance",
            aggfunc="sum"
        )
        .reindex(index=CHANNEL_NAMES, columns=BANDS)
    )

    plt.figure(figsize=(7, 7))
    plt.imshow(pivot.values, aspect="auto")
    plt.colorbar(label="Feature importance")

    plt.xticks(range(len(BANDS)), BANDS)
    plt.yticks(range(len(CHANNEL_NAMES)), CHANNEL_NAMES)

    plt.title("Channel x frequency band importance")
    plt.xlabel("Frequency band")
    plt.ylabel("EEG channel")

    plt.tight_layout()
    plt.savefig(output_dir / "channel_band_importance_heatmap.png", dpi=150)
    plt.close()


def append_to_results_md(df):
    results_path = Path("reports/results.md")

    top_features = df.head(10)

    lines = []
    lines.append("")
    lines.append("## Feature importance")
    lines.append("")
    lines.append("Random Forest feature importance was used to estimate which spectral EEG features contributed most to the window-level classification.")
    lines.append("")
    lines.append("### Top 10 features")
    lines.append("")
    lines.append("| Feature | Importance |")
    lines.append("|---|---:|")

    for _, row in top_features.iterrows():
        lines.append(f"| {row['feature']} | {row['importance']:.5f} |")

    lines.append("")
    lines.append("Generated figures:")
    lines.append("")
    lines.append("- `reports/figures/top20_feature_importance.png`")
    lines.append("- `reports/figures/channel_importance.png`")
    lines.append("- `reports/figures/band_importance.png`")
    lines.append("- `reports/figures/channel_band_importance_heatmap.png`")
    lines.append("")

    with results_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    output_dir = Path("reports/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load("data/processed/stew_bandpower_features.npz")

    X_features = data["X_features"]
    y = data["y"]

    print("X_features:", X_features.shape)
    print("y:", y.shape)

    print("Training RandomForest on all windows for feature importance...")
    model = train_random_forest(X_features, y)

    importances = model.feature_importances_

    print("Importances:", importances.shape)

    df = save_feature_importance(
        importances=importances,
        output_path=Path("reports/feature_importance.csv")
    )

    print("\nTop 10 features:")
    print(df.head(10))

    plot_top20_features(df, output_dir)
    plot_channel_importance(df, output_dir)
    plot_band_importance(df, output_dir)
    plot_channel_band_heatmap(df, output_dir)
    append_to_results_md(df)

    print("\nSaved:")
    print("- reports/feature_importance.csv")
    print("- reports/figures/top20_feature_importance.png")
    print("- reports/figures/channel_importance.png")
    print("- reports/figures/band_importance.png")
    print("- reports/figures/channel_band_importance_heatmap.png")
    print("- updated reports/results.md")


if __name__ == "__main__":
    main()