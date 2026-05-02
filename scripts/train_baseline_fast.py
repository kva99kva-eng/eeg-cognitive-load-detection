from pathlib import Path
import sys

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.features.bandpower import build_feature_matrix


RANDOM_STATE = 42


def get_features():
    features_path = Path("data/processed/stew_bandpower_features.npz")

    if features_path.exists():
        print("Loading cached features...")
        data = np.load(features_path)
        return data["X_features"], data["y"]

    print("Cached features not found. Building features...")

    data = np.load("data/processed/stew_windows.npz")
    X = data["X"]
    y = data["y"]

    X_features = build_feature_matrix(X, sfreq=128)

    np.savez_compressed(
        features_path,
        X_features=X_features,
        y=y
    )

    print("Saved features:", features_path)

    return X_features, y


def get_models():
    return {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ]),

        "RandomForest": RandomForestClassifier(
            n_estimators=150,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def evaluate_model(model_name, model, X_features, y):
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    results = []
    all_y_true = []
    all_y_pred = []
    all_y_proba = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X_features, y)):
        X_train = X_features[train_idx]
        X_test = X_features[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        print(f"\nTraining {model_name}, fold {fold}...")

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        row = {
            "model": model_name,
            "fold": fold,
            "accuracy": accuracy_score(y_test, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
            "macro_f1": f1_score(y_test, y_pred, average="macro"),
            "weighted_f1": f1_score(y_test, y_pred, average="weighted"),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }

        print(row)

        results.append(row)

        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())
        all_y_proba.extend(y_proba.tolist())

    print(f"\nClassification report for {model_name}:")
    print(classification_report(all_y_true, all_y_pred))

    print(f"Confusion matrix for {model_name}:")
    print(confusion_matrix(all_y_true, all_y_pred))

    return results


def main():
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    X_features, y = get_features()

    print("X_features:", X_features.shape)
    print("y:", y.shape)
    print("Classes:", np.unique(y, return_counts=True))

    all_results = []

    for model_name, model in get_models().items():
        print("\n" + "=" * 80)
        print("Model:", model_name)
        print("=" * 80)

        model_results = evaluate_model(
            model_name=model_name,
            model=model,
            X_features=X_features,
            y=y,
        )

        all_results.extend(model_results)

    results_df = pd.DataFrame(all_results)

    results_path = reports_dir / "baseline_fast_cv_results.csv"
    summary_path = reports_dir / "baseline_fast_summary.csv"

    results_df.to_csv(results_path, index=False)

    summary = (
        results_df
        .groupby("model")
        [["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "roc_auc"]]
        .agg(["mean", "std"])
    )

    summary.to_csv(summary_path)

    print("\nSummary:")
    print(summary)

    print("\nSaved:")
    print(results_path)
    print(summary_path)


if __name__ == "__main__":
    main()