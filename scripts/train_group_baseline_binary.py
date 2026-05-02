from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd

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

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:
    StratifiedGroupKFold = None
    from sklearn.model_selection import GroupKFold


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.features.bandpower import build_feature_matrix


RANDOM_STATE = 42


def load_or_build_features():
    features_path = Path("data/processed/stew_kaggle_bandpower_binary.npz")

    if features_path.exists():
        print("Loading cached Kaggle binary bandpower features...")
        data = np.load(features_path)
        return data["X_features"], data["y"], data["groups"]

    print("Cached features not found. Building features...")

    data = np.load("data/processed/stew_kaggle_windows_binary.npz")

    X = data["X"]
    y = data["y"]
    groups = data["groups"]

    print("Raw windows:", X.shape)
    print("Labels:", y.shape)
    print("Groups:", groups.shape)

    X_features = build_feature_matrix(X, sfreq=128)

    np.savez_compressed(
        features_path,
        X_features=X_features,
        y=y,
        groups=groups,
    )

    print("Saved features:", features_path)

    return X_features, y, groups


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
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def make_cv(n_splits=5):
    if StratifiedGroupKFold is not None:
        return StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

    print("WARNING: StratifiedGroupKFold unavailable. Falling back to GroupKFold.")
    return GroupKFold(n_splits=n_splits)


def evaluate_model(model_name, model, X_features, y, groups, n_splits=5):
    cv = make_cv(n_splits=n_splits)

    fold_results = []

    all_y_true = []
    all_y_pred = []
    all_y_proba = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X_features, y, groups)):
        X_train = X_features[train_idx]
        X_test = X_features[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        train_groups = np.unique(groups[train_idx])
        test_groups = np.unique(groups[test_idx])

        print("\n" + "-" * 80)
        print(f"{model_name} | Fold {fold}")
        print("Train subjects:", train_groups)
        print("Test subjects:", test_groups)
        print("Train class distribution:", np.unique(y_train, return_counts=True))
        print("Test class distribution:", np.unique(y_test, return_counts=True))

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = y_pred

        result = {
            "model": model_name,
            "fold": fold,
            "n_train_subjects": len(train_groups),
            "n_test_subjects": len(test_groups),
            "accuracy": accuracy_score(y_test, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
            "macro_f1": f1_score(y_test, y_pred, average="macro"),
            "weighted_f1": f1_score(y_test, y_pred, average="weighted"),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }

        print(json.dumps(result, indent=2))

        fold_results.append(result)

        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())
        all_y_proba.extend(y_proba.tolist())

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_proba = np.array(all_y_proba)

    print("\n" + "=" * 80)
    print(f"Classification report for {model_name}")
    print("=" * 80)
    print(classification_report(all_y_true, all_y_pred))

    print("Confusion matrix:")
    print(confusion_matrix(all_y_true, all_y_pred))

    print("Overall ROC-AUC:", roc_auc_score(all_y_true, all_y_proba))

    return fold_results


def main():
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    X_features, y, groups = load_or_build_features()

    print("\nLoaded feature dataset")
    print("X_features:", X_features.shape)
    print("y:", y.shape)
    print("groups:", groups.shape)
    print("Classes:", np.unique(y, return_counts=True))
    print("Subjects:", len(np.unique(groups)))

    all_results = []

    for model_name, model in get_models().items():
        print("\n" + "=" * 80)
        print("Training subject-independent model:", model_name)
        print("=" * 80)

        results = evaluate_model(
            model_name=model_name,
            model=model,
            X_features=X_features,
            y=y,
            groups=groups,
            n_splits=5,
        )

        all_results.extend(results)

    results_df = pd.DataFrame(all_results)

    results_path = reports_dir / "group_binary_cv_results.csv"
    summary_path = reports_dir / "group_binary_summary.csv"

    results_df.to_csv(results_path, index=False)

    metrics = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "roc_auc",
    ]

    summary = (
        results_df
        .groupby("model")[metrics]
        .agg(["mean", "std"])
    )

    summary.to_csv(summary_path)

    print("\nFinal subject-independent CV results:")
    print(results_df)

    print("\nSummary:")
    print(summary)

    print("\nSaved:")
    print("-", results_path)
    print("-", summary_path)


if __name__ == "__main__":
    main()