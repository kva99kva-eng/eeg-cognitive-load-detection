from pathlib import Path
import sys
import json

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
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.features.bandpower import build_feature_matrix


RANDOM_STATE = 42


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
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "SVM_RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                probability=True,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ]),
    }


def evaluate_model_cv(model_name, model, X_features, y, groups, n_splits=5):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_results = []
    all_y_true = []
    all_y_pred = []
    all_y_proba = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X_features, y)):
        X_train = X_features[train_idx]
        X_test = X_features[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = y_pred

        result = {
            "model": model_name,
            "fold": fold,
            "accuracy": accuracy_score(y_test, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
            "macro_f1": f1_score(y_test, y_pred, average="macro"),
            "weighted_f1": f1_score(y_test, y_pred, average="weighted"),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }

        fold_results.append(result)

        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())
        all_y_proba.extend(y_proba.tolist())

        print(f"\n{model_name} | Fold {fold}")
        print(json.dumps(result, indent=2))

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    print(f"\nClassification report for {model_name}:")
    print(classification_report(all_y_true, all_y_pred))

    print(f"Confusion matrix for {model_name}:")
    print(confusion_matrix(all_y_true, all_y_pred))

    return fold_results


def main():
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    data = np.load("data/processed/stew_windows.npz")

    X = data["X"]
    y = data["y"]
    groups = data["groups"]

    print("Loaded data")
    print("X:", X.shape)
    print("y:", y.shape)
    print("groups:", groups.shape)
    print("Classes:", np.unique(y, return_counts=True))
    print("Subjects:", len(np.unique(groups)))

    print("\nExtracting bandpower features...")
    X_features = build_feature_matrix(X, sfreq=128)

    print("X_features:", X_features.shape)

    np.savez_compressed(
        "data/processed/stew_bandpower_features.npz",
        X_features=X_features,
        y=y,
        groups=groups
    )

    print("Saved features to data/processed/stew_bandpower_features.npz")

    all_results = []

    models = get_models()

    for model_name, model in models.items():
        print("\n" + "=" * 80)
        print("Training:", model_name)
        print("=" * 80)

        results = evaluate_model_cv(
            model_name=model_name,
            model=model,
            X_features=X_features,
            y=y,
            groups=groups,
            n_splits=5,
        )

        all_results.extend(results)

    results_df = pd.DataFrame(all_results)

    results_path = reports_dir / "baseline_cv_results.csv"
    results_df.to_csv(results_path, index=False)

    summary = (
        results_df
        .groupby("model")
        [["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "roc_auc"]]
        .agg(["mean", "std"])
    )

    summary_path = reports_dir / "baseline_summary.csv"
    summary.to_csv(summary_path)

    print("\nFinal CV results:")
    print(results_df)

    print("\nSummary:")
    print(summary)

    print("\nSaved:")
    print("-", results_path)
    print("-", summary_path)


if __name__ == "__main__":
    main()