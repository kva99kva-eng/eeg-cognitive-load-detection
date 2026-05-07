from pathlib import Path
import json
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.bandpower import build_feature_matrix


def load_binary_windows() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_path = PROJECT_ROOT / "data" / "processed" / "stew_kaggle_windows_binary.npz"

    if not data_path.exists():
        raise FileNotFoundError(
            "Missing data/processed/stew_kaggle_windows_binary.npz. "
            "Run scripts/prepare_stew_kaggle.py first."
        )

    data = np.load(data_path)
    return data["X"], data["y"], data["groups"]


def main() -> None:
    reports_dir = PROJECT_ROOT / "reports"
    models_dir = PROJECT_ROOT / "models"
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    X_windows, y, groups = load_binary_windows()
    print("Loaded windows:", X_windows.shape)

    X_features = build_feature_matrix(X_windows, sfreq=128)
    print("Feature matrix:", X_features.shape)

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=10,
                    min_samples_leaf=5,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    cv = GroupKFold(n_splits=min(5, len(np.unique(groups))))

    y_pred = cross_val_predict(
        model,
        X_features,
        y,
        cv=cv,
        groups=groups,
        method="predict",
    )

    y_prob = cross_val_predict(
        model,
        X_features,
        y,
        cv=cv,
        groups=groups,
        method="predict_proba",
    )[:, 1]

    metrics = {
        "validation": "subject_independent_group_kfold",
        "accuracy": float(accuracy_score(y, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
        "macro_f1": float(f1_score(y, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y, y_pred, average="weighted")),
        "roc_auc": float(roc_auc_score(y, y_prob)),
    }

    report = classification_report(
        y,
        y_pred,
        target_names=["low_load", "high_load"],
        output_dict=True,
        zero_division=0,
    )

    model.fit(X_features, y)
    joblib.dump(model, models_dir / "random_forest_subject_independent.joblib")

    pd.DataFrame([metrics]).to_csv(
        reports_dir / "group_baseline_binary_metrics.csv",
        index=False,
    )

    (reports_dir / "group_baseline_binary_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("Subject-independent baseline metrics:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
