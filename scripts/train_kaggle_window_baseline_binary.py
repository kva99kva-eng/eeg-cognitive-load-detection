from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.bandpower import build_feature_matrix


def main() -> None:
    data_path = PROJECT_ROOT / "data" / "processed" / "stew_kaggle_windows_binary.npz"
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(
            "Missing data/processed/stew_kaggle_windows_binary.npz. "
            "Run scripts/prepare_stew_kaggle.py first."
        )

    data = np.load(data_path)
    X_windows = data["X"]
    y = data["y"]

    X_features = build_feature_matrix(X_windows, sfreq=128)

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

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    y_pred = cross_val_predict(
        model,
        X_features,
        y,
        cv=cv,
        method="predict",
    )

    y_prob = cross_val_predict(
        model,
        X_features,
        y,
        cv=cv,
        method="predict_proba",
    )[:, 1]

    metrics = {
        "validation": "window_level_stratified_kfold",
        "accuracy": float(accuracy_score(y, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
        "macro_f1": float(f1_score(y, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y, y_pred, average="weighted")),
        "roc_auc": float(roc_auc_score(y, y_prob)),
    }

    pd.DataFrame([metrics]).to_csv(
        reports_dir / "window_baseline_binary_metrics.csv",
        index=False,
    )

    (reports_dir / "window_baseline_binary_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print("Window-level baseline metrics:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
