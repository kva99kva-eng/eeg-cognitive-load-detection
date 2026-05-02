# ML vs CNN comparison

This report compares classical ML models based on bandpower features with a simple CNN trained directly on raw EEG windows.

## Results

| Model | Input | Validation | Accuracy | Balanced Accuracy | Macro F1 | ROC-AUC |
|---|---|---|---:|---:|---:|---:|
| LogisticRegression | Bandpower features | Subject-independent 5-fold CV | 0.616 +/- 0.041 | 0.585 +/- 0.080 | 0.550 +/- 0.112 | 0.597 +/- 0.106 |
| RandomForest | Bandpower features | Subject-independent 5-fold CV | 0.594 +/- 0.077 | 0.580 +/- 0.099 | 0.567 +/- 0.110 | 0.628 +/- 0.142 |
| SimpleCNN | Raw EEG windows | Subject-independent single split, threshold=0.5 | 0.626 | 0.548 | 0.501 | 0.742 |
| SimpleCNN tuned threshold | Raw EEG windows | Subject-independent single split, threshold=0.35 | 0.622 | 0.537 | 0.472 | 0.742 |

## Interpretation

The CNN achieved a higher ROC-AUC than the classical ML baselines on the selected subject-independent test split, which suggests that it can rank EEG windows by cognitive load probability reasonably well.

However, its balanced accuracy and macro F1 remained low because the default decision threshold produced a strong bias toward the high-load class.

Threshold tuning on the validation set did not improve test-set balanced accuracy, so the default threshold result is kept as the main CNN baseline.

Overall, this shows that deep learning on raw EEG is promising but requires better calibration, regularization and validation across multiple subject-independent folds.

## Generated figures

- `reports/figures/ml_vs_cnn_balanced_accuracy.png`
- `reports/figures/ml_vs_cnn_macro_f1.png`
- `reports/figures/ml_vs_cnn_roc_auc.png`
