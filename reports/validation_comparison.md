# Validation strategy comparison

This report compares two validation strategies on the Kaggle STEW binary workload dataset.

## Why this comparison matters

EEG windows from the same subject are often highly correlated. If windows from one subject appear in both train and test sets, the model may learn subject-specific patterns instead of general cognitive load patterns.

Therefore, subject-independent validation is more realistic for evaluating generalization to unseen people.

## Compared strategies

| Strategy | Description |
|---|---|
| Window-level CV | Random stratified split of EEG windows. Subjects can overlap between train and test. |
| Subject-independent CV | Grouped split by subject. Test subjects are unseen during training. |

## Results

| Validation | Model | Accuracy | Balanced Accuracy | Macro F1 | ROC-AUC |
|---|---|---:|---:|---:|---:|
| Subject-independent CV | LogisticRegression | 0.616 +/- 0.041 | 0.585 +/- 0.080 | 0.550 +/- 0.112 | 0.597 +/- 0.106 |
| Subject-independent CV | RandomForest | 0.594 +/- 0.077 | 0.580 +/- 0.099 | 0.567 +/- 0.110 | 0.628 +/- 0.142 |
| Window-level CV | LogisticRegression | 0.705 +/- 0.018 | 0.695 +/- 0.019 | 0.678 +/- 0.024 | 0.812 +/- 0.017 |
| Window-level CV | RandomForest | 0.949 +/- 0.003 | 0.948 +/- 0.003 | 0.949 +/- 0.003 | 0.990 +/- 0.001 |

## Interpretation

Window-level cross-validation gives much higher scores, especially for Random Forest.
However, this setup allows subject overlap between training and test folds.

Subject-independent validation is substantially harder and gives lower, more realistic performance.
This indicates that a large part of the apparent window-level performance is likely driven by subject-specific EEG patterns.

## Generated figures

- `reports/figures/validation_comparison_accuracy.png`
- `reports/figures/validation_comparison_balanced_accuracy.png`
- `reports/figures/validation_comparison_macro_f1.png`
- `reports/figures/validation_comparison_roc_auc.png`
