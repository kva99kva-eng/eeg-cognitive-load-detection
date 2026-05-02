# Baseline results

This report summarizes the first baseline for EEG cognitive load detection.

## Validation setup

- Features: spectral bandpower
- Frequency bands: theta, alpha, beta, gamma
- Validation: 5-fold stratified window-level cross-validation
- Models: Logistic Regression, Random Forest

> Note: this is a window-level baseline. Subject-independent validation should be added as the next stage when true subject IDs are available.

## Results

| Model | Accuracy | Balanced Accuracy | Macro F1 | Weighted F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| LogisticRegression | 0.715 ± 0.006 | 0.715 ± 0.006 | 0.714 ± 0.006 | 0.714 ± 0.006 | 0.783 ± 0.005 |
| RandomForest | 0.902 ± 0.003 | 0.902 ± 0.003 | 0.902 ± 0.003 | 0.902 ± 0.003 | 0.962 ± 0.003 |

## Current interpretation

Random Forest clearly outperforms Logistic Regression on bandpower features.
This suggests that the relationship between spectral EEG features and cognitive load is likely non-linear.

## Generated figures

- `reports/figures/baseline_metric_comparison.png`
- `reports/figures/random_forest_confusion_matrix.png`

## Feature importance

Random Forest feature importance was used to estimate which spectral EEG features contributed most to the window-level classification.

### Top 10 features

| Feature | Importance |
|---|---:|
| F8_theta | 0.07363 |
| FC6_theta | 0.04138 |
| F8_gamma | 0.03234 |
| F8_beta | 0.03210 |
| O2_theta | 0.03021 |
| T8_theta | 0.02829 |
| O1_theta | 0.02513 |
| F7_gamma | 0.02449 |
| O2_alpha | 0.02431 |
| F8_alpha | 0.02319 |

Generated figures:

- `reports/figures/top20_feature_importance.png`
- `reports/figures/channel_importance.png`
- `reports/figures/band_importance.png`
- `reports/figures/channel_band_importance_heatmap.png`

<!-- VALIDATION_COMPARISON_START -->

## Window-level vs subject-independent validation

A key part of this project is comparing two validation strategies on the same Kaggle STEW binary dataset.

EEG windows from the same subject are often highly correlated. If windows from the same person appear in both train and test sets, the model may learn subject-specific patterns instead of general cognitive load patterns.

For this reason, subject-independent validation is a more realistic test of generalization to unseen people.

### Validation strategies

| Strategy | Description |
|---|---|
| Window-level CV | Random stratified split of EEG windows. The same subjects can appear in both train and test. |
| Subject-independent CV | Grouped split by subject. Test subjects are unseen during training. |

### Results

| Validation | Model | Accuracy | Balanced Accuracy | Macro F1 | ROC-AUC |
|---|---|---:|---:|---:|---:|
| Window-level CV | LogisticRegression | 0.705 +/- 0.018 | 0.695 +/- 0.019 | 0.678 +/- 0.024 | 0.812 +/- 0.017 |
| Window-level CV | RandomForest | 0.949 +/- 0.003 | 0.948 +/- 0.003 | 0.949 +/- 0.003 | 0.990 +/- 0.001 |
| Subject-independent CV | LogisticRegression | 0.616 +/- 0.041 | 0.585 +/- 0.080 | 0.550 +/- 0.112 | 0.597 +/- 0.106 |
| Subject-independent CV | RandomForest | 0.594 +/- 0.077 | 0.580 +/- 0.099 | 0.567 +/- 0.110 | 0.628 +/- 0.142 |

### Interpretation

Window-level cross-validation gives much higher performance, especially for Random Forest.
However, this setup allows subject overlap between train and test folds.

Subject-independent validation is substantially harder and gives lower, more realistic performance.
This suggests that a large part of the window-level performance is likely driven by subject-specific EEG patterns rather than fully generalizable cognitive load markers.

### Generated figures

- `reports/figures/validation_comparison_accuracy.png`
- `reports/figures/validation_comparison_balanced_accuracy.png`
- `reports/figures/validation_comparison_macro_f1.png`
- `reports/figures/validation_comparison_roc_auc.png`

<!-- VALIDATION_COMPARISON_END -->
