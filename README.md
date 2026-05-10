# EEG Cognitive Load Detection

[![Tests](https://github.com/kva99kva-eng/eeg-cognitive-load-detection/actions/workflows/tests.yml/badge.svg)](https://github.com/kva99kva-eng/eeg-cognitive-load-detection/actions/workflows/tests.yml)

EEG cognitive load detection project with classical machine learning baselines, subject-independent validation, CNN modeling, leakage analysis and a Streamlit demo.

This project is designed as a portfolio-ready neurotechnology / BCI case study. It focuses not only on model training, but also on validation strategy, leakage risk, feature interpretation and honest limitations.

## Executive Summary

This project demonstrates an end-to-end EEG cognitive load classification workflow.

The strongest part of the project is the validation design: I explicitly compare optimistic window-level validation with subject-independent validation, where test subjects are unseen during training.

Main contribution:

- built EEG windowing and spectral bandpower feature extraction;
- trained classical ML baselines on bandpower features;
- trained a CNN baseline on raw EEG windows;
- compared window-level and subject-independent validation;
- showed that window-level validation can strongly overestimate model performance;
- built a Streamlit demo for model inference;
- documented model limitations and next steps.

## Project Goal

The goal is to classify EEG windows into two cognitive load states:

- `0` — low cognitive load
- `1` — high cognitive load

The project compares two approaches:

- classical machine learning on spectral bandpower features;
- convolutional neural network modeling on raw EEG windows.

## Why This Project Matters

EEG classification can easily produce misleadingly high scores when windows from the same subject appear in both train and test sets.

This project explicitly compares:

- **window-level cross-validation** — optimistic validation where windows from the same subject can appear in both train and test folds;
- **subject-independent validation** — more realistic validation where test subjects are unseen during training.

The main finding is that window-level validation strongly overestimates performance. This suggests that part of the optimistic performance is driven by subject-specific EEG patterns rather than fully generalizable cognitive load markers.

## Datasets

The project uses two versions of the STEW EEG workload dataset.

### Hugging Face STEW

Used for the first window-level baseline.

Characteristics:

- 28,512 EEG windows
- 14 EEG channels
- 256 time points per window
- sampling rate: 128 Hz
- binary labels: low / high cognitive load
- balanced classes

Input tensor shape:

```text
X.shape == (28512, 14, 256)
y.shape == (28512,)
```

### Kaggle STEW MAT Dataset

Used for subject-independent validation.

Raw shape:

```text
dataset.mat: (14, 19200, 45)
class_012.mat: (45, 1)
```

This means:

- 14 EEG channels
- 19,200 time points per subject
- 45 subjects
- 128 Hz sampling rate
- 150 seconds per subject recording

The signals were split into EEG windows:

- window size: 256 samples, 2 seconds
- step size: 64 samples, 0.5 seconds

Binary version:

```text
X: (8019, 14, 256)
classes: 0 / 1
subjects: 27
```

## EEG Channels

The dataset contains 14 EEG channels:

```text
AF3, F7, F3, FC5, T7, P7, O1,
O2, P8, T8, FC6, F4, F8, AF4
```

## Methodology

The project pipeline includes:

1. EEG dataset loading
2. EEG windowing
3. Exploratory EEG visualization
4. Power spectral density analysis
5. Bandpower feature extraction
6. Classical ML baseline training
7. Window-level cross-validation
8. Subject-independent validation
9. Feature importance analysis
10. CNN training on raw EEG windows
11. Threshold tuning
12. Streamlit demo

## Feature Extraction

Spectral bandpower features were extracted for each EEG channel.

| Band | Frequency range |
|---|---:|
| Theta | 4–8 Hz |
| Alpha | 8–13 Hz |
| Beta | 13–30 Hz |
| Gamma | 30–45 Hz |

For each EEG window:

```text
14 channels x 4 frequency bands = 56 features
```

## Classical ML Models

Two classical ML baselines were evaluated:

- Logistic Regression
- Random Forest

Metrics:

- Accuracy
- Balanced accuracy
- Macro F1
- Weighted F1
- ROC AUC

## Window-Level vs Subject-Independent Validation

A key part of this project is comparing two validation strategies on the same Kaggle STEW binary dataset.

| Strategy | Description |
|---|---|
| Window-level CV | Random stratified split of EEG windows. The same subjects can appear in both train and test folds. |
| Subject-independent CV | Grouped split by subject. Test subjects are unseen during training. |

### Results

| Validation | Model | Accuracy | Balanced accuracy | Macro F1 | ROC AUC |
|---|---|---:|---:|---:|---:|
| Window-level CV | Logistic Regression | 0.705 ± 0.018 | 0.695 ± 0.019 | 0.678 ± 0.024 | 0.812 ± 0.017 |
| Window-level CV | Random Forest | 0.949 ± 0.003 | 0.948 ± 0.003 | 0.949 ± 0.003 | 0.990 ± 0.001 |
| Subject-independent CV | Logistic Regression | 0.616 ± 0.041 | 0.585 ± 0.080 | 0.550 ± 0.112 | 0.597 ± 0.106 |
| Subject-independent CV | Random Forest | 0.594 ± 0.077 | 0.580 ± 0.099 | 0.567 ± 0.110 | 0.628 ± 0.142 |

Window-level cross-validation gives much higher performance, especially for Random Forest. However, this setup allows subject overlap between train and test folds.

Subject-independent validation is substantially harder and gives lower, more realistic performance.

![Validation comparison](reports/figures/validation_comparison_balanced_accuracy.png)

## Classical ML vs CNN

A simple CNN was trained directly on raw EEG windows.

| Model | Input | Validation | Accuracy | Balanced accuracy | Macro F1 | ROC AUC |
|---|---|---|---:|---:|---:|---:|
| Logistic Regression | Bandpower features | Subject-independent 5-fold CV | 0.616 ± 0.041 | 0.585 ± 0.080 | 0.550 ± 0.112 | 0.597 ± 0.106 |
| Random Forest | Bandpower features | Subject-independent 5-fold CV | 0.594 ± 0.077 | 0.580 ± 0.099 | 0.567 ± 0.110 | 0.628 ± 0.142 |
| Simple CNN | Raw EEG windows | Subject-independent single split, threshold=0.5 | 0.626 | 0.548 | 0.501 | 0.742 |
| Simple CNN tuned threshold | Raw EEG windows | Subject-independent single split, threshold=0.35 | 0.622 | 0.537 | 0.472 | 0.742 |

The CNN achieved the best ROC AUC on the selected subject-independent split, but its balanced accuracy and macro F1 remained limited due to bias toward the high-load class.

Threshold tuning did not improve test-set balanced accuracy, so the default threshold result is kept as the main CNN baseline.

![ML vs CNN balanced accuracy](reports/figures/ml_vs_cnn_balanced_accuracy.png)

## Feature Importance

Random Forest feature importance was used to estimate which EEG spectral features contributed most to the classification.

Top features included:

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

The most important features were mainly located in frontal and temporal channels, especially in the theta band.

This analysis should be interpreted as model-level feature importance, not as a causal neuroscientific conclusion.

![Band importance](reports/figures/band_importance.png)

![Channel-band importance](reports/figures/channel_band_importance_heatmap.png)

## Streamlit Demo

The project includes an interactive Streamlit demo for EEG cognitive load prediction.

The demo allows the user to:

- select an EEG window;
- visualize the 14-channel EEG signal;
- inspect CNN predicted probabilities;
- adjust the decision threshold;
- compare predicted and true cognitive load labels.

Run locally:

```bash
streamlit run app/streamlit_app.py
```

Demo files:

- `app/streamlit_app.py`
- `app/demo_samples.npz`
- `models/eeg_cnn_subject_split_binary.pt`

## Project Structure

```text
eeg-cognitive-load-detection/
├── app/
│   ├── demo_samples.npz
│   └── streamlit_app.py
├── models/
│   └── eeg_cnn_subject_split_binary.pt
├── reports/
│   ├── figures/
│   ├── validation_comparison_summary.csv
│   ├── ml_vs_cnn_summary.csv
│   └── results.md
├── scripts/
├── src/
│   ├── data/
│   ├── features/
│   └── models/
├── .gitignore
├── README.md
└── requirements.txt
```

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```bash
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Reproduce Results

Prepare Kaggle STEW windows:

```bash
python scripts/prepare_stew_kaggle.py
```

Train subject-independent ML baselines:

```bash
python scripts/train_group_baseline_binary.py
```

Train window-level baselines:

```bash
python scripts/train_kaggle_window_baseline_binary.py
```

Compare validation strategies:

```bash
python scripts/compare_validation_strategies.py
```

Train CNN:

```bash
python scripts/train_cnn_subject_split_binary.py
```

Tune CNN threshold:

```bash
python scripts/tune_cnn_threshold.py
```

Compare ML and CNN:

```bash
python scripts/compare_ml_and_cnn.py
```

Run demo:

```bash
streamlit run app/streamlit_app.py
```

## Current Limitations

- The CNN was evaluated on a single subject-independent split, not full cross-validation.
- The dataset is relatively small for deep learning.
- The CNN requires better calibration and regularization.
- The current project focuses on offline classification, not real-time inference.
- Results are intended for educational and portfolio purposes, not clinical use.

## Next Steps

- Train CNN with subject-independent cross-validation.
- Add EEGNet-style architecture.
- Add probability calibration.
- Add more robust preprocessing.
- Improve the Streamlit demo.
- Add unit tests and CI checks.

## Tech Stack

- Python
- NumPy
- pandas
- SciPy
- scikit-learn
- PyTorch
- Matplotlib
- Streamlit
- Hugging Face Datasets

## Resume Summary

Built an EEG cognitive load detection pipeline using spectral bandpower features and raw EEG CNN modeling. Implemented subject-independent validation, leakage analysis, feature importance visualization, threshold tuning and an interactive Streamlit demo.

## Limitations

This project should be interpreted as a neurotechnology portfolio case study, not as a production-ready medical or BCI system.

Main limitations:

- the dataset is relatively small for robust EEG model generalization;
- EEG windows from the same subject can inflate performance under window-level validation;
- subject-independent validation is more realistic and should be treated as the primary evaluation setting;
- model performance may be affected by subject-specific EEG patterns, recording conditions and preprocessing choices;
- the CNN baseline is included as a methodological comparison, not as a final optimized architecture;
- the project does not provide clinical, diagnostic or medical conclusions.

The key methodological conclusion is that validation strategy matters more than the highest reported score. Window-level validation can look impressive, but subject-independent validation gives a more honest estimate of generalization to unseen users.


## License

This project is licensed under the MIT License.
