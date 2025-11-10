<h1 align="center">Grammar Scoring Engine 🎙️</h1>

<p align="center">
  <strong>AI-Powered Assessment of Spoken Grammar Proficiency:<br>Audio Input + Transcript → Grammar Proficiency Score</strong>
</p>

<p align="center">
  <a href="https://github.com/abhijeetpandeygithub/grammar-scoring-engine/actions/workflows/ci.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/abhijeetpandeygithub/grammar-scoring-engine/ci.yml?style=for-the-badge&logo=github" alt="CI Status"/>
  </a>
  <img src="https://img.shields.io/badge/Status-Experimental-orange?style=for-the-badge&logo=appveyor" alt="Experimental Status"/>
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python Version"/>
  <img src="https://img.shields.io/badge/Jupyter-Compatible-4CAF50?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter Compatible"/>
  <img src="https://img.shields.io/badge/License-MIT-4CAF50?style=for-the-badge&logo=github&logoColor=white" alt="MIT License"/>
</p>

---

## 📌 Project Overview

The **Grammar Scoring Engine** is a machine learning pipeline for evaluating spoken grammar proficiency. It processes audio recordings and their transcripts to predict a numeric grammar score (on a 0–5 scale). The core implementation is in the Jupyter Notebook [`grammar-scoring-engine_final.ipynb`](grammar-scoring-engine_final.ipynb), which handles data loading, audio feature extraction, model training, ensemble strategies, and performance evaluation.

Key capabilities:
- **Audio Feature Extraction**: Utilizes `librosa` for acoustic features like MFCCs, spectral characteristics, tempo, RMS energy, and zero-crossing rate.
- **Transcript Analysis**: Computes linguistic metrics such as word count and average word length.
- **Modeling**: Employs regression models (Random Forest, XGBoost, LightGBM, Gradient Boosting) and ensemble techniques (stacking with Ridge meta-learner, blending).
- **Evaluation**: Focuses on RMSE, with an estimated accuracy metric derived from predictions.
- **Explainability**: Integrates SHAP for feature importance insights.

This README is derived directly from the notebook's experiments, ensuring alignment with actual code and results.

---

## 🔬 Notebook Structure

The notebook is organized into logical sections for reproducibility and clarity:

| Section | Key Components |
|---------|----------------|
| **Environment Setup & Data Loading** | Imports libraries (e.g., `numpy`, `pandas`, `librosa`); Lists and loads audio files from Kaggle-style paths (`/kaggle/input/...`); Handles train/test splits. |
| **Audio Preprocessing** | Trims silence using `librosa.effects.trim`; Loads audio with `librosa.load` at 16kHz sample rate. |
| **Feature Extraction** | - **Acoustic Features**: MFCC (mean/std/min/max/skew/kurtosis), spectral centroid/bandwidth/rolloff/contrast/flatness, ZCR, RMS, tempo.<br>- **Temporal Features**: Audio duration.<br>- **Transcript Features**: Word count, average word length.<br>- Feature engineering: Standardization with `StandardScaler`, polynomial features. |
| **Model Training & Ensembles** | - Single models: `RandomForestRegressor`, `XGBRegressor`, `LGBMRegressor`, `GradientBoostingRegressor`.<br>- Ensembles: Stacking (`StackingRegressor` with Ridge meta-learner), weighted blending.<br>- Hyperparameters: Tuned via trial runs (e.g., n_estimators=100 for RF). |
| **Evaluation & Visualization** | - Metrics: RMSE on validation set.<br>- Plots: Residuals (`matplotlib`, `seaborn`), feature importance (SHAP).<br>- Artifacts: Model saving with `joblib`. |
| **Inference & Deployment** | Sample code for predicting on new audio/transcript pairs. |

---

## 🧰 Technology Stack

| Category | Libraries/Tools |
|----------|-----------------|
| **Core & Data Handling** | `numpy`, `pandas`, `scikit-learn` (for preprocessing, metrics, ensembles) |
| **Audio Processing** | `librosa` (feature extraction), `soundfile` (audio I/O) |
| **Modeling** | `xgboost` (`XGBRegressor`), `lightgbm` (`LGBMRegressor`), `sklearn.ensemble` (`RandomForestRegressor`, `GradientBoostingRegressor`, `StackingRegressor`) |
| **Explainability** | `shap` (SHAP values for interpretability) |
| **Visualization** | `matplotlib`, `seaborn` (residual plots, feature importance) |
| **Utilities** | `joblib` (model serialization), `tqdm` (progress bars) |
| **Environment** | Python 3.11+ (notebook tested on Kaggle with GPU acceleration); Optional: `torch` for advanced extensions. |

**Note**: No external API calls or internet access required during runtime (Kaggle offline mode compatible).

---

## ✅ Experimental Results & Metrics

Experiments in the notebook include multiple model runs, hyperparameter variations, and ensemble comparisons. The final stacked ensemble (using Random Forest, XGBoost, LightGBM, and Gradient Boosting as base learners with a Ridge meta-learner) was selected for its balance of performance and robustness.

| Model / Configuration | RMSE (Validation) | Estimated Accuracy* (%) |
|-----------------------|-------------------|--------------------------|
| **Random Forest (Best Single Run)** | **0.6833** | **86.08** |
| Random Forest (Alternate Run) | 0.6853 | — |
| XGBoost (Various Runs) | 0.7113 – 0.7164 | — |
| LightGBM (Various Runs) | 0.7123 – 0.7186 | — |
| Gradient Boosting | 0.7253 | — |
| Blended Ensembles (Weighted) | 0.6931 – 0.6958 | — |
| **Stacked Ensemble (Final)** | **0.7110** | **85.78** |
| Validation Snapshot (Holdout) | 0.7926 | 84.15 |

> **Estimated Accuracy Formula** (assuming 0–5 score scale):
> $$
> \text{Estimated Accuracy} = \left(1 - \frac{\text{RMSE}}{5}\right) \times 100
> $$

### Insights from Experiments
- **Best Performer**: Random Forest excelled in single-model scenarios due to its handling of feature interactions.
- **Ensemble Benefits**: Stacking reduced overfitting and improved generalization, as evidenced by lower variance in residuals.
- **Feature Importance (via SHAP)**: MFCC statistics (e.g., mean, std), tempo, and RMS energy were top predictors, indicating prosody's role in grammar assessment.
- **Warnings & Notes**: LightGBM produced "no further splits" warnings, typical for small datasets or leaf-wise growth; no impact on final performance.

---

## 🧩 Reproducibility & Setup Guide

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/abhijeetpandeygithub/grammar-scoring-engine.git
   cd grammar-scoring-engine
