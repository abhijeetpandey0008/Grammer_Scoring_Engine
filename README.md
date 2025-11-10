# 🧮 Grammar Scoring using Machine Learning

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LightGBM](https://img.shields.io/badge/LightGBM-Enabled-success)
![Kaggle](https://img.shields.io/badge/Platform-Kaggle-orange)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

A complete end-to-end **Grammar Scoring** pipeline built on **machine learning and audio signal processing**.  
This notebook implements advanced **feature engineering**, **LightGBM modeling**, and **ensemble learning** to predict grammar quality scores accurately.

---

## 📘 Project Overview

The goal of this project is to build an automated system that can **score grammar quality** (or language proficiency) using extracted features from audio/text data.

Key highlights:
- End-to-end ML workflow: preprocessing → feature extraction → modeling → evaluation  
- Custom feature engineering using **MFCC**, **Chroma**, **Spectral Contrast**, **Zero Crossing Rate (ZCR)**, and **Root Mean Square (RMS)**  
- Comparison between **classical ML models** and **hybrid CNN–LSTM architectures**  
- Optimized with **early stopping**, **cross-validation**, and **LightGBM v4 syntax**

---

## 🧩 Features

| Feature Type | Description |
|---------------|-------------|
| **MFCC** | Captures the timbral characteristics of audio |
| **Chroma** | Represents pitch class intensity |
| **Spectral Contrast** | Measures difference between peaks and valleys in spectrum |
| **ZCR** | Measures noisiness / frequency transitions |
| **RMS Energy** | Overall energy of the sound signal |

---

## ⚙️ Technologies Used

- 🐍 Python 3.10+
- 📦 NumPy, Pandas, Scikit-learn
- 🔥 LightGBM, XGBoost
- 🎵 Librosa for audio feature extraction
- 🧠 TensorFlow / Keras (for hybrid deep learning model)
- 📊 Matplotlib, Seaborn for visualization

---

## 🚀 Project Workflow

1. **Data Loading & Exploration**  
   Load audio/text data from Kaggle input directories.

2. **Feature Extraction**  
   Compute MFCC, Chroma, Spectral Contrast, ZCR, RMS features.

3. **Preprocessing**  
   - Handle missing values  
   - Standardize and scale features  
   - Merge labels with extracted features  

4. **Modeling**  
   - Train LightGBM and CNN-LSTM hybrid models  
   - Use K-Fold Cross Validation  
   - Apply early stopping and hyperparameter tuning  

5. **Evaluation**  
   Compare models based on RMSE and feature importance.

6. **Submission Generation**  
   Save final predictions as `submission_ensemble.csv` for Kaggle evaluation.

---

## 🧪 Results

| Model | Validation RMSE |
|--------|-----------------|
| Hybrid LightGBM | 0.6933 |
| CNN–LSTM | 0.5842 |
| **Ensemble (0.6H + 0.4C)** | ✅ Final Submission |

---


---

## 🧪 Results

| Model | Validation RMSE |
|--------|-----------------|
| LightGBM | 0.6933 |
| CNN–LSTM | 0.5842 |
| **Ensemble (0.6H + 0.4C)** | ✅ **Final Model** |

> The ensemble model achieved the **lowest RMSE**, outperforming individual baselines and producing stable predictions.

---

## 📸 Output Preview

Here’s a glimpse of the final ensemble predictions (`submission_ensemble.csv`):

```text
filename,label
test_001.wav,3.89
test_002.wav,4.12
test_003.wav,2.74
test_004.wav,4.55
test_005.wav,3.63
...



## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/grammar-scoring.git
cd grammar-scoring

# Install dependencies
pip install -r requirements.txt
