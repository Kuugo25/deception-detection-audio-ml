# Deception Detection from Audio using Machine Learning

This project explores the detection of deceptive speech using machine learning techniques. By extracting acoustic features from audio recordings, we train and evaluate multiple classification models to distinguish between truthful and deceptive narratives.

---

## Project Overview

- Classifies spoken stories as **truthful** or **deceptive** using extracted acoustic features
- Trained multiple ML models:  
  `Logistic Regression`, `k-Nearest Neighbors`, `Support Vector Machine`, `Decision Tree`, `Random Forest`, `Gradient Boosting`
- Utilized **GridSearchCV** for hyperparameter optimization
- Used **5-fold cross-validation** and built an ensemble model using **soft voting**
- Achieved improved performance through ensemble learning

---

## Objectives

- Understand which acoustic patterns correlate with deception
- Benchmark traditional ML classifiers on speech-based deception data
- Explore the use of ensemble models to enhance robustness

---

## Techniques Used

- **Acoustic Feature Engineering** using `librosa` (e.g., MFCCs, pitch, energy)
- **Supervised Learning Models** for binary classification
- **Cross-validation**, grid search, and ensemble voting
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-score, Confusion Matrix

---

## ⚠️ Audio Files

Raw audio files are not included in this repository due to file size constraints.

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/deception-detection-audio-ml.git
cd deception-detection-audio-ml
```

### 2. Install dependencies
pip install -r requirements.txt

### 3. Run the notebook
Open the notebook in notebooks/model_training.ipynb and run the cells to load the data, train models, and evaluate performance.


