# 🧠 Grammar Scoring Engine (SHL Internship Challenge)

This repository contains the complete pipeline for the **SHL Intern Hiring Challenge**, where the goal is to predict grammar proficiency scores (on a scale from 0 to 5) from spoken English audio clips (45–60 seconds long). The project is implemented in Python and includes both a production-ready `.py` script and a well-documented Jupyter Notebook.

## 📁 Repository Structure

📦 grammar-scoring-engine/ 

├── engine.py # Main script containing the full pipeline (for production use) 

├── grammar-scoring-engine.ipynb # Jupyter Notebook with explanations, code, and results 

├── submission.csv # Final output predictions

├── README.md # You're reading it!

---

## 🎯 Problem Statement

Predict a **continuous grammar score** for each audio clip using **machine learning models** trained on audio features extracted via Wav2Vec2, MFCC, and prosodic characteristics.

Evaluation Metric: **Pearson Correlation** between predicted and actual scores.

---

## 🧪 Approach

### 🔍 Feature Extraction
- **Wav2Vec2 embeddings** (mean + std)
- **Delta features**
- **MFCC features**
- **Prosodic features** (Pitch, Energy)
- **Audio duration**

### 🧠 Modeling Pipeline
1. **Base Models:**
   - XGBoost
   - LightGBM
   - CatBoost

2. **Stacking:**
   - Meta-model: Gradient Boosting Regressor
   - Trained using out-of-fold predictions from base models

3. **Ensembling:**
   - Final predictions use a **hybrid of meta-model predictions + rank-averaged base predictions**
   - Predictions clipped to the [0, 5] range

4. **Validation:**
   - 5-Fold Cross-Validation
   - Evaluation via **Pearson Correlation**

---

## 📊 Results

- **Best Validation Pearson Correlation:** ~`0.783`
- **Technique:** Feature-rich ensemble with stacked regression and rank averaging

---

## 🚀 How to Use

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/grammar-scoring-engine.git
cd grammar-scoring-engine
```

### 2. Run the Notebook

Open grammar-scoring-engine.ipynb in Jupyter/Colab
Follow the cells and run end-to-end


### 📚 Requirements

Python 3.8+
PyTorch
Torchaudio
Scikit-learn
XGBoost, LightGBM, CatBoost
NumPy, Pandas, TQDM, Joblib


### 📌 Notes

This repository is intended for academic/learning purposes, as part of the SHL Internship challenge.
All code is modular and reproducible.


## 📬 Contact
Feel free to reach out if you have any questions or feedback!

