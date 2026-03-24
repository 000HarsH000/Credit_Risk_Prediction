# Credit Risk Prediction System

An end-to-end Machine Learning system to predict credit risk categories using XGBoost, with explainability using SHAP and a full-stack deployment using FastAPI and Streamlit.

---

## Features

- ✅ Multi-class credit risk prediction (P1–P4)
- ✅ Probability-based outputs instead of hard labels
- ✅ Threshold tuning for business decision making
- ✅ SHAP explainability (feature-level insights)
- ✅ FastAPI backend for model serving
- ✅ Streamlit UI for interactive predictions
- ✅ Upload-based batch prediction system

---

## Problem Statement

Financial institutions need to assess the creditworthiness of applicants.  
This project predicts risk categories based on historical financial and behavioral features.

---

## Tech Stack

- **Machine Learning:** XGBoost, Scikit-learn  
- **Backend:** FastAPI  
- **Frontend:** Streamlit  
- **Explainability:** SHAP  
- **Data Processing:** Pandas, NumPy  

---

## Model Highlights

- Engineered domain-specific features:
  - Loan-to-income ratio  
  - Credit activity gap  
  - Delinquency aggregation  
- Applied:
  - Hyperparameter tuning (RandomizedSearchCV)
  - K-Fold Cross Validation
- Handled:
  - Class imbalance using weighted training
- Output:
  - Probability distribution across 4 risk classes

---

## Threshold Tuning

Instead of default classification, a custom threshold was applied:

- Tuned on validation data using F1-score
- Enables business-driven classification (High Risk vs Others)

---

## Explainability (SHAP)

- Integrated SHAP TreeExplainer
- Provides top contributing features per prediction
- Enhances model transparency for financial decisions

---

##  Dataset

- Data is taken from https://www.kaggle.com/datasets/saurabhbadole/leading-indian-bank-and-cibil-real-world-dataset
- External_Cibil_Dataset, Internal_Bank_Dataset, Unseen_Dataset are the original dataset downloaded fron the above link.
- unknown_external_dataset, unknown_internal_dataset are the 500 datas randomly taken from External_Cibil_Dataset and Internal_Bank_Dataset - respectively.
- train_external_dataset, train_internal_dataset are the remaining datapoints.
- Rest datasets are generated during preprocessing