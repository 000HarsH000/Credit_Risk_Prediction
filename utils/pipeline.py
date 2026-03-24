# utils/pipeline.py

import joblib
import pandas as pd
import numpy as np
import shap

MODEL_PATH = "model/model.pkl"
SCALER_PATH = "model/scaler.pkl"
FEATURE_PATH = "model/features.pkl"
THRESHOLD_PATH = "model/threshold.pkl"

# Load once
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_columns = joblib.load(FEATURE_PATH)
threshold = joblib.load(THRESHOLD_PATH)

explainer = shap.TreeExplainer(model) 

def predict_from_dataframe(df):
    
    df=df.copy()
    df.replace(-99999, np.nan, inplace=True)

    cols_to_remove = [
        "num_dbt_6mts","num_lss_6mts","num_sub_12mts","num_sub_6mts",
        "num_sub","num_dbt","num_dbt_12mts","num_lss","num_lss_12mts"
    ]
    df.drop(columns=cols_to_remove, inplace=True)

    num_cols = df.select_dtypes(include=["int64","float64"]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # FEATURE ENGINEERING
    df["active_loan_ratio"] = df["Tot_Active_TL"] / (df["Total_TL"] + 1)
    df["loan_income_ratio"] = (df["Total_TL"]*100)/(df["NETMONTHLYINCOME"]+1)
    df["recent_enq_ratio"] = df["enq_L3m"]/(df["tot_enq"]+1)

    df["total_delinquency"] = df["num_deliq_6mts"] + df["num_deliq_12mts"]
    df["credit_activity_gap"] = df["Age_Oldest_TL"] - df["Age_Newest_TL"]

    # FLAG ENGINEERING
    flag_cols = ["HL_Flag","GL_Flag","PL_Flag","CC_Flag"]

    df["total_loan_flags"] = df[flag_cols].sum(axis=1)
    df["secured_loan_flags"] = df[["HL_Flag","GL_Flag"]].sum(axis=1)
    df["unsecured_loan_flags"] = df[["PL_Flag","CC_Flag"]].sum(axis=1)

    df.drop(columns=flag_cols, inplace=True)

    # ENCODING
    df["MARITALSTATUS"] = df["MARITALSTATUS"].map({"Single":0,"Married":1})
    df["GENDER"] = df["GENDER"].map({"M":1,"F":0})

    education_map = {
        "OTHERS":0,"SSC":1,"12TH":2,"UNDER GRADUATE":3,
        "GRADUATE":4,"POST-GRADUATE":5,"PROFESSIONAL":6
    }
    df["EDUCATION"] = df["EDUCATION"].map(education_map)

    df = pd.get_dummies(df, columns=["last_prod_enq2","first_prod_enq2"], drop_first=True)

    # ALIGN FEATURES
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_columns]

    # SCALING
    scale_cols = [
        "time_since_recent_payment",
        "NETMONTHLYINCOME",
        "Credit_Score"
    ]
    df[scale_cols] = scaler.transform(df[scale_cols])

    # PREDICT
    probs = model.predict_proba(df)
    probs = np.round(probs * 100, 2)

    # SHAP values
    shap_values = explainer(df)

    results = []

    for i in range(len(df)):

        prob = probs[i]
        risk_score = prob[3]  # P4

        # Applying threshold
        decision = "High Risk" if risk_score > threshold * 100 else "Low/Medium Risk"

        # SHAP explanation (P4)
        shap_row = shap_values.values[i, :, 3]

        feature_importance = sorted(
            zip(feature_columns, shap_row),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:3]

        top_features = [f[0] for f in feature_importance]

        results.append({
            "P1_prob": float(prob[0]),
            "P2_prob": float(prob[1]),
            "P3_prob": float(prob[2]),
            "P4_prob": float(prob[3]),
            "risk_category": decision,
            "top_features": top_features
})

    return results
