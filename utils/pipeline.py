# utils/pipeline.py

import joblib
import pandas as pd
import numpy as np

MODEL_PATH = "model/model.pkl"
SCALER_PATH = "model/scaler.pkl"
FEATURE_PATH = "model/features.pkl"

# Load once
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_columns = joblib.load(FEATURE_PATH)


def predict_from_dataframe(df):

    # ===== SAME STEPS AS YOUR predict.py =====
    
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

    return probs