import joblib
import warnings
import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# PATH CONFIGURATION
DATA_PATH_1 = "data/train_external_data.xlsx"
DATA_PATH_2 = "data/train_internal_data.xlsx"

MODEL_PATH = "model/model.pkl"
SCALER_PATH = "model/scaler.pkl"
FEATURE_PATH = "model/features.pkl"


# LOAD DATA
def load_data(path_1, path_2):

    df_1 = pd.read_excel(path_1)
    df_2 = pd.read_excel(path_2)

    df = pd.merge(df_1, df_2, on="PROSPECTID")
    df = df.drop(columns=["PROSPECTID"])

    return df


# PREPROCESSING
def pre_processing(df):

    df.replace(-99999, np.nan, inplace=True)

    cols_to_remove = [
        "num_dbt_6mts",
        "num_lss_6mts",
        "num_sub_12mts",
        "num_sub_6mts",
        "num_sub",
        "num_dbt", 
        "num_dbt_12mts", 
        "num_lss",  
        "num_lss_12mts"
    ]
    df.drop(columns=cols_to_remove, inplace=True)

    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    return df


# FEATURE ENGINEERING
def create_feature(df):
    df["active_loan_ratio"] = df["Tot_Active_TL"] / df["Total_TL"]          
    df["loan_income_ratio"] = (df["Total_TL"]*100) / df["NETMONTHLYINCOME"]       
    df["recent_enq_ratio"] = df["enq_L3m"] / (df["tot_enq"] + 1)            
    df["total_delinquency"] = df["num_deliq_6mts"] + df["num_deliq_12mts"]  
    df["credit_activity_gap"] = df["Age_Oldest_TL"] - df["Age_Newest_TL"]   

    df["loan_income_ratio"].replace([np.inf, -np.inf], np.nan, inplace=True)
    df["loan_income_ratio"].fillna(df["loan_income_ratio"].median(), inplace=True)

    flag_cols = ["HL_Flag", "GL_Flag", "PL_Flag", "CC_Flag"]
    df["total_loan_flags"] = df[flag_cols].sum(axis=1)

    df["secured_loan_flags"] = df[["HL_Flag","GL_Flag"]].sum(axis=1)
    df["unsecured_loan_flags"] = df[["PL_Flag","CC_Flag"]].sum(axis=1)

    df.drop(columns=flag_cols, inplace=True)

    return df


# ENCODING
def encoding(df):

    df["MARITALSTATUS"] = df["MARITALSTATUS"].map({
        "Single": 0,
        "Married": 1
    })

    df["GENDER"] = df["GENDER"].map({
        "M": 1,
        "F": 0
    })

    education_map = {
        "OTHERS": 0,
        "SSC": 1,
        "12TH": 2,
        "UNDER GRADUATE": 3,
        "GRADUATE": 4,
        "POST-GRADUATE": 5,
        "PROFESSIONAL": 6
    }

    df["EDUCATION"] = df["EDUCATION"].map(education_map)

    df = pd.get_dummies(
        df,
        columns=["last_prod_enq2", "first_prod_enq2"],
        drop_first=True,
        dtype=int
    )

    target_map = {"P1": 0, "P2": 1, "P3": 2, "P4": 3}

    df["Approved_Flag"] = df["Approved_Flag"].map(target_map)

    return df


# SPLIT TARGET
def target_feature_split(df):

    y = df["Approved_Flag"]
    X = df.drop(columns=["Approved_Flag"])

    return X, y


# SCALING
def feature_scaling(X):

    scale_cols = [
        "time_since_recent_payment",
        "NETMONTHLYINCOME",
        "Credit_Score"
    ]

    scaler = StandardScaler()
    X[scale_cols] = scaler.fit_transform(X[scale_cols])

    return X, scaler


# MODEL TRAINING
def model_training(X, y):

    model = XGBClassifier(
        colsample_bytree=0.9,
        gamma=0,
        learning_rate=0.01,
        max_depth=4,
        min_child_weight=1,
        n_estimators=300,
        subsample=0.9,
        random_state=42,
        objective="multi:softprob",
        num_class=4
    )

    model.fit(X, y)

    return model


# SAVE MODEL
def save_artifacts(model, scaler, features):

    os.makedirs("model", exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(features, FEATURE_PATH)


# MAIN PIPELINE
def main():

    print("Loading data...")
    df_original = load_data(DATA_PATH_1, DATA_PATH_2)

    print("Pre-processing...")
    df_processed = pre_processing(df_original)

    print("Adding features...")
    df_featured = create_feature(df_processed)

    print("Encoding...")
    df_encoded = encoding(df_featured)

    print("Splitting features...")
    X, y = target_feature_split(df_encoded)

    print("Scaling...")
    X_scaled, scaler = feature_scaling(X)

    print("Training model...")
    model = model_training(X_scaled, y)

    print("Saving artifacts...")
    save_artifacts(model, scaler, X_scaled.columns)

    print("Training completed successfully!")


if __name__ == "__main__":
    main()