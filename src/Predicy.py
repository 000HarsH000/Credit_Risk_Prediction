import joblib
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# PATHS
DATA_PATH_1 = "data/unknown_external_data.xlsx"
DATA_PATH_2 = "data/unknown_internal_data.xlsx"

MODEL_PATH = "model/model.pkl"
SCALER_PATH = "model/scaler.pkl"
FEATURE_PATH = "model/features.pkl"


# LOAD ARTIFACTS
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_columns = joblib.load(FEATURE_PATH)
    return model, scaler, feature_columns


# LOAD DATA
def load_data(path_1, path_2):
    df1 = pd.read_excel(path_1)
    df2 = pd.read_excel(path_2)

    df = pd.merge(df1, df2, on="PROSPECTID")
    prospect_ids = df["PROSPECTID"]
    df = df.drop(columns=["PROSPECTID"])

    return df, prospect_ids


# PREPROCESSING (same as train)
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


# FEATURE ENGINEERING (same as train)
def create_feature(df):

    df["active_loan_ratio"] = df["Tot_Active_TL"] / (df["Total_TL"] + 1)
    df["loan_income_ratio"] = (df["Total_TL"] * 100) / (df["NETMONTHLYINCOME"] + 1)
    df["recent_enq_ratio"] = df["enq_L3m"] / (df["tot_enq"] + 1)

    df["total_delinquency"] = df["num_deliq_6mts"] + df["num_deliq_12mts"]
    df["credit_activity_gap"] = df["Age_Oldest_TL"] - df["Age_Newest_TL"]

    df["loan_income_ratio"].replace([np.inf, -np.inf], np.nan, inplace=True)
    df["loan_income_ratio"].fillna(df["loan_income_ratio"].median(), inplace=True)

    # FLAG ENGINEERING
    flag_cols = ["HL_Flag", "GL_Flag", "PL_Flag", "CC_Flag"]

    df["total_loan_flags"] = df[flag_cols].sum(axis=1)
    df["secured_loan_flags"] = df[["HL_Flag", "GL_Flag"]].sum(axis=1)
    df["unsecured_loan_flags"] = df[["PL_Flag", "CC_Flag"]].sum(axis=1)

    df.drop(columns=flag_cols, inplace=True)

    return df


# ENCODING (same as train but NO target mapping)
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

    return df


# ALIGN FEATURES
def align_features(X, feature_columns):

    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0

    X = X[feature_columns]

    return X


# SCALING (same fixed columns)
def feature_scaling(X, scaler):

    scale_cols = [
        "time_since_recent_payment",
        "NETMONTHLYINCOME",
        "Credit_Score"
    ]

    X[scale_cols] = scaler.transform(X[scale_cols])

    return X


# PREDICT
def predict(model, X):

    probs = model.predict_proba(X)
    probs_percent = np.round(probs * 100, 2)

    return probs_percent


# MAIN
def main():

    print("Loading model...")
    model, scaler, feature_columns = load_artifacts()

    print("Loading data...")
    df, prospect_ids = load_data(DATA_PATH_1, DATA_PATH_2)

    print("Pre-processing...")
    df = pre_processing(df)

    print("Feature engineering...")
    df = create_feature(df)

    print("Encoding...")
    df = encoding(df)

    print("Aligning features...")
    df = align_features(df, feature_columns)

    print("Scaling...")
    df = feature_scaling(df, scaler)

    print("Predicting...")
    probs_percent = predict(model, df)

    result = pd.DataFrame({
        "PROSPECTID": prospect_ids,
        "P1_prob": probs_percent[:, 0],
        "P2_prob": probs_percent[:, 1],
        "P3_prob": probs_percent[:, 2],
        "P4_prob": probs_percent[:, 3]
    })

    print(result.head(10))


if __name__ == "__main__":
    main()