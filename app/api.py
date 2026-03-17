import pandas as pd
from fastapi import FastAPI, UploadFile, File
from utils.pipeline import predict_from_dataframe
from io import BytesIO

app = FastAPI()


@app.post("/predict")
async def predict(
    external_file: UploadFile = File(...),
    internal_file: UploadFile = File(...)
):

    # Read both files
    external_bytes = await external_file.read()
    internal_bytes = await internal_file.read()

    df1 = pd.read_excel(BytesIO(external_bytes))
    df2 = pd.read_excel(BytesIO(internal_bytes))

    # Validate PROSPECTID
    if "PROSPECTID" not in df1.columns or "PROSPECTID" not in df2.columns:
        return {"error": "PROSPECTID column missing in one of the files"}

    # Merge
    df = pd.merge(df1, df2, on="PROSPECTID")

    prospect_ids = df["PROSPECTID"]
    df = df.drop(columns=["PROSPECTID"])

    # Predict
    probs = predict_from_dataframe(df)

    # Output
    result = pd.DataFrame({
        "PROSPECTID": prospect_ids,
        "P1_prob": probs[:, 0],
        "P2_prob": probs[:, 1],
        "P3_prob": probs[:, 2],
        "P4_prob": probs[:, 3]
    })

    return result.to_dict(orient="records")