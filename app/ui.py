import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="Credit Risk Prediction", layout="wide")

st.title("Credit Risk Prediction System")

st.markdown("Upload both datasets (External + Internal) to generate predictions.")

# File uploaders
external_file = st.file_uploader("Upload External Data", type=["xlsx"])
internal_file = st.file_uploader("Upload Internal Data", type=["xlsx"])

if external_file and internal_file:

    # Preview data
    df1 = pd.read_excel(external_file)
    df2 = pd.read_excel(internal_file)

    st.subheader("External Data Preview")
    st.write(df1.head())

    st.subheader("Internal Data Preview")
    st.write(df2.head())

    # Validation
    if "PROSPECTID" not in df1.columns or "PROSPECTID" not in df2.columns:
        st.error("PROSPECTID column missing in one of the files")
    else:

        if st.button("Predict"):

            with st.spinner("Running model..."):

                try:
                    #FIX: send correct file format
                    files = {
                        "external_file": (
                            external_file.name,
                            external_file.getvalue(),
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        ),
                        "internal_file": (
                            internal_file.name,
                            internal_file.getvalue(),
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    }

                    response = requests.post(
                        "http://127.0.0.1:8000/predict",
                        files=files
                    )

                    if response.status_code == 200:

                        result = pd.DataFrame(response.json())

                        st.success("Prediction completed!")

                        st.subheader("Prediction Results")
                        st.write(result)

                        # Download button
                        csv = result.to_csv(index=False).encode("utf-8")

                        st.download_button(
                            label="Download Predictions",
                            data=csv,
                            file_name="predictions.csv",
                            mime="text/csv"
                        )

                    else:
                        st.error("Error from API")
                        st.write(response.text)

                except Exception as e:
                    st.error(f"Error: {e}")