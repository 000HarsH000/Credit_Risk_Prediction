import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="Credit Risk Prediction", layout="wide")

st.title("Credit Risk Prediction System")
st.markdown("Upload both datasets (External + Internal) to generate predictions.")

# -------------------------
# SESSION STATE INIT
# -------------------------
if "result_json" not in st.session_state:
    st.session_state.result_json = None

# File uploaders
external_file = st.file_uploader("Upload External Data", type=["xlsx"])
internal_file = st.file_uploader("Upload Internal Data", type=["xlsx"])

if external_file and internal_file:

    # Preview data
    df1 = pd.read_excel(external_file)
    df2 = pd.read_excel(internal_file)

    st.subheader("External Data Preview")
    st.dataframe(df1.head())

    st.subheader("Internal Data Preview")
    st.dataframe(df2.head())

    # Validation
    if "PROSPECTID" not in df1.columns or "PROSPECTID" not in df2.columns:
        st.error("PROSPECTID column missing in one of the files")
    else:

        # -------------------------
        # PREDICT BUTTON
        # -------------------------
        if st.button("Predict"):

            with st.spinner("Running model..."):

                try:
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
                        st.session_state.result_json = response.json()
                    else:
                        st.error("Error from API")
                        st.write(response.text)

                except Exception as e:
                    st.error(f"Error: {e}")

        # -------------------------
        # DISPLAY RESULTS
        # -------------------------
        if st.session_state.result_json is not None:

            result_json = st.session_state.result_json
            result_df = pd.DataFrame(result_json)

            st.success("Prediction completed!")

            # -------------------------
            # TABLE OUTPUT
            # -------------------------
            st.subheader("Prediction Results")

            def highlight_risk(row):
                if row["risk_category"] == "High Risk":
                    return ["background-color: #ffcccc"] * len(row)
                return [""] * len(row)

            st.dataframe(result_df.style.apply(highlight_risk, axis=1))

            # -------------------------
            # INSPECT ANY ROW
            # -------------------------
            st.subheader("🔎 Inspect Any Prediction")

            selected_index = st.number_input(
                "Enter row index",
                min_value=0,
                max_value=len(result_json) - 1,
                value=0,
                step=1
            )

            selected = result_json[selected_index]

            st.write("**Risk Category:**", selected["risk_category"])
            st.write("**P4 Probability:**", selected["P4_prob"])

            st.write("**Top Features:**")
            for f in selected["top_features"]:
                st.write(f"• {f}")

            # -------------------------
            # DOWNLOAD
            # -------------------------
            csv = result_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Predictions",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv"
            )