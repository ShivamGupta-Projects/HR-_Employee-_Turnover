import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from pipeline_hr import run_pipeline

if "result" not in st.session_state:
    st.session_state["result"] = None
    st.session_state["df"] = None
    st.session_state["target_col"] = None

st.title("Employee Retention")

st.warning("""

This application is designed for HR Employee Retention datasets.

Ensure the following before uploading:

- CSV file format only
- Target column must be binary (0/1)
- Minimum 20 records required
- No invalid or mixed data types
- Column names should be consistent

Limitations:
- The model assumes schema consistency
- Unexpected columns or formats may cause errors

For best results, use structured HR datasets similar to training data.
""")

file = st.file_uploader("Upload CSV File")

if file is not None:

    st.write("File Uploaded")

    if not file.name.endswith(".csv"):
        st.error("Please upload a CSV file")
        st.stop()

    st.write("CSV Check Passed")

    try:
        df = pd.read_csv(file)
    except:
        st.error("Invalid CSV file")
        st.stop()

    st.write("Dataset Preview")
    st.dataframe(df.head())

    target_col = st.selectbox("Select Target Column", df.columns)

    if st.button("Run Pipeline"):

        if not pd.api.types.is_numeric_dtype(df[target_col]):
            st.warning("Target column must contain numeric values (0 and 1 only)")
            st.stop()

        if df[target_col].nunique() != 2:
            st.error("Target column must have 2 unique values.")
            st.stop()

        result = run_pipeline(df, target_col)

        st.session_state["result"] = result
        st.session_state["df"] = df
        st.session_state["target_col"] = target_col

if st.session_state["result"] is not None:

    result = st.session_state["result"]
    df = st.session_state["df"]
    target_col = st.session_state["target_col"]

    st.success("Pipeline executed successfully")

    st.write("Target Column", target_col)

    st.subheader("Data Shape")
    st.write(df.shape)

    st.subheader("Missing Values")
    st.dataframe(df.isna().sum())

    st.subheader("Best Model")
    st.write(result["best_model_name"].replace("_", " ").title())

    st.subheader("Accuracy")
    st.write(round(result["best_accuracy"], 4))

    st.subheader("Classification Report")
    report_df = pd.DataFrame(result["report"]).transpose()
    st.dataframe(report_df)

    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(result["cm"], annot=True, fmt='d', cmap="Blues", ax=ax)
    st.pyplot(fig)

    zone_df = result["full_zone_df"]

    st.subheader("Employee Risk Distribution")
    zone_counts = zone_df["Zone"].value_counts()

    for zone, count in zone_counts.items():
        st.write(f"{zone}: {count} employees")

    st.subheader("Employee Zone Details")

    selected_zone = st.selectbox(
        "Select Zone",
        zone_df["Zone"].unique(),
        key="zone_select"
    )

    filtered = zone_df[zone_df["Zone"] == selected_zone]
    st.dataframe(filtered)

    st.subheader("Check Single Employee")

    input_data = {}

    for col in df.columns:
        if col == target_col:
            continue
        input_data[col] = st.text_input(f"Enter {col}")

    if st.button("Predict Employee Risk"):

        input_df = pd.DataFrame([input_data])

        for col in input_df.columns:
            try:
                input_df[col] = pd.to_numeric(input_df[col])
            except:
                pass

        input_df = pd.get_dummies(input_df)

        input_df = input_df.reindex(columns=result["columns"], fill_value=0)

        prob = result["best_model"].predict_proba(input_df)[0][1]

        if prob <= 0.3:
            zone = "Safe Zone"
        elif prob <= 0.7:
            zone = "Medium Risk Zone"
        else:
            zone = "High Risk Zone"

        st.write(f"Risk Probability: {round(prob, 2)}")
        st.write(f"Zone: {zone}")