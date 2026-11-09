
import streamlit as st
import pandas as pd


def render_data_explorer():

    st.header("1. Dataset Explorer")

    uploaded_file = st.file_uploader(
        "Upload your CSV dataset",
        type=["csv"]
    )

    if uploaded_file is None:
        return None

    try:
        df = pd.read_csv(uploaded_file)

        if df.empty:
            st.error("The uploaded CSV file is empty.")
            return None

        st.success("Dataset uploaded successfully!")

        return df

    except Exception as error:
        st.error(f"Unable to read the CSV file: {error}")
        return None
