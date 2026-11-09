
import streamlit as st
import pandas as pd


def render_data_quality(df):

    # ==========================================
    # 1. DATASET OVERVIEW
    # ==========================================

    st.header("3. Dataset Overview")

    st.markdown("### Dataset Performance Metrics")

    col1, col2, col3, col4 = st.columns(4)

    metrics = [
        ("📊", "Total Rows", f"{df.shape[0]:,}", "#22D3EE"),
        ("📋", "Total Columns", f"{df.shape[1]:,}", "#A78BFA"),
        ("⚠️", "Missing Values", f"{int(df.isna().sum().sum()):,}", "#FBBF24"),
        ("📋", "Duplicate Rows", f"{int(df.duplicated().sum()):,}", "#34D399")
    ]

    for col, metric in zip([col1, col2, col3, col4], metrics):

        with col:

            card_html = (
                f'<div style="background:linear-gradient(145deg,#172235,#1E293B);'
                f'border:1px solid #334155;border-radius:15px;padding:22px;min-height:135px;">'
                f'<div style="font-size:25px;">{metric[0]}</div>'
                f'<div style="color:#94A3B8;font-size:13px;margin-top:12px;">{metric[1]}</div>'
                f'<div style="color:{metric[3]};font-size:28px;font-weight:700;margin-top:5px;">{metric[2]}</div>'
                f'</div>'
            )

            st.markdown(card_html, unsafe_allow_html=True)

    # ==========================================
    # 2. COLUMN INFORMATION
    # ==========================================

    st.header("4. Column Information")

    st.markdown(
        "Explore data types, completeness, and uniqueness for every column."
    )

    column_info = pd.DataFrame({
        "Column": df.columns.tolist(),
        "Data Type": df.dtypes.astype(str).tolist(),
        "Non-Missing Values": df.notna().sum().tolist(),
        "Unique Values": df.nunique().tolist()
    })

    st.markdown("### Column Profiling")

    info_col1, info_col2, info_col3 = st.columns(3)

    with info_col1:
        st.metric("Total Columns", df.shape[1])

    with info_col2:
        st.metric("Numeric Columns", len(df.select_dtypes(include="number").columns))

    with info_col3:
        st.metric("Categorical Columns", len(df.select_dtypes(include=["object", "category", "string"]).columns))

    with st.container(border=True):
        st.dataframe(
            column_info,
            width="stretch",
            height=350,
            hide_index=True
        )

    # ==========================================
    # 3. MISSING VALUE ANALYSIS
    # ==========================================

    st.header("5. Missing Value Analysis")

    st.markdown(
        "Monitor dataset completeness and identify missing information."
    )

    missing_data = pd.DataFrame({
        "Column": df.columns,
        "Missing Values": df.isna().sum().values,
        "Missing Percentage": (
            df.isna().mean().values * 100
        ).round(2)
    })

    total_missing = int(df.isna().sum().sum())
    total_cells = df.shape[0] * df.shape[1]

    completeness = (
        ((total_cells - total_missing) / total_cells) * 100
        if total_cells > 0 else 100
    )

    missing_col1, missing_col2, missing_col3 = st.columns(3)

    with missing_col1:
        st.metric("Total Missing Values", f"{total_missing:,}")

    with missing_col2:
        st.metric("Data Completeness", f"{completeness:.2f}%")

    with missing_col3:
        affected_columns = int((df.isna().sum() > 0).sum())
        st.metric("Affected Columns", affected_columns)

    st.markdown("### Missing Value Details")

    with st.container(border=True):
        st.dataframe(
            missing_data,
            width="stretch",
            hide_index=True
        )

    if total_missing == 0:
        st.success("Excellent! Your dataset contains no missing values.")
    else:
        st.warning(
            f"Your dataset contains {total_missing:,} missing values. "
            "Consider cleaning these values before analysis."
        )

    # ==========================================
    # 4. DUPLICATE ANALYSIS
    # ==========================================

    st.header("6. Duplicate Analysis")

    st.markdown(
        "Identify repeated records and monitor dataset uniqueness."
    )

    duplicate_count = int(df.duplicated().sum())
    total_rows = len(df)

    duplicate_percentage = (
        (duplicate_count / total_rows) * 100
        if total_rows > 0 else 0
    )

    unique_rows = total_rows - duplicate_count

    duplicate_col1, duplicate_col2, duplicate_col3 = st.columns(3)

    with duplicate_col1:
        st.metric("Duplicate Rows", f"{duplicate_count:,}")

    with duplicate_col2:
        st.metric("Unique Rows", f"{unique_rows:,}")

    with duplicate_col3:
        st.metric("Duplicate Percentage", f"{duplicate_percentage:.2f}%")

    st.markdown("### Data Uniqueness Status")

    with st.container(border=True):

        if duplicate_count > 0:
            st.warning(
                f"Found {duplicate_count:,} duplicate rows in your dataset."
            )
            st.write(
                "Consider reviewing duplicate records before performing analysis."
            )
        else:
            st.success("Your dataset contains no duplicate rows.")
            st.write(
                "All records are unique according to the full-row comparison."
            )
