
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import StringIO

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="DataSphere",
    page_icon="📊",
    layout="wide"
)

st.title("📊 DataSphere")
st.subheader("Multi-Domain Data Analytics Platform")

st.write(
    "Explore datasets, analyze information, "
    "and discover meaningful insights."
)

st.sidebar.title("DataSphere Menu")
st.sidebar.info("Your Data Science workspace")


# ==========================================
# 2. DATASET UPLOAD
# ==========================================

st.header("1. Dataset Explorer")

uploaded_file = st.file_uploader(
    "Upload your CSV dataset",
    type=["csv"]
)

if uploaded_file is not None:

    # Safe CSV File Loading
    try:
        df = pd.read_csv(uploaded_file)

        if df.empty:
            st.error("The uploaded CSV file is empty.")
            st.stop()

        st.success("Dataset uploaded successfully!")

    except Exception as error:
        st.error(f"Unable to read the CSV file: {error}")
        st.stop()


    # ==========================================
    # 3. DATASET PREVIEW
    # ==========================================

    st.header("2. Dataset Preview")
    st.dataframe(df.head(10), width="stretch")


    # ==========================================
    # 4. DATASET OVERVIEW
    # ==========================================

    st.header("3. Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Rows", df.shape[0])

    with col2:
        st.metric("Total Columns", df.shape[1])

    with col3:
        st.metric("Missing Values", int(df.isna().sum().sum()))

    with col4:
        st.metric("Duplicate Rows", int(df.duplicated().sum()))


    # ==========================================
    # 5. COLUMN INFORMATION
    # ==========================================

    st.header("4. Column Information")

    column_info = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Non-Missing Values": df.notna().sum().values,
        "Unique Values": df.nunique().values
    })

    st.dataframe(column_info, width="stretch")


    # ==========================================
    # 6. MISSING VALUE ANALYSIS
    # ==========================================

    st.header("5. Missing Value Analysis")

    missing_data = pd.DataFrame({
        "Column": df.columns,
        "Missing Values": df.isna().sum().values,
        "Missing Percentage": (
            df.isna().mean().values * 100
        ).round(2)
    })

    st.dataframe(missing_data, width="stretch")


    # ==========================================
    # 7. DUPLICATE ANALYSIS
    # ==========================================

    st.header("6. Duplicate Analysis")

    duplicate_count = int(df.duplicated().sum())

    if duplicate_count > 0:
        st.warning(f"Found {duplicate_count} duplicate rows.")
    else:
        st.success("No duplicate rows found.")


    # ==========================================
    # 8. STATISTICAL SUMMARY
    # ==========================================

    st.header("7. Statistical Summary")

    st.dataframe(
        df.describe(include="all"),
        width="stretch"
    )


    # ==========================================
    # 9. UNIQUE VALUE ANALYSIS
    # ==========================================

    st.header("8. Unique Values")

    selected_column = st.selectbox(
        "Select a column",
        df.columns
    )

    st.write(
        df[selected_column].value_counts().head(20)
    )


    # ==========================================
    # 10. EXPLORATORY DATA ANALYSIS (EDA)
    # ==========================================

    st.header("9. Exploratory Data Analysis")

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    # Numerical Distribution

    if numeric_columns:

        st.subheader("Numerical Distribution")

        selected_numeric = st.selectbox(
            "Select numerical column for histogram",
            numeric_columns,
            key="histogram_column"
        )

        fig = px.histogram(
            df,
            x=selected_numeric,
            title=f"Distribution of {selected_numeric}"
        )

        st.plotly_chart(fig, width="stretch")

    # Categorical Distribution

    if categorical_columns:

        st.subheader("Categorical Distribution")

        selected_category = st.selectbox(
            "Select categorical column",
            categorical_columns,
            key="category_column"
        )

        category_counts = (
            df[selected_category]
            .value_counts()
            .reset_index()
        )

        category_counts.columns = ["Category", "Count"]

        fig = px.bar(
            category_counts,
            x="Category",
            y="Count",
            title=f"Distribution of {selected_category}"
        )

        st.plotly_chart(fig, width="stretch")

    # Relationship Analysis

    if len(numeric_columns) >= 2:

        st.subheader("Relationship Analysis")

        col1, col2 = st.columns(2)

        with col1:
            x_column = st.selectbox(
                "Select X-axis",
                numeric_columns,
                key="scatter_x"
            )

        with col2:
            y_column = st.selectbox(
                "Select Y-axis",
                numeric_columns,
                index=1,
                key="scatter_y"
            )

        fig = px.scatter(
            df,
            x=x_column,
            y=y_column,
            title=f"{x_column} vs {y_column}"
        )

        st.plotly_chart(fig, width="stretch")

    # Correlation Heatmap

    if len(numeric_columns) >= 2:

        st.subheader("Correlation Heatmap")

        correlation = df[numeric_columns].corr()

        fig = px.imshow(
            correlation,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            title="Numerical Feature Correlation"
        )

        st.plotly_chart(fig, width="stretch")


    # ==========================================
    # 11. BUSINESS INTELLIGENCE
    # ==========================================

    st.header("10. Business Intelligence")

    required_columns = [
        "Sales",
        "Profit",
        "Product",
        "Category",
        "Region"
    ]

    if all(col in df.columns for col in required_columns):

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Total Sales",
                f"{df['Sales'].sum():,.2f}"
            )

        with col2:
            st.metric(
                "Total Profit",
                f"{df['Profit'].sum():,.2f}"
            )

        with col3:
            st.metric(
                "Average Sales",
                f"{df['Sales'].mean():,.2f}"
            )

        # Category Analysis

        st.subheader("Sales by Category")

        category_sales = (
            df.groupby("Category")["Sales"]
            .sum()
            .reset_index()
            .sort_values("Sales", ascending=False)
        )

        fig = px.bar(
            category_sales,
            x="Category",
            y="Sales",
            title="Sales by Category",
            text_auto=".2s"
        )

        st.plotly_chart(fig, width="stretch")

        # Profit Analysis

        st.subheader("Profit by Category")

        category_profit = (
            df.groupby("Category")["Profit"]
            .sum()
            .reset_index()
        )

        fig = px.bar(
            category_profit,
            x="Category",
            y="Profit",
            title="Profit by Category"
        )

        st.plotly_chart(fig, width="stretch")

        # Regional Analysis

        st.subheader("Sales by Region")

        region_sales = (
            df.groupby("Region")["Sales"]
            .sum()
            .reset_index()
        )

        fig = px.pie(
            region_sales,
            names="Region",
            values="Sales",
            title="Regional Sales Distribution"
        )

        st.plotly_chart(fig, width="stretch")

        # Top Product

        top_product = (
            df.groupby("Product")["Sales"]
            .sum()
            .idxmax()
        )

        st.success(f"Top Selling Product: {top_product}")

    else:

        st.info(
            "Business Intelligence requires these columns: "
            "Sales, Profit, Product, Category and Region."
        )


    # ==========================================
    # 12. MULTI-DOMAIN ANALYTICS
    # ==========================================

    st.header("11. Multi-Domain Analytics")

    domain = st.selectbox(
        "Select your analysis domain",
        ["Business", "Education", "E-commerce", "Finance"]
    )

    domain_descriptions = {
        "Business": "Analyze business performance and trends.",
        "Education": "Analyze student performance and academic data.",
        "E-commerce": "Analyze product sales and customer behavior.",
        "Finance": "Analyze financial figures and transactions."
    }

    st.info(domain_descriptions[domain])

    st.subheader("Dynamic Data Analysis")

    if numeric_columns:

        metric_column = st.selectbox(
            "Select numerical variable",
            numeric_columns,
            key="domain_metric"
        )

        aggregation = st.selectbox(
            "Select calculation",
            ["Sum", "Mean", "Median", "Maximum", "Minimum", "Count"],
            key="domain_aggregation"
        )

        aggregation_functions = {
            "Sum": "sum",
            "Mean": "mean",
            "Median": "median",
            "Maximum": "max",
            "Minimum": "min",
            "Count": "count"
        }

        function = aggregation_functions[aggregation]

        if categorical_columns:

            category_column = st.selectbox(
                "Group analysis by",
                categorical_columns,
                key="domain_category"
            )

            grouped_data = (
                df.groupby(category_column)[metric_column]
                .agg(function)
                .reset_index()
            )

            grouped_data.columns = ["Category", "Value"]

            grouped_data = grouped_data.sort_values(
                "Value",
                ascending=False
            )

            st.subheader("Analysis Results")

            st.dataframe(
                grouped_data,
                width="stretch"
            )

            fig = px.bar(
                grouped_data,
                x="Category",
                y="Value",
                title=f"{aggregation} of {metric_column} by {category_column}",
                text_auto=".2s"
            )

            st.plotly_chart(fig, width="stretch")

            if not grouped_data.empty:

                top_category = grouped_data.iloc[0]["Category"]
                top_value = grouped_data.iloc[0]["Value"]

                st.success(
                    f"Highest {aggregation.lower()}: "
                    f"{top_category} ({top_value:,.2f})"
                )

        else:

            st.info(
                "No categorical columns found for grouped analysis."
            )

    else:

        st.warning(
            "This dataset does not contain numerical columns."
        )


    # ==========================================
    # 13. AUTOMATED ANALYTICS REPORT
    # ==========================================

    st.header("12. Download Analytics Report")

    report = StringIO()

    report.write("DATASPHERE ANALYTICS REPORT\n")
    report.write("=" * 40 + "\n\n")

    report.write("DATASET OVERVIEW\n")
    report.write(f"Total Rows: {df.shape[0]}\n")
    report.write(f"Total Columns: {df.shape[1]}\n")
    report.write(
        f"Missing Values: {int(df.isna().sum().sum())}\n"
    )
    report.write(
        f"Duplicate Rows: {int(df.duplicated().sum())}\n\n"
    )

    report.write("STATISTICAL SUMMARY\n")
    report.write(
        df.describe(include="all").to_string()
    )

    report.write("\n\nMISSING VALUE ANALYSIS\n")
    report.write(df.isna().sum().to_string())

    st.download_button(
        label="Download Analytics Report",
        data=report.getvalue(),
        file_name="datasphere_report.txt",
        mime="text/plain"
    )
    # ==========================================
    # 14. ADVANCED DATA CLEANING
    # ==========================================

    st.header("13. Advanced Data Cleaning")

    st.subheader("Missing Value Handling")

    # Display missing values
    missing_values = df.isnull().sum()

    missing_df = pd.DataFrame({
        "Column": missing_values.index,
        "Missing Values": missing_values.values
    })

    st.dataframe(missing_df, width="stretch")

    # Create a copy for cleaning
    cleaned_df = df.copy()

    if missing_values.sum() > 0:

        cleaning_method = st.selectbox(
            "Select Cleaning Method",
            [
                "Remove Rows with Missing Values",
                "Fill Numerical Values with Mean",
                "Fill Numerical Values with Median",
                "Fill Categorical Values with Mode"
            ]
        )

        if st.button("Apply Cleaning"):

            if cleaning_method == "Remove Rows with Missing Values":
                cleaned_df = cleaned_df.dropna()

            elif cleaning_method == "Fill Numerical Values with Mean":
                for column in cleaned_df.select_dtypes(include="number").columns:
                    cleaned_df[column] = cleaned_df[column].fillna(
                        cleaned_df[column].mean()
                    )

            elif cleaning_method == "Fill Numerical Values with Median":
                for column in cleaned_df.select_dtypes(include="number").columns:
                    cleaned_df[column] = cleaned_df[column].fillna(
                        cleaned_df[column].median()
                    )

            elif cleaning_method == "Fill Categorical Values with Mode":
                for column in cleaned_df.select_dtypes(
                    include=["object", "category", "str"]
                ).columns:
                    if not cleaned_df[column].mode().empty:
                        cleaned_df[column] = cleaned_df[column].fillna(
                            cleaned_df[column].mode()[0]
                        )

            st.success("Data cleaning completed!")

            st.subheader("Cleaned Dataset")
            st.dataframe(cleaned_df, width="stretch")

            csv_data = cleaned_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="Download Cleaned CSV",
                data=csv_data,
                file_name="cleaned_dataset.csv",
                mime="text/csv"
            )

    else:
        st.success("No missing values found in this dataset!")    

else:

    st.info(
        "Upload a CSV dataset to start exploring DataSphere."
    )
