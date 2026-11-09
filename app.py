
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import StringIO

from modules.data_explorer import render_data_explorer
from modules.data_quality import render_data_quality

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="DataSphere",
    page_icon="📊",
    layout="wide"
)

# ==========================================
# LOAD DATASPHERE STYLES
# ==========================================

with open("assets/style.css", "r", encoding="utf-8") as css_file:
    st.markdown(
        f"<style>{css_file.read()}</style>",
        unsafe_allow_html=True
    )
st.sidebar.markdown("""
<div style="padding: 15px 5px 25px 5px;">

<h2 style="color: #67E8F9; font-size: 25px;">
◈ DataSphere
</h2>

<p style="color: #94A3B8; font-size: 12px; letter-spacing: 2px;">
ANALYTICS WORKSPACE
</p>

<hr style="border-color: #263449;">

<p style="color: #CBD5E1; font-size: 14px;">
Your intelligent data analysis environment.
</p>

</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("### WORKSPACE")
st.sidebar.caption("Explore · Analyze · Visualize")

# ==========================================
# 2. DATASET UPLOAD
# ==========================================

df = render_data_explorer()

if df is not None:

        # ==========================================
    # 3. DATASET PREVIEW
    # ==========================================

    st.header("2. Dataset Preview")

    st.markdown(
        "Preview the first 10 records from your uploaded dataset."
    )

    preview_col1, preview_col2 = st.columns([3, 1])

    with preview_col1:
        st.markdown("### Data Explorer")

    with preview_col2:
        st.caption(f"Showing {min(len(df), 10)} of {len(df):,} rows")

    with st.container(border=True):
        st.dataframe(
            df.head(10),
            width="stretch",
            height=350
        )

    st.caption(
        f"Dataset contains {df.shape[0]:,} rows and {df.shape[1]:,} columns."
    )


    # ==========================================
    # DATA QUALITY ANALYSIS
    # ==========================================

    render_data_quality(df)


    # ==========================================
    # 8. STATISTICAL SUMMARY
    # ==========================================
  

    st.header("7. Statistical Summary")

    st.markdown(
        "Understand the distribution, central tendency, and spread of your dataset."
    )

    statistical_summary = df.describe(include="all").T.reset_index()
    statistical_summary.rename(
        columns={"index": "Column"},
        inplace=True
    )

    numeric_columns = df.select_dtypes(include="number").columns

    stat_col1, stat_col2, stat_col3 = st.columns(3)

    with stat_col1:
        st.metric("Total Features", df.shape[1])

    with stat_col2:
        st.metric("Numeric Features", len(numeric_columns))

    with stat_col3:
        st.metric(
            "Categorical Features",
            df.shape[1] - len(numeric_columns)
        )

    st.markdown("### Descriptive Statistics")

    with st.container(border=True):
        st.dataframe(
            statistical_summary,
            width="stretch",
            hide_index=True
        )


        # ==========================================
    # 9. UNIQUE VALUE ANALYSIS
    # ==========================================

    st.header("8. Unique Values")

    st.markdown(
        "Explore distinct values and understand how frequently they appear."
    )

    selected_column = st.selectbox(
        "Select a column",
        df.columns,
        key="unique_values_column"
    )

    value_counts = df[selected_column].value_counts().head(20)

    unique_count = int(df[selected_column].nunique())
    most_frequent = (
        str(value_counts.index[0])
        if not value_counts.empty else "N/A"
    )

    frequency_col1, frequency_col2, frequency_col3 = st.columns(3)

    with frequency_col1:
        st.metric("Unique Values", f"{unique_count:,}")

    with frequency_col2:
        st.metric("Displayed Values", len(value_counts))

    with frequency_col3:
        st.metric("Most Frequent Value", most_frequent)

    st.markdown("### Value Frequency Distribution")

    frequency_data = value_counts.rename_axis(
        "Value"
    ).reset_index(name="Frequency")

    with st.container(border=True):
        st.dataframe(
            frequency_data,
            width="stretch",
            hide_index=True
        )
        # ==========================================
    # 10. EXPLORATORY DATA ANALYSIS (EDA)
    # ==========================================

    st.header("9. Exploratory Data Analysis")

    st.markdown(
        "Discover patterns, distributions, relationships, and correlations in your data."
    )

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    # EDA Overview

    eda_col1, eda_col2, eda_col3 = st.columns(3)

    with eda_col1:
        st.metric("Numerical Features", len(numeric_columns))

    with eda_col2:
        st.metric("Categorical Features", len(categorical_columns))

    with eda_col3:
        st.metric(
            "Total Features Analyzed",
            len(numeric_columns) + len(categorical_columns)
        )

    st.divider()

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
            title=f"Distribution of {selected_numeric}",
            color_discrete_sequence=["#22D3EE"]
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            title_x=0.02
        )

        with st.container(border=True):
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
            title=f"Distribution of {selected_category}",
            color="Category",
            color_discrete_sequence=px.colors.qualitative.Bold
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            title_x=0.02,
            showlegend=False
        )

        with st.container(border=True):
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
            title=f"{x_column} vs {y_column}",
            color_discrete_sequence=["#A78BFA"]
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            title_x=0.02
        )

        with st.container(border=True):
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

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            title_x=0.02
        )

        with st.container(border=True):
            st.plotly_chart(fig, width="stretch")
        # ==========================================
    # 11. BUSINESS INTELLIGENCE
    # ==========================================

    st.header("10. Business Intelligence")

    st.markdown(
        "Analyze business performance, profitability, products, and regional sales."
    )

    required_columns = [
        "Sales",
        "Profit",
        "Product",
        "Category",
        "Region"
    ]

    if all(col in df.columns for col in required_columns):

        # Business KPI Cards

        st.markdown("### Business Performance")

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

        st.divider()

        # Category Analysis

        st.subheader("Category Performance")

        category_col1, category_col2 = st.columns(2)

        category_sales = (
            df.groupby("Category")["Sales"]
            .sum()
            .reset_index()
            .sort_values("Sales", ascending=False)
        )

        category_profit = (
            df.groupby("Category")["Profit"]
            .sum()
            .reset_index()
            .sort_values("Profit", ascending=False)
        )

        with category_col1:

            fig = px.bar(
                category_sales,
                x="Category",
                y="Sales",
                title="Sales by Category",
                text_auto=".2s",
                color="Category",
                color_discrete_sequence=px.colors.qualitative.Bold
            )

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False
            )

            with st.container(border=True):
                st.plotly_chart(fig, width="stretch")

        with category_col2:

            fig = px.bar(
                category_profit,
                x="Category",
                y="Profit",
                title="Profit by Category",
                text_auto=".2s",
                color="Category",
                color_discrete_sequence=px.colors.qualitative.Bold
            )

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False
            )

            with st.container(border=True):
                st.plotly_chart(fig, width="stretch")

        st.divider()

        # Regional Analysis

        st.subheader("Regional Sales Intelligence")

        region_sales = (
            df.groupby("Region")["Sales"]
            .sum()
            .reset_index()
            .sort_values("Sales", ascending=False)
        )

        region_col1, region_col2 = st.columns(2)

        with region_col1:

            fig = px.pie(
                region_sales,
                names="Region",
                values="Sales",
                title="Regional Sales Distribution",
                hole=0.45,
                color_discrete_sequence=px.colors.qualitative.Bold
            )

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)"
            )

            with st.container(border=True):
                st.plotly_chart(fig, width="stretch")

        with region_col2:

            fig = px.bar(
                region_sales,
                x="Region",
                y="Sales",
                title="Regional Sales Comparison",
                text_auto=".2s",
                color="Region",
                color_discrete_sequence=px.colors.qualitative.Bold
            )

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False
            )

            with st.container(border=True):
                st.plotly_chart(fig, width="stretch")

        st.divider()

        # Top Product Analysis

        st.subheader("Top Product Performance")

        product_sales = (
            df.groupby("Product")["Sales"]
            .sum()
            .reset_index()
            .sort_values("Sales", ascending=False)
        )

        top_product = product_sales.iloc[0]["Product"]
        top_product_sales = product_sales.iloc[0]["Sales"]

        top_col1, top_col2 = st.columns(2)

        with top_col1:
            st.metric(
                "Top Selling Product",
                top_product
            )

        with top_col2:
            st.metric(
                "Top Product Sales",
                f"{top_product_sales:,.2f}"
            )

        with st.container(border=True):
            st.dataframe(
                product_sales,
                width="stretch",
                hide_index=True
            )

    else:

        st.info(
            "Business Intelligence requires these columns: "
            "Sales, Profit, Product, Category and Region."
        )


        # ==========================================
    # 12. MULTI-DOMAIN ANALYTICS
    # ==========================================

    st.header("11. Multi-Domain Analytics")

    st.markdown(
        "Analyze datasets across different domains using customizable calculations and visualizations."
    )

    # Domain Selection

    domain_col1, domain_col2 = st.columns([2, 3])

    with domain_col1:
        domain = st.selectbox(
            "Select Analysis Domain",
            ["Business", "Education", "E-commerce", "Finance"],
            key="analysis_domain"
        )

    domain_descriptions = {
        "Business": "Analyze business performance, operations, and trends.",
        "Education": "Explore student performance and academic data.",
        "E-commerce": "Understand product sales and customer behavior.",
        "Finance": "Explore financial figures and transaction data."
    }

    with domain_col2:
        st.info(domain_descriptions[domain])

    st.divider()

    # Dynamic Data Analysis

    st.subheader("Analysis Configuration")

    if numeric_columns:

        config_col1, config_col2 = st.columns(2)

        with config_col1:
            metric_column = st.selectbox(
                "Select Numerical Variable",
                numeric_columns,
                key="domain_metric"
            )

        with config_col2:
            aggregation = st.selectbox(
                "Select Calculation",
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
                "Group Analysis By",
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
                ascending=(aggregation == "Minimum")
            )

            st.divider()

            # Analysis Results

            st.subheader("Analysis Results")

            result_col1, result_col2, result_col3 = st.columns(3)

            with result_col1:
                st.metric(
                    "Groups Analyzed",
                    len(grouped_data)
                )

            with result_col2:
                st.metric(
                    "Selected Variable",
                    metric_column
                )

            with result_col3:
                st.metric(
                    "Calculation",
                    aggregation
                )

            # Results Table

            st.markdown("### Grouped Data")

            with st.container(border=True):
                st.dataframe(
                    grouped_data,
                    width="stretch",
                    hide_index=True
                )

            # Visualization

            st.markdown("### Analytical Visualization")

            fig = px.bar(
                grouped_data,
                x="Category",
                y="Value",
                title=f"{aggregation} of {metric_column} by {category_column}",
                text_auto=".2s",
                color="Category",
                color_discrete_sequence=px.colors.qualitative.Bold
            )

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                title_x=0.02
            )

            with st.container(border=True):
                st.plotly_chart(fig, width="stretch")

            # Key Insight

            if not grouped_data.empty:

                top_category = grouped_data.iloc[0]["Category"]
                top_value = grouped_data.iloc[0]["Value"]

                result_label = (
                    "Lowest" if aggregation == "Minimum" else "Highest"
                )

                st.success(
                    f"{result_label} {aggregation.lower()}: "
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

    st.header("12. Automated Analytics Report")

    st.markdown(
        "Generate and export a structured report containing dataset statistics and data quality information."
    )

    # Report Summary

    st.subheader("Report Overview")

    report_col1, report_col2, report_col3 = st.columns(3)

    with report_col1:
        st.metric("Total Records", f"{len(df):,}")

    with report_col2:
        st.metric("Total Features", f"{df.shape[1]:,}")

    with report_col3:
        st.metric(
            "Data Quality",
            f"{((df.notna().sum().sum() / df.size) * 100):.2f}%"
            if df.size > 0 else "100%"
        )

    # Generate Report

    report = StringIO()

    report.write("DATASPHERE ANALYTICS REPORT\n")
    report.write("=" * 50 + "\n\n")

    report.write("DATASET OVERVIEW\n")
    report.write("-" * 30 + "\n")
    report.write(f"Total Rows: {df.shape[0]}\n")
    report.write(f"Total Columns: {df.shape[1]}\n")
    report.write(
        f"Missing Values: {int(df.isna().sum().sum())}\n"
    )
    report.write(
        f"Duplicate Rows: {int(df.duplicated().sum())}\n\n"
    )

    report.write("COLUMN INFORMATION\n")
    report.write("-" * 30 + "\n")
    report.write(df.dtypes.astype(str).to_string())
    report.write("\n\n")

    report.write("STATISTICAL SUMMARY\n")
    report.write("-" * 30 + "\n")
    report.write(df.describe(include="all").to_string())

    report.write("\n\nMISSING VALUE ANALYSIS\n")
    report.write("-" * 30 + "\n")
    report.write(df.isna().sum().to_string())

    report.write("\n\nDUPLICATE ANALYSIS\n")
    report.write("-" * 30 + "\n")
    report.write(f"Duplicate Rows: {int(df.duplicated().sum())}\n")

    report.write("\n\nEND OF REPORT\n")
    report.write("=" * 50)

    report_content = report.getvalue()

    # Report Preview

    st.subheader("Report Preview")

    with st.expander("View Generated Analytics Report"):

        st.code(
            report_content,
            language="text"
        )

    # Report Download

    st.divider()

    st.subheader("Export Report")

    with st.container(border=True):

        st.write("Your analytics report is ready.")

        st.download_button(
            label="Download Analytics Report",
            data=report_content,
            file_name="datasphere_report.txt",
            mime="text/plain",
            use_container_width=True
        )

        st.caption(
            "Format: TXT | Includes dataset overview, column information, statistics, and data quality analysis."
        )
        # ==========================================
    # 14. ADVANCED DATA CLEANING
    # ==========================================

    st.header("13. Advanced Data Cleaning")

    st.markdown(
        "Identify missing data, apply cleaning strategies, and export your cleaned dataset."
    )

    # Missing Value Analysis

    st.subheader("Missing Value Assessment")

    missing_values = df.isnull().sum()

    missing_df = pd.DataFrame({
        "Column": missing_values.index,
        "Missing Values": missing_values.values,
        "Missing Percentage": (
            (missing_values.values / len(df)) * 100
        ).round(2) if len(df) > 0 else 0
    })

    total_missing = int(missing_values.sum())
    affected_columns = int((missing_values > 0).sum())

    completeness = (
        ((df.size - total_missing) / df.size) * 100
        if df.size > 0 else 100
    )

    clean_col1, clean_col2, clean_col3 = st.columns(3)

    with clean_col1:
        st.metric("Missing Values", f"{total_missing:,}")

    with clean_col2:
        st.metric("Affected Columns", affected_columns)

    with clean_col3:
        st.metric("Data Completeness", f"{completeness:.2f}%")

    st.markdown("### Missing Value Details")

    with st.container(border=True):
        st.dataframe(
            missing_df,
            width="stretch",
            hide_index=True
        )

    # Cleaning Operations

    st.divider()

    st.subheader("Cleaning Configuration")

    if total_missing > 0:

        cleaning_method = st.selectbox(
            "Select Cleaning Strategy",
            [
                "Remove Rows with Missing Values",
                "Fill Numerical Values with Mean",
                "Fill Numerical Values with Median",
                "Fill Categorical Values with Mode"
            ],
            key="advanced_cleaning_method"
        )

        st.info(
            f"Selected Strategy: {cleaning_method}"
        )

        if st.button(
            "Apply Cleaning",
            use_container_width=True,
            key="apply_advanced_cleaning"
        ):

            cleaned_df = df.copy()

            if cleaning_method == "Remove Rows with Missing Values":

                cleaned_df = cleaned_df.dropna()

            elif cleaning_method == "Fill Numerical Values with Mean":

                for column in cleaned_df.select_dtypes(
                    include="number"
                ).columns:

                    cleaned_df[column] = cleaned_df[column].fillna(
                        cleaned_df[column].mean()
                    )

            elif cleaning_method == "Fill Numerical Values with Median":

                for column in cleaned_df.select_dtypes(
                    include="number"
                ).columns:

                    cleaned_df[column] = cleaned_df[column].fillna(
                        cleaned_df[column].median()
                    )

            elif cleaning_method == "Fill Categorical Values with Mode":

                for column in cleaned_df.select_dtypes(
                    include=["object", "category", "string"]
                ).columns:

                    if not cleaned_df[column].mode().empty:

                        cleaned_df[column] = cleaned_df[column].fillna(
                            cleaned_df[column].mode()[0]
                        )

            # Cleaning Results

            remaining_missing = int(
                cleaned_df.isnull().sum().sum()
            )

            removed_rows = len(df) - len(cleaned_df)

            st.success("Cleaning operation completed!")

            result_col1, result_col2, result_col3 = st.columns(3)

            with result_col1:
                st.metric(
                    "Original Rows",
                    f"{len(df):,}"
                )

            with result_col2:
                st.metric(
                    "Remaining Rows",
                    f"{len(cleaned_df):,}"
                )

            with result_col3:
                st.metric(
                    "Remaining Missing Values",
                    f"{remaining_missing:,}"
                )

            st.caption(
                f"Rows removed: {removed_rows:,}"
            )

            st.subheader("Cleaned Dataset Preview")

            with st.container(border=True):
                st.dataframe(
                    cleaned_df,
                    width="stretch",
                    hide_index=True
                )

            # Download Cleaned Dataset

            st.divider()

            st.subheader("Export Cleaned Dataset")

            csv_data = cleaned_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="Download Cleaned CSV",
                data=csv_data,
                file_name="cleaned_dataset.csv",
                mime="text/csv",
                use_container_width=True
            )

    else:

        st.success(
            "Excellent! No missing values found. Your dataset is complete."
        )
           # ==========================================
    # 15. DUPLICATE DATA CLEANING
    # ==========================================

    st.subheader("Duplicate Data Handling")

    st.markdown(
        "Detect repeated records and remove unnecessary duplicate entries."
    )

    duplicate_count = int(df.duplicated().sum())
    total_rows = len(df)

    unique_rows = total_rows - duplicate_count

    duplicate_percentage = (
        (duplicate_count / total_rows) * 100
        if total_rows > 0 else 0
    )

    # Duplicate Statistics

    dup_col1, dup_col2, dup_col3 = st.columns(3)

    with dup_col1:
        st.metric("Duplicate Rows", f"{duplicate_count:,}")

    with dup_col2:
        st.metric("Unique Rows", f"{unique_rows:,}")

    with dup_col3:
        st.metric("Duplicate Percentage", f"{duplicate_percentage:.2f}%")

    st.divider()

    # Duplicate Removal

    st.subheader("Duplicate Removal")

    if duplicate_count > 0:

        st.warning(
            f"{duplicate_count:,} duplicate rows were detected."
        )

        if st.button(
            "Remove Duplicate Rows",
            use_container_width=True,
            key="remove_duplicate_rows"
        ):

            duplicate_cleaned_df = df.drop_duplicates().copy()

            removed_count = len(df) - len(duplicate_cleaned_df)

            st.success(
                f"Successfully removed {removed_count:,} duplicate rows!"
            )

            # Cleaning Results

            result_col1, result_col2, result_col3 = st.columns(3)

            with result_col1:
                st.metric("Original Rows", f"{len(df):,}")

            with result_col2:
                st.metric("Cleaned Rows", f"{len(duplicate_cleaned_df):,}")

            with result_col3:
                st.metric(
                    "Remaining Duplicates",
                    f"{int(duplicate_cleaned_df.duplicated().sum()):,}"
                )

            # Cleaned Dataset

            st.subheader("Dataset After Duplicate Removal")

            with st.container(border=True):
                st.dataframe(
                    duplicate_cleaned_df,
                    width="stretch",
                    hide_index=True
                )

            # Export

            st.divider()

            st.subheader("Export Cleaned Dataset")

            duplicate_csv = duplicate_cleaned_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="Download Duplicate-Free CSV",
                data=duplicate_csv,
                file_name="duplicate_free_dataset.csv",
                mime="text/csv",
                use_container_width=True
            )

    else:

        st.success(
            "Excellent! No duplicate rows found. Your dataset is unique."
        ) 
           # ==========================================
    # 16. DATA TYPE CORRECTION
    # ==========================================

    st.header("14. Data Type Correction")

    st.markdown(
        "Inspect column data types, convert values, and validate the results."
    )

    # Current Data Type Overview

    st.subheader("Current Data Type Overview")

    dtype_info = pd.DataFrame({
        "Column": df.columns,
        "Current Data Type": df.dtypes.astype(str).values,
        "Non-Missing Values": df.notna().sum().values
    })

    with st.container(border=True):
        st.dataframe(
            dtype_info,
            width="stretch",
            hide_index=True
        )

    # Conversion Configuration

    st.divider()

    st.subheader("Conversion Configuration")

    convert_col1, convert_col2 = st.columns(2)

    with convert_col1:
        selected_column = st.selectbox(
            "Select a column",
            df.columns,
            key="dtype_column"
        )

    with convert_col2:
        target_dtype = st.selectbox(
            "Select the new data type",
            ["Integer", "Decimal", "Text", "Date"],
            key="target_dtype"
        )

    st.info(
        f"Converting **{selected_column}** "
        f"from **{df[selected_column].dtype}** "
        f"to **{target_dtype}**."
    )

    # Execute Conversion

    if st.button(
        "Convert Data Type",
        use_container_width=True,
        key="convert_data_type"
    ):

        converted_df = df.copy()

        original_missing = int(
            converted_df[selected_column].isna().sum()
        )

        try:

            if target_dtype == "Integer":

                converted_df[selected_column] = pd.to_numeric(
                    converted_df[selected_column],
                    errors="coerce"
                ).astype("Int64")

            elif target_dtype == "Decimal":

                converted_df[selected_column] = pd.to_numeric(
                    converted_df[selected_column],
                    errors="coerce"
                ).astype("float64")

            elif target_dtype == "Text":

                converted_df[selected_column] = (
                    converted_df[selected_column].astype("string")
                )

            elif target_dtype == "Date":

                converted_df[selected_column] = pd.to_datetime(
                    converted_df[selected_column],
                    errors="coerce"
                )

            # Conversion Validation

            converted_missing = int(
                converted_df[selected_column].isna().sum()
            )

            newly_missing = max(
                0,
                converted_missing - original_missing
            )

            converted_values = (
                len(df) - converted_missing
            )

            # Conversion Results

            st.success("Data type conversion completed!")

            result_col1, result_col2, result_col3 = st.columns(3)

            with result_col1:
                st.metric(
                    "Converted Data Type",
                    str(converted_df[selected_column].dtype)
                )

            with result_col2:
                st.metric(
                    "Valid Values",
                    f"{converted_values:,}"
                )

            with result_col3:
                st.metric(
                    "New Missing Values",
                    f"{newly_missing:,}"
                )

            if newly_missing > 0:
                st.warning(
                    f"{newly_missing:,} values could not be converted "
                    "and became missing values (NaN/NaT)."
                )
            else:
                st.success(
                    "No additional missing values were created."
                )

            # Converted Dataset

            st.subheader("Converted Dataset Preview")

            with st.container(border=True):
                st.dataframe(
                    converted_df,
                    width="stretch",
                    hide_index=True
                )

            # Conversion Details

            st.markdown("### Conversion Details")

            st.write("Selected Column:", selected_column)

            st.write(
                "Original Data Type:",
                str(df[selected_column].dtype)
            )

            st.write(
                "New Data Type:",
                str(converted_df[selected_column].dtype)
            )

            # Export Converted Dataset

            st.divider()

            st.subheader("Export Converted Dataset")

            converted_csv = converted_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="Download Converted CSV",
                data=converted_csv,
                file_name="converted_dataset.csv",
                mime="text/csv",
                use_container_width=True
            )

        except Exception as error:

            st.error(
                f"Conversion failed: {error}"
            )
               # ==========================================
    # 17. OUTLIER DETECTION
    # ==========================================

    st.header("15. Outlier Detection")

    st.markdown(
        "Identify unusual numerical values using the Interquartile Range (IQR) method."
    )

    if numeric_columns:

        outlier_results = []

        total_outliers = 0

        for column in numeric_columns:

            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)

            IQR = Q3 - Q1

            lower_limit = Q1 - (1.5 * IQR)
            upper_limit = Q3 + (1.5 * IQR)

            outlier_count = int(
                (
                    (df[column] < lower_limit) |
                    (df[column] > upper_limit)
                ).sum()
            )

            outlier_percentage = (
                (outlier_count / len(df)) * 100
                if len(df) > 0 else 0
            )

            total_outliers += outlier_count

            outlier_results.append({
                "Column": column,
                "Q1": round(Q1, 2),
                "Q3": round(Q3, 2),
                "IQR": round(IQR, 2),
                "Lower Limit": round(lower_limit, 2),
                "Upper Limit": round(upper_limit, 2),
                "Outliers Found": outlier_count,
                "Outlier Percentage": round(outlier_percentage, 2)
            })

        outlier_df = pd.DataFrame(outlier_results)

        # Outlier Overview

        st.subheader("Outlier Analysis Overview")

        affected_columns = int(
            (outlier_df["Outliers Found"] > 0).sum()
        )

        total_numeric_columns = len(numeric_columns)

        overview_col1, overview_col2, overview_col3 = st.columns(3)

        with overview_col1:
            st.metric(
                "Numerical Columns",
                total_numeric_columns
            )

        with overview_col2:
            st.metric(
                "Columns with Outliers",
                affected_columns
            )

        with overview_col3:
            st.metric(
                "Total Outlier Instances",
                f"{total_outliers:,}"
            )

        st.divider()

        # Outlier Analysis Table

        st.subheader("Detailed Outlier Analysis")

        with st.container(border=True):
            st.dataframe(
                outlier_df,
                width="stretch",
                hide_index=True
            )

        # Outlier Visualization

        st.divider()

        st.subheader("Outlier Visualization")

        selected_outlier_column = st.selectbox(
            "Select numerical column",
            numeric_columns,
            key="outlier_visualization"
        )

        fig = px.box(
            df,
            y=selected_outlier_column,
            title=f"Outlier Distribution: {selected_outlier_column}",
            points="outliers",
            color_discrete_sequence=["#22D3EE"]
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            title_x=0.02,
            yaxis_title=selected_outlier_column
        )

        with st.container(border=True):
            st.plotly_chart(
                fig,
                width="stretch"
            )

        # Outlier Removal

        st.divider()

        st.subheader("Outlier Removal")

        st.markdown(
            "Remove rows containing outliers from the selected numerical column."
        )

        selected_stats = outlier_df[
            outlier_df["Column"] == selected_outlier_column
        ].iloc[0]

        selected_outlier_count = int(
            selected_stats["Outliers Found"]
        )

        if selected_outlier_count > 0:

            st.warning(
                f"{selected_outlier_count:,} outlier rows detected "
                f"in {selected_outlier_column}."
            )

            if st.button(
                "Remove Outliers",
                use_container_width=True,
                key="remove_outliers"
            ):

                Q1 = df[selected_outlier_column].quantile(0.25)
                Q3 = df[selected_outlier_column].quantile(0.75)

                IQR = Q3 - Q1

                lower_limit = Q1 - (1.5 * IQR)
                upper_limit = Q3 + (1.5 * IQR)

                outlier_cleaned_df = df[
                    (df[selected_outlier_column] >= lower_limit) &
                    (df[selected_outlier_column] <= upper_limit)
                ].copy()

                removed_count = len(df) - len(outlier_cleaned_df)

                remaining_outliers = int(
                    (
                        (outlier_cleaned_df[selected_outlier_column] < lower_limit) |
                        (outlier_cleaned_df[selected_outlier_column] > upper_limit)
                    ).sum()
                )

                st.success(
                    f"Successfully removed {removed_count:,} outlier rows!"
                )

                # Cleaning Results

                result_col1, result_col2, result_col3 = st.columns(3)

                with result_col1:
                    st.metric(
                        "Original Rows",
                        f"{len(df):,}"
                    )

                with result_col2:
                    st.metric(
                        "Cleaned Rows",
                        f"{len(outlier_cleaned_df):,}"
                    )

                with result_col3:
                    st.metric(
                        "Remaining Outliers",
                        remaining_outliers
                    )

                # Cleaned Dataset

                st.subheader("Dataset After Outlier Removal")

                with st.container(border=True):
                    st.dataframe(
                        outlier_cleaned_df,
                        width="stretch",
                        hide_index=True
                    )

                # Export

                st.divider()

                st.subheader("Export Cleaned Dataset")

                outlier_csv = outlier_cleaned_df.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="Download Outlier-Free CSV",
                    data=outlier_csv,
                    file_name="outlier_free_dataset.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        else:

            st.success(
                f"No outliers detected in {selected_outlier_column}."
            )

    else:

        st.warning(
            "No numerical columns available for outlier detection."
        )
            # ==========================================
    # 18. INTERACTIVE DATA VISUALIZATION DASHBOARD
    # ==========================================

    st.header("16. Interactive Data Visualization Dashboard")

    st.markdown(
        "Build customized, interactive charts and explore your dataset visually."
    )

    # Visualization Overview

    viz_col1, viz_col2, viz_col3 = st.columns(3)

    with viz_col1:
        st.metric("Available Columns", len(df.columns))

    with viz_col2:
        st.metric("Numerical Columns", len(numeric_columns))

    with viz_col3:
        st.metric("Categorical Columns", len(categorical_columns))

    st.divider()

    # Chart Selection

    st.subheader("Visualization Builder")

    chart_type = st.selectbox(
        "Select Visualization Type",
        [
            "Bar Chart",
            "Line Chart",
            "Pie Chart",
            "Histogram",
            "Scatter Plot"
        ],
        key="dashboard_chart_type"
    )

    st.divider()

    # Chart Configuration

    st.subheader("Chart Configuration")

    fig = None

    if chart_type in ["Bar Chart", "Line Chart"]:

        if not numeric_columns:

            st.warning(
                "This chart requires at least one numerical column."
            )

        else:

            config_col1, config_col2 = st.columns(2)

            with config_col1:
                x_column = st.selectbox(
                    "Select X-axis column",
                    df.columns,
                    key="dashboard_x"
                )

            with config_col2:
                y_column = st.selectbox(
                    "Select Y-axis column",
                    numeric_columns,
                    key="dashboard_y"
                )

            if chart_type == "Bar Chart":

                fig = px.bar(
                    df,
                    x=x_column,
                    y=y_column,
                    color=x_column,
                    title=f"{y_column} by {x_column}",
                    color_discrete_sequence=px.colors.qualitative.Bold
                )

                fig.update_layout(
                    showlegend=False
                )

            else:

                fig = px.line(
                    df,
                    x=x_column,
                    y=y_column,
                    title=f"{y_column} Trend by {x_column}",
                    markers=True
                )

    elif chart_type == "Pie Chart":

        if not categorical_columns:

            st.warning(
                "Pie charts require at least one categorical column."
            )

        else:

            pie_column = st.selectbox(
                "Select category column",
                categorical_columns,
                key="dashboard_pie"
            )

            pie_data = (
                df[pie_column]
                .value_counts()
                .reset_index()
            )

            pie_data.columns = ["Category", "Count"]

            fig = px.pie(
                pie_data,
                names="Category",
                values="Count",
                title=f"Distribution of {pie_column}",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Bold
            )

    elif chart_type == "Histogram":

        if not numeric_columns:

            st.warning(
                "Histograms require at least one numerical column."
            )

        else:

            histogram_column = st.selectbox(
                "Select numerical column",
                numeric_columns,
                key="dashboard_histogram"
            )

            fig = px.histogram(
                df,
                x=histogram_column,
                title=f"Distribution of {histogram_column}",
                color_discrete_sequence=["#22D3EE"]
            )

    elif chart_type == "Scatter Plot":

        if len(numeric_columns) < 2:

            st.warning(
                "Scatter plots require at least two numerical columns."
            )

        else:

            scatter_col1, scatter_col2 = st.columns(2)

            with scatter_col1:
                scatter_x = st.selectbox(
                    "Select X-axis",
                    numeric_columns,
                    key="dashboard_scatter_x"
                )

            with scatter_col2:
                scatter_y = st.selectbox(
                    "Select Y-axis",
                    numeric_columns,
                    index=1,
                    key="dashboard_scatter_y"
                )

            fig = px.scatter(
                df,
                x=scatter_x,
                y=scatter_y,
                title=f"{scatter_y} vs {scatter_x}",
                color_discrete_sequence=["#A78BFA"]
            )

    # Chart Customization and Display

    if fig is not None:

        st.divider()

        st.subheader("Chart Customization")

        custom_title = st.text_input(
            "Enter Chart Title",
            value="My DataSphere Visualization",
            key="custom_chart_title"
        )

        fig.update_layout(
            title=custom_title,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            title_x=0.02,
            font=dict(
                family="Arial",
                size=13
            ),
            margin=dict(
                l=30,
                r=30,
                t=70,
                b=30
            )
        )

        # Visualization Result

        st.subheader("Visualization Result")

        with st.container(border=True):
            st.plotly_chart(
                fig,
                width="stretch"
            )

        # Export Visualization

        st.divider()

        st.subheader("Export Visualization")

        html_chart = fig.to_html(
            full_html=True,
            include_plotlyjs=True
        )

        st.download_button(
            label="Download Interactive Chart (HTML)",
            data=html_chart,
            file_name="datasphere_chart.html",
            mime="text/html",
            use_container_width=True
        )

        st.caption(
            "Export your interactive Plotly chart and open it in a web browser."
        )
else:

    st.info(
        "Upload a CSV dataset to start exploring DataSphere."
    )
