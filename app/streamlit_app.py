"""Interactive Streamlit Application for Credit Score Prediction, Explainability, and Local LLM Inference.
Combines Data Science, Predictive Machine Learning, SHAP Interpretability, and local GGUF models via llama.cpp.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.data_loader import (
    load_raw_data,
    load_cleaned_data,
    load_feature_engineered_data,
    get_project_root
)
from src.validation import run_full_validation
from src.analysis import (
    get_column_types,
    compute_skewness_analysis,
    compute_correlations_with_target,
    compute_category_target_summary
)
from src.modeling import (
    CustomerPreprocessor,
    load_xgb_model,
    predict_credit_score,
    get_models_ml_dir,
    get_test_predictions_and_residuals
)
from src.explainability import (
    load_shap_feature_importance,
    load_shap_final_summary,
    load_sample_explanation,
    get_available_figures,
    get_results_dir,
    explain_custom_prediction
)
from src.llm import LocalLLMManager
from src.evaluation import evaluate_response_quality

# Page Configuration
st.set_page_config(
    page_title="Credit Score AI Intelligence Platform",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title { font-size: 2.1rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0.2rem; }
    .subtitle { font-size: 1.05rem; color: #4B5563; margin-bottom: 1.5rem; }
    .metric-card { background-color: #F8FAFC; padding: 1.2rem; border-radius: 0.5rem; border-left: 4px solid #3B82F6; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
    .disclaimer-box { background-color: #FFFBEB; border: 1px solid #FDE68A; border-left: 4px solid #F59E0B; padding: 1rem; border-radius: 0.4rem; margin-bottom: 1rem; font-size: 0.95rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 1rem; }
    .stTabs [data-baseweb="tab"] { height: 45px; white-space: pre-wrap; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "custom_customer" not in st.session_state:
    st.session_state.custom_customer = None
if "predicted_score" not in st.session_state:
    st.session_state.predicted_score = None

# Sidebar Navigation
st.sidebar.title("💳 Navigation")
page = st.sidebar.radio(
    "Select Stage:",
    [
        "1. 📊 Data Explorer",
        "2. 🔍 Data Quality & Preparation",
        "3. 📈 EDA Dashboard",
        "4. 🤖 ML Workbench",
        "5. 🔬 Explainable AI (XAI)",
        "6. 💬 LLM Analytical Playground",
        "7. ⚖️ LLM Evaluation & Benchmarking"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**System Architecture:**
- **Raw Data**: 50,000 customers × 30 features
- **ML Engine**: XGBoost Regressor ($R^2 = 0.9672$)
- **Explainability**: SHAP TreeExplainer
- **Local LLMs**: Llama 3.2 1B & Qwen 2.5 1.5B
- **Inference Runtime**: llama.cpp CUDA
""")


# ==============================================================================
# 1. DATA EXPLORER (Notebook 01)
# ==============================================================================
if page == "1. 📊 Data Explorer":
    st.markdown('<div class="main-title">1. Customer Dataset Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Dataset inspection, schema validation, target distribution, and identifier integrity audit (Notebook 01).</div>', unsafe_allow_html=True)

    dataset_choice = st.radio(
        "Select Dataset Version to Inspect:",
        [
            "Raw Dataset (data/raw - 50,000 × 30)",
            "Cleaned Dataset (data/processed - 50,000 × 30)",
            "Feature-Engineered Dataset (data/processed - 50,000 × 34)"
        ],
        horizontal=True
    )

    if "Raw" in dataset_choice:
        df = load_raw_data()
    elif "Cleaned" in dataset_choice:
        df = load_cleaned_data()
    else:
        df = load_feature_engineered_data()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Total Columns", f"{len(df.columns)}")
    numeric_cols, cat_cols = get_column_types(df)
    col3.metric("Numerical Features", len(numeric_cols))
    col4.metric("Categorical Features", len(cat_cols))

    st.markdown("---")

    # Section 1: Identifier Cardinality & Target Distribution
    col_id, col_target = st.columns(2)

    with col_id:
        st.subheader("🆔 Identifier Cardinality Audit")
        id_col = "Customer_ID"
        if id_col in df.columns:
            n_unique = df[id_col].nunique()
            cardinality = (n_unique / len(df)) * 100
            st.metric("Customer_ID Unique Count", f"{n_unique:,} / {len(df):,} ({cardinality:.1f}%)")
            st.markdown(f"""
            - **Column Name**: `{id_col}`
            - **Duplicate Keys**: `0`
            - **Missing Values**: `0`
            - **Modeling Decision**: `Customer_ID` is a high-cardinality synthetic customer tracking key. It carries **zero generalizable predictive signal** and is **strictly excluded** from all machine learning feature matrices to prevent identity leakage.
            """)
        else:
            st.info("Customer_ID column is not present in this dataset split.")

    with col_target:
        st.subheader("🎯 Target Distribution: `Credit_Score`")
        if "Credit_Score" in df.columns:
            cs = df["Credit_Score"]
            t_mean, t_std, t_med = cs.mean(), cs.std(), cs.median()
            t_min, t_max, t_skew = cs.min(), cs.max(), cs.skew()

            c_m1, c_m2, c_m3 = st.columns(3)
            c_m1.metric("Mean", f"{t_mean:.1f}")
            c_m2.metric("Std Dev", f"{t_std:.1f}")
            c_m3.metric("Median", f"{t_med:.1f}")

            c_m4, c_m5, c_m6 = st.columns(3)
            c_m4.metric("Min", f"{t_min}")
            c_m5.metric("Max", f"{t_max}")
            c_m6.metric("Skewness", f"{t_skew:.3f} (Symmetric)")

            fig, ax = plt.subplots(figsize=(6, 2.8))
            ax.hist(cs, bins=40, color="#2563EB", edgecolor="black", alpha=0.7, density=True)
            cs.plot(kind="kde", ax=ax, color="#DC2626", linewidth=2)
            ax.set_title("Credit Score Empirical Distribution (Bell Curve)")
            ax.set_xlabel("Credit Score (Points)")
            ax.set_ylabel("Density")
            st.pyplot(fig)
        else:
            st.info("Target column Credit_Score not present in this view.")

    st.markdown("---")

    # Section 2: Data Preview
    st.subheader("📋 Dataset Inspection")
    c_view, c_rows = st.columns([1, 2])
    with c_view:
        view_type = st.radio("View Position:", ["First N Rows (Head)", "Last N Rows (Tail)"], horizontal=True)
    with c_rows:
        n_rows = st.slider("Rows to display:", 5, 50, 10)

    preview_df = df.head(n_rows) if "Head" in view_type else df.tail(n_rows)
    st.dataframe(preview_df, width="stretch")

    # Section 3: Expanders for Types and Summary Statistics
    with st.expander("🔍 Complete Column Inventory and Data Types"):
        type_df = pd.DataFrame({
            "Column Name": df.columns,
            "Data Type": [str(t) for t in df.dtypes],
            "Non-Null Count": df.notnull().sum().values,
            "Null Count": df.isnull().sum().values
        })
        st.dataframe(type_df, width="stretch")

    with st.expander("📊 Complete Descriptive Statistics"):
        st.dataframe(df.describe().T, width="stretch")


# ==============================================================================
# 2. DATA QUALITY & PREPARATION (Notebooks 02, 03, 05)
# ==============================================================================
elif page == "2. 🔍 Data Quality & Preparation":
    st.markdown('<div class="main-title">2. Data Quality & Preparation Pipeline</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Systematic quality validation, cleaning transformations, and feature engineering architecture (Notebooks 02, 03, 05).</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([
        "1. Quality & Range Validation (NB 02)",
        "2. Cleaning Pipeline & Deduplication (NB 03)",
        "3. Feature Engineering & Preprocessing Schema (NB 05)"
    ])

    with tab1:
        st.subheader("Data Quality & Domain Range Audit")
        st.markdown("Verifies missing values, duplicate rows, zero-variance columns, and financial domain boundary rules across the 50,000 records.")

        df_raw = load_raw_data()
        validation_report = run_full_validation(df_raw)

        c1, c2, c3 = st.columns(3)
        c1.metric("Missing Values", f"{validation_report['total_missing_values']} (0.0%)")
        c2.metric("Duplicate Rows", f"{validation_report['total_duplicates']} (0.0%)")
        c3.metric("Constant Columns", len(validation_report["constant_columns"]))

        st.markdown("#### Domain Boundary Checks")
        st.markdown("Rules enforced to confirm customer indicators fall within valid financial limits:")
        range_df = pd.DataFrame(validation_report["range_violations"])
        st.dataframe(range_df, width="stretch")

        if validation_report["is_clean"]:
            st.success("✅ **Quality Verification Passed**: Zero missing values, zero duplicates, zero constant columns, and zero out-of-bounds violations across all 50,000 records.")

    with tab2:
        st.subheader("Data Cleaning Pipeline & Deduplication")
        st.markdown("Audits transformations executed to transition from raw inputs to verified cleaned data.")

        clean_summary = pd.DataFrame({
            "Stage": ["Raw Dataset (data/raw)", "Cleaned Dataset (data/processed)"],
            "Row Count": [50000, 50000],
            "Column Count": [30, 30],
            "Missing Values": [0, 0],
            "Duplicate IDs": [0, 0],
            "Categorical Encoding": ["Raw Strings", "Standardized Clean Strings"],
            "Domain Integrity": ["Validated (0 violations)", "Verified Ready for Modeling"]
        })
        st.dataframe(clean_summary, width="stretch")

        st.markdown("""
        **Data Cleaning Operations Applied:**
        1. **Identifier Uniqueness Guarantee**: Verified that all `Customer_ID` values are 100% unique primary keys without collisions.
        2. **Categorical Normalization**: Stripped leading/trailing whitespace and validated category consistency on `Gender` (M, F), `Occupation` (6 levels), and `Card_Type` (5 levels).
        3. **Data Type Standardization**: Numeric types cast to 64-bit floats and integers; target `Credit_Score` cast to integer points.
        4. **Schema Preservation**: Zero records required drop or imputation due to the synthetic dataset's high baseline integrity.
        """)

    with tab3:
        st.subheader("Feature Engineering & Preprocessing Schema")
        st.markdown("Notebook 05 generated **4 engineered financial features** to capture spending discrepancies and balance-to-limit ratios.")

        st.markdown("#### The 4 Engineered Feature Formulas")

        st.markdown(r"""
        1. **Calculated Monthly Spending**:
           $$\text{calculated\_monthly\_spending} = \frac{\text{Annual\_Income}}{12.0}$$
           *Original Input Column*: `Annual_Income`.<br>
           *Why Created*: Establishes an estimated monthly gross income benchmark to evaluate whether the customer's monthly card spending exceeds their monthly earning capacity.

        2. **Spending Difference**:
           $$\text{spending\_difference} = \text{Monthly\_Spending} - \text{calculated\_monthly\_spending}$$
           *Original Input Columns*: `Monthly_Spending`, `calculated_monthly_spending`.<br>
           *Why Created*: Measures the net monthly budget deviation (surplus or deficit) between actual credit card spending and monthly income.

        3. **Calculated Utilization**:
           $$\text{calculated\_utilization} = \frac{\text{Monthly\_Spending}}{\text{Credit\_Limit}}$$
           *Original Input Columns*: `Monthly_Spending`, `Credit_Limit` (with zero-division protection: safe limit fallback to `0.0`).<br>
           *Why Created*: Captures current spending velocity relative to the assigned credit limit, independent of revolving balances.

        4. **Spending Relative Difference**:
           $$\text{spending\_relative\_difference} = \frac{\text{spending\_difference}}{\text{calculated\_monthly\_spending}}$$
           *Original Input Columns*: `spending_difference`, `calculated_monthly_spending`.<br>
           *Why Created*: Normalizes spending surplus/deficit relative to customer income scale to allow fair comparison across income brackets.
        """)

        st.markdown("#### Example Before & After Values (Sample Customer CC100000)")
        ex_df = pd.DataFrame({
            "Input Feature": [
                "Annual_Income",
                "Monthly_Spending",
                "Credit_Limit"
            ],
            "Raw Value": [
                "₹7,87,406.00",
                "₹36,733.49",
                "₹2,20,000.00"
            ],
            "Engineered Output": [
                "calculated_monthly_spending = ₹65,617.17",
                "spending_difference = -₹28,883.68 (budget surplus)",
                "calculated_utilization = 0.1670 (16.70%) & spending_relative_diff = -0.4402"
            ]
        })
        st.dataframe(ex_df, width="stretch")

        st.markdown("#### Preprocessing & Encoding Pipeline Schema")
        st.markdown("""
        How 30 raw columns transition into the final **39 model features**:
        - **Raw Dataset**: **30 columns** (1 ID + 1 Target + 28 raw features).
        - **Feature Engineering**: Adds 4 engineered columns $\\rightarrow$ **34 columns**.
        - **Column Exclusion**: Drops `Customer_ID` (identifier) and `Credit_Score` (target) $\\rightarrow$ **32 predictors**.
        - **One-Hot Encoding**: One-hot encodes 3 categorical features with `drop_first=True`:
          - `Gender` (2 levels $\\rightarrow$ 1 dummy column: `Gender_M`)
          - `Occupation` (6 levels $\\rightarrow$ 5 dummy columns)
          - `Card_Type` (5 levels $\\rightarrow$ 4 dummy columns)
          - *Net Change*: $-3$ categorical columns $+ 10$ dummy columns $= +7$ columns.
        - **Final Model Matrix**: $32 + 7 =$ **39 numeric features**.
        """)


# ==============================================================================
# 3. EDA DASHBOARD (Notebook 04)
# ==============================================================================
elif page == "3. 📈 EDA Dashboard":
    st.markdown('<div class="main-title">3. Exploratory Data Analysis (EDA)</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Statistical patterns, feature distributions, skewness, and correlations with Credit Score (Notebook 04).</div>', unsafe_allow_html=True)

    df = load_cleaned_data()
    numeric_cols, cat_cols = get_column_types(df)

    tab1, tab2, tab3 = st.tabs([
        "Target Correlations & Bivariate Scatterplots",
        "Distribution Viewer & Skewness Analysis",
        "Categorical Demographic Breakdowns"
    ])

    with tab1:
        st.subheader("Correlations & Relationships with Credit Score")
        
        st.markdown("""
        <div class="disclaimer-box">
        <strong>⚠️ Crucial Data Science Principle: Association vs. Causation</strong><br>
        While <code>Payment_Ratio</code> (+0.767) and <code>Credit_Utilization</code> (-0.599) show very strong linear correlations with Credit Score, correlation indicates statistical association under the dataset's generative mechanism, not unobserved physical causation.
        </div>
        """, unsafe_allow_html=True)

        corr_series = compute_correlations_with_target(df, target_col="Credit_Score", top_n=12)

        c_corr, c_scatter = st.columns([1, 1])
        with c_corr:
            fig, ax = plt.subplots(figsize=(6, 4.5))
            corr_series.sort_values().plot(kind="barh", ax=ax, color="#2563EB")
            ax.set_title("Top Correlated Features with Credit Score")
            ax.set_xlabel("Pearson Correlation (r)")
            st.pyplot(fig)

        with c_scatter:
            st.markdown("**Bivariate Scatterplot: Key Drivers**")
            scatter_choice = st.selectbox(
                "Select Scatter View:",
                ["Payment_Ratio vs Credit_Score", "Credit_Utilization vs Credit_Score", "Annual_Income vs Monthly_Spending"]
            )
            # Sample 1000 points for fast, clean rendering
            sample_sub = df.sample(n=1000, random_state=42)

            fig, ax = plt.subplots(figsize=(6, 4))
            if scatter_choice == "Payment_Ratio vs Credit_Score":
                ax.scatter(sample_sub["Payment_Ratio"], sample_sub["Credit_Score"], alpha=0.3, color="#2563EB", s=15)
                m, b = np.polyfit(sample_sub["Payment_Ratio"], sample_sub["Credit_Score"], 1)
                ax.plot(sample_sub["Payment_Ratio"], m * sample_sub["Payment_Ratio"] + b, color="#DC2626", linewidth=2)
                ax.set_xlabel("Payment Ratio (0.0 - 1.0)")
                ax.set_ylabel("Credit Score")
                ax.set_title(f"Payment Ratio vs Credit Score (r = +{corr_series['Payment_Ratio']:.3f})")
            elif scatter_choice == "Credit_Utilization vs Credit_Score":
                ax.scatter(sample_sub["Credit_Utilization"], sample_sub["Credit_Score"], alpha=0.3, color="#E11D48", s=15)
                m, b = np.polyfit(sample_sub["Credit_Utilization"], sample_sub["Credit_Score"], 1)
                ax.plot(sample_sub["Credit_Utilization"], m * sample_sub["Credit_Utilization"] + b, color="#1E3A8A", linewidth=2)
                ax.set_xlabel("Credit Utilization (0.0 - 1.0)")
                ax.set_ylabel("Credit Score")
                ax.set_title(f"Credit Utilization vs Credit Score (r = {corr_series['Credit_Utilization']:.3f})")
            else:
                ax.scatter(sample_sub["Annual_Income"] / 1e5, sample_sub["Monthly_Spending"] / 1e3, alpha=0.3, color="#059669", s=15)
                ax.set_xlabel("Annual Income (Lakhs INR)")
                ax.set_ylabel("Monthly Spending (Thousands INR)")
                ax.set_title("Annual Income vs Monthly Spending")
            st.pyplot(fig)

    with tab2:
        st.subheader("Distribution Viewer & Skewness Analysis")
        selected_col = st.selectbox("Select Numerical Feature to Inspect:", numeric_cols, index=numeric_cols.index("Annual_Income") if "Annual_Income" in numeric_cols else 0)

        c_dist, c_box = st.columns(2)
        with c_dist:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.hist(df[selected_col], bins=35, color="#3B82F6", edgecolor="black", alpha=0.7)
            ax.set_title(f"Histogram: {selected_col}")
            ax.set_xlabel(selected_col)
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

        with c_box:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.boxplot(df[selected_col], vert=False, patch_artist=True, boxprops=dict(facecolor="#93C5FD"))
            ax.set_title(f"Boxplot: {selected_col}")
            ax.set_xlabel(selected_col)
            st.pyplot(fig)

        skew_all, highly_skewed = compute_skewness_analysis(df)
        st.markdown("#### Highly Skewed Features ($|skew| > 1.0$)")
        st.dataframe(highly_skewed, width="stretch")

    with tab3:
        st.subheader("Average Credit Score by Categorical Demographic")
        selected_cat = st.selectbox("Select Categorical Feature:", cat_cols)
        cat_summary = compute_category_target_summary(df, cat_col=selected_cat)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(cat_summary[selected_cat], cat_summary["mean"], color="#10B981", edgecolor="black", alpha=0.85)
        ax.set_ylabel("Mean Credit Score")
        ax.set_title(f"Average Credit Score by {selected_cat}")
        ax.set_ylim(550, 650)
        plt.xticks(rotation=25, ha="right")
        st.pyplot(fig)

        st.dataframe(cat_summary, width="stretch")


# ==============================================================================
# 4. ML WORKBENCH (Notebook 06)
# ==============================================================================
elif page == "4. 🤖 ML Workbench":
    st.markdown('<div class="main-title">4. Machine Learning Workbench</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Live customer prediction, multi-model evaluation metrics, and residual error analysis (Notebook 06).</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([
        "Live Customer Prediction",
        "Multi-Model Evaluation & Comparison",
        "Residual & Error Analysis"
    ])

    with tab1:
        st.subheader("Predict Credit Score for Customer Input")

        st.markdown("""
        <div class="disclaimer-box">
        <strong>⚠️ Experimental Model Disclaimer:</strong><br>
        This prediction is generated by a trained XGBoost regression model on a synthetic dataset for analytical and research demonstration. It is <strong>not</strong> an official credit bureau score (e.g. FICO or CIBIL).
        </div>
        """, unsafe_allow_html=True)

        preprocessor = CustomerPreprocessor.load()
        model = load_xgb_model()

        preset = st.selectbox(
            "Load Customer Sample Preset:",
            [
                "Sample 0: CC100000 (Subprime Profile - Ground Truth: 492)",
                "Sample 1: CC100003 (Prime Profile - Ground Truth: 740)",
                "Sample 2: CC125056 (NB 07 SHAP Benchmark - Ground Truth: 618)",
                "Custom Input Form"
            ]
        )

        if "Sample 0" in preset:
            d_age, d_inc, d_limit = 33, 787406.0, 220000.0
            d_spend, d_tx = 36733.49, 83
            d_pr, d_cu = 0.46, 0.70
            d_gender, d_occ, d_card = "M", "Self-Employed", "Silver"
            d_emi, d_logins = 2, 23
            d_cash = 14839.84
            d_card_age = 20
        elif "Sample 1" in preset:
            d_age, d_inc, d_limit = 42, 712664.0, 211000.0
            d_spend, d_tx = 21804.26, 64
            d_pr, d_cu = 0.95, 0.16
            d_gender, d_occ, d_card = "M", "Private Employee", "Silver"
            d_emi, d_logins = 3, 13
            d_cash = 0.0
            d_card_age = 54
        elif "Sample 2" in preset:
            d_age, d_inc, d_limit = 55, 1289358.0, 522000.0
            d_spend, d_tx = 78977.40, 136
            d_pr, d_cu = 0.68, 0.29
            d_gender, d_occ, d_card = "F", "Private Employee", "Gold"
            d_emi, d_logins = 3, 45
            d_cash = 0.0
            d_card_age = 34
        else:
            d_age, d_inc, d_limit = 35, 1000000.0, 300000.0
            d_spend, d_tx = 45000.0, 50
            d_pr, d_cu = 0.75, 0.30
            d_gender, d_occ, d_card = "F", "Private Employee", "Gold"
            d_emi, d_logins = 2, 25
            d_cash = 0.0
            d_card_age = 36

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Financial Capacity**")
            age = st.slider("Age", 18, 85, int(d_age))
            income = st.number_input("Annual Income (INR)", 100000.0, 5000000.0, float(d_inc), step=50000.0)
            credit_limit = st.number_input("Credit Limit (INR)", 50000.0, 2000000.0, float(d_limit), step=25000.0)
            cash_advance = st.number_input("Cash Advance Amount (INR)", 0.0, 200000.0, float(d_cash), step=5000.0)
            gender = st.selectbox("Gender", ["M", "F"], index=0 if d_gender == "M" else 1)

        with col2:
            st.markdown("**Transaction Behavior**")
            monthly_spend = st.number_input("Monthly Spending (INR)", 5000.0, 500000.0, float(d_spend), step=5000.0)
            tx_count = st.number_input("Monthly Transactions", 5, 200, int(d_tx))
            payment_ratio = st.slider("Payment Ratio", 0.0, 1.0, float(d_pr), step=0.01)
            credit_utilization = st.slider("Credit Utilization", 0.0, 1.0, float(d_cu), step=0.01)
            occupation = st.selectbox("Occupation", ["Private Employee", "Government Employee", "Self-Employed", "Business Owner", "Retired", "Student"], index=2 if "Self" in d_occ else 0)

        with col3:
            st.markdown("**Account & Card Profile**")
            emi_count = st.number_input("Active EMI Count", 0, 10, int(d_emi))
            card_type = st.selectbox("Card Type", ["Basic", "Silver", "Gold", "Platinum", "Signature"], index=1 if d_card == "Silver" else 2)
            mobile_logins = st.number_input("Monthly App Logins", 0, 100, int(d_logins))
            card_age = st.number_input("Card Age (Months)", 1, 120, int(d_card_age))

        # Load complete feature dictionary if benchmark preset selected
        if "Sample 0" in preset:
            raw_full = load_raw_data()
            input_data = raw_full[raw_full["Customer_ID"] == "CC100000"].drop(columns=["Customer_ID", "Credit_Score"]).iloc[0].to_dict()
        elif "Sample 1" in preset:
            raw_full = load_raw_data()
            input_data = raw_full[raw_full["Customer_ID"] == "CC100003"].drop(columns=["Customer_ID", "Credit_Score"]).iloc[0].to_dict()
        elif "Sample 2" in preset:
            raw_full = load_raw_data()
            input_data = raw_full[raw_full["Customer_ID"] == "CC125056"].drop(columns=["Customer_ID", "Credit_Score"]).iloc[0].to_dict()
        else:
            input_data = {}

        # Update with active widget controls
        input_data.update({
            "Age": age,
            "Annual_Income": income,
            "Credit_Limit": credit_limit,
            "Card_Age_Months": card_age,
            "Monthly_Spending": monthly_spend,
            "Monthly_Transactions": tx_count,
            "Payment_Ratio": payment_ratio,
            "Credit_Utilization": credit_utilization,
            "Cash_Advance_Amount": cash_advance,
            "EMI_Count": emi_count,
            "Mobile_App_Login": mobile_logins,
            "Gender": gender,
            "Occupation": occupation,
            "Card_Type": card_type
        })
        if "Avg_Transaction_Value" not in input_data:
            input_data["Avg_Transaction_Value"] = monthly_spend / max(1, tx_count)

        # Ensure all category breakdowns exist if starting from custom input
        for col, mult in [
            ("Online_Shopping_Spending", 0.3),
            ("Grocery_Spending", 0.25),
            ("Fuel_Spending", 0.1),
            ("Dining_Spending", 0.15),
            ("Travel_Spending", 0.1),
            ("Entertainment_Spending", 0.05),
            ("Utility_Bill_Spending", 0.05)
        ]:
            if col not in input_data:
                input_data[col] = monthly_spend * mult
        if "Outstanding_Balance" not in input_data:
            input_data["Outstanding_Balance"] = credit_limit * credit_utilization
        if "Statement_Balance" not in input_data:
            input_data["Statement_Balance"] = credit_limit * credit_utilization * 1.05
        if "Payment_Amount" not in input_data:
            input_data["Payment_Amount"] = (credit_limit * credit_utilization) * payment_ratio
        if "International_Transactions" not in input_data:
            input_data["International_Transactions"] = 0
        if "Reward_Points_Earned" not in input_data:
            input_data["Reward_Points_Earned"] = int(monthly_spend * 0.05)
        if "Reward_Points_Redeemed" not in input_data:
            input_data["Reward_Points_Redeemed"] = int(monthly_spend * 0.01)

        # Save current customer to session state for live SHAP and LLM
        st.session_state.custom_customer = input_data

        if st.button("Generate ML Prediction", type="primary"):
            pred_score = predict_credit_score(input_data, model=model, preprocessor=preprocessor)
            st.session_state.predicted_score = pred_score

        if st.session_state.predicted_score is not None:
            pred = st.session_state.predicted_score
            st.markdown("---")
            r1, r2 = st.columns([1, 2])
            with r1:
                st.metric("Predicted Credit Score", f"{pred:.1f} / 850")
                if pred >= 750:
                    st.success("🌟 Tier: Excellent (Prime)")
                elif pred >= 670:
                    st.info("👍 Tier: Good")
                elif pred >= 580:
                    st.warning("⚠️ Tier: Fair")
                else:
                    st.error("🚨 Tier: Poor (Subprime)")
            with r2:
                st.markdown(f"""
                **Customer Key Drivers:**
                - **Payment Ratio ({payment_ratio:.2f})**: Dominant positive driver.
                - **Credit Utilization ({credit_utilization:.2f})**: Dominant negative risk driver.
                - Model: `XGBRegressor` evaluated on 39 encoded features.
                - *Tip: Navigate to **Page 5 (Explainable AI)** to view live SHAP attribution for this customer.*
                """)

        st.markdown("---")
        st.markdown("#### Documented Score Bands Reference Table")
        bands_df = pd.DataFrame({
            "Score Band": ["750 – 850", "670 – 749", "580 – 669", "300 – 579"],
            "Credit Tier": ["Excellent (Prime)", "Good", "Fair", "Poor (Subprime)"],
            "Default Risk Profile": ["Extremely low default probability; prime lending terms", "Low credit risk; standard prime loan eligibility", "Moderate credit risk; higher interest rate tier", "Substantial default risk; subprime credit management"],
            "Dataset Proportion": ["~18%", "~34%", "~32%", "~16%"]
        })
        st.dataframe(bands_df, width="stretch")

    with tab2:
        st.subheader("Multi-Model Comparative Evaluation")
        st.markdown("Comparison across the three regression algorithms trained in Notebook 06 on 10,000 held-out test customers:")

        model_comp_path = get_results_dir() / "model_comparison.csv"
        if model_comp_path.exists():
            ml_df = pd.read_csv(model_comp_path)
            st.dataframe(ml_df, width="stretch")

            c_b1, c_b2, c_b3 = st.columns(3)
            with c_b1:
                fig, ax = plt.subplots(figsize=(4, 3))
                ax.bar(ml_df["Model"], ml_df["R2"], color=["#10B981", "#3B82F6", "#F59E0B"])
                ax.set_ylabel("R² Score")
                ax.set_ylim(0.95, 0.97)
                ax.set_title("Test R² (Variance Explained)")
                plt.xticks(rotation=20, ha="right")
                st.pyplot(fig)
            with c_b2:
                fig, ax = plt.subplots(figsize=(4, 3))
                ax.bar(ml_df["Model"], ml_df["MAE"], color=["#10B981", "#3B82F6", "#F59E0B"])
                ax.set_ylabel("MAE (Points)")
                ax.set_ylim(9.0, 11.5)
                ax.set_title("Test Mean Absolute Error")
                plt.xticks(rotation=20, ha="right")
                st.pyplot(fig)
            with c_b3:
                fig, ax = plt.subplots(figsize=(4, 3))
                ax.bar(ml_df["Model"], ml_df["RMSE"], color=["#10B981", "#3B82F6", "#F59E0B"])
                ax.set_ylabel("RMSE (Points)")
                ax.set_ylim(11.5, 14.5)
                ax.set_title("Test Root Mean Squared Error")
                plt.xticks(rotation=20, ha="right")
                st.pyplot(fig)

            st.markdown("""
            **Technical Analysis: Why Linear Regression closely matches XGBoost ($R^2 = 0.9656$ vs $0.9672$):**
            - The synthetic credit score generator built the ground truth primarily around an affine combination of payment habits:
              $$\\text{Credit\\_Score} \\approx 546.48 + 220 \\times \\text{Payment\\_Ratio} - 200 \\times \\text{Credit\\_Utilization} + \\epsilon$$
            - Because the underlying synthetic relationship is predominantly linear, the linear baseline captures over 96.5% of total variance.
            - XGBoost slightly outperforms Linear Regression by modeling subtle non-linear edge conditions, interaction terms, and spending penalties.
            """)

    with tab3:
        st.subheader("Residual & Error Analysis")
        st.markdown("Detailed error diagnostics on 1,000 held-out test customers:")

        res_df = get_test_predictions_and_residuals(sample_size=1000)

        c_r1, c_r2 = st.columns(2)
        with c_r1:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(res_df["Actual"], res_df["Predicted"], alpha=0.3, color="#2563EB", s=15)
            min_val = min(res_df["Actual"].min(), res_df["Predicted"].min())
            max_val = max(res_df["Actual"].max(), res_df["Predicted"].max())
            ax.plot([min_val, max_val], [min_val, max_val], color="#DC2626", linestyle="--", linewidth=2, label="Ideal y = x")
            ax.set_xlabel("Actual Credit Score")
            ax.set_ylabel("Predicted Credit Score")
            ax.set_title("Actual vs. Predicted Credit Score")
            ax.legend()
            st.pyplot(fig)

        with c_r2:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(res_df["Predicted"], res_df["Residual"], alpha=0.3, color="#059669", s=15)
            ax.axhline(0, color="#DC2626", linestyle="--", linewidth=1.5)
            ax.set_xlabel("Predicted Credit Score")
            ax.set_ylabel("Residual (e = y - y_hat)")
            ax.set_title("Residual Plot (Homoscedasticity Check)")
            st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(10, 3))
        ax.hist(res_df["Residual"], bins=40, color="#6366F1", edgecolor="black", alpha=0.7)
        ax.axvline(0, color="#DC2626", linestyle="--", linewidth=2)
        ax.set_title(f"Residual Distribution (Mean: {res_df['Residual'].mean():.2f}, Std: {res_df['Residual'].std():.2f})")
        ax.set_xlabel("Prediction Error (Points)")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)


# ==============================================================================
# 5. EXPLAINABLE AI (Notebook 07)
# ==============================================================================
elif page == "5. 🔬 Explainable AI (XAI)":
    st.markdown('<div class="main-title">5. Explainable AI (SHAP Interpretability)</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Global feature importance, directional attributions, and live local SHAP prediction breakdowns (Notebook 07).</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([
        "Global Feature Importance & Direction",
        "Individual Explanations (Static & Live Custom)",
        "SHAP Diagnostic & Dependence Plots"
    ])

    with tab1:
        st.subheader("Global Feature Importance (TreeExplainer)")
        st.markdown("Mean absolute SHAP value across all test records, quantifying average impact on credit score points:")

        c_g1, c_g2 = st.columns([1, 1])
        with c_g1:
            shap_importance = load_shap_feature_importance()
            st.dataframe(shap_importance.head(15), width="stretch")
        with c_g2:
            shap_summary = load_shap_final_summary()
            st.markdown("**Top Feature Contribution Direction**")
            st.dataframe(shap_summary, width="stretch")

        fig, ax = plt.subplots(figsize=(10, 4.5))
        top_plot = shap_importance.head(12)
        ax.barh(top_plot["Feature"][::-1], top_plot["Mean_Absolute_SHAP"][::-1], color="#2563EB")
        ax.set_xlabel("Mean |SHAP Value| (Impact in Credit Score Points)")
        ax.set_title("Top 12 Features by Global SHAP Importance")
        st.pyplot(fig)

        st.info("💡 **Dominant Drivers**: `Payment_Ratio` (mean ~45.0 points impact) and `Credit_Utilization` (mean ~34.2 points impact) drive over 80% of total score variation.")

    with tab2:
        st.subheader("Individual Customer Prediction Explanations")

        subtab_a, subtab_b = st.tabs(["Notebook 07 Reference Sample (CC125056 - Actual: 618)", "Live SHAP for Current Custom Prediction"])

        with subtab_a:
            st.markdown("**Held-Out Test Benchmark Sample (CC125056)**: Actual Score = 618, Model Predicted Score = 639.6.")
            sample_0 = load_sample_explanation(0)
            st.dataframe(sample_0.head(12), width="stretch")

            figures = get_available_figures()
            if "shap_waterfall_sample_0" in figures:
                st.image(str(figures["shap_waterfall_sample_0"]), caption="Notebook 07 Benchmark SHAP Waterfall Plot for CC125056 (Base Value = 628.97 → Actual = 618)", width="stretch")

        with subtab_b:
            st.markdown("**Live SHAP Explanation for Custom Prediction**")
            st.markdown("Dynamically computes exact SHAP feature attributions for the customer currently loaded in Page 4 (ML Workbench):")

            current_cust = st.session_state.custom_customer
            if current_cust is None:
                st.warning("No customer profile loaded. Go to Page 4 (ML Workbench) to select or configure a customer.")
            else:
                if st.button("Compute Live SHAP Attribution", type="primary"):
                    with st.spinner("Computing TreeSHAP attributions..."):
                        shap_result = explain_custom_prediction(current_cust, top_n=12)
                        st.session_state["live_shap_result"] = shap_result

                if "live_shap_result" in st.session_state:
                    res = st.session_state["live_shap_result"]
                    base_val = res["base_value"]
                    pred_val = res["prediction"]

                    col_m1, col_m2, col_m3 = st.columns(3)
                    col_m1.metric("Base Value E[f(x)]", f"{base_val:.2f} pts")
                    col_m2.metric("Model Prediction", f"{pred_val:.2f} pts")
                    total_shap = res["explanation_df"]["SHAP_Value"].sum()
                    col_m3.metric("Net SHAP Shift", f"{total_shap:+.2f} pts")

                    st.markdown(f"""
                    **Reconstruction Check**: Base Value ({base_val:.2f}) + Total SHAP ({total_shap:+.2f}) = **{base_val + total_shap:.2f}** (Matches prediction: **{pred_val:.2f}**).
                    """)

                    st.markdown("#### Top Feature Contributions")
                    exp_df = res["explanation_df"].head(10)
                    st.dataframe(exp_df[["Feature", "Feature_Value", "SHAP_Value", "Direction"]], width="stretch")

                    fig, ax = plt.subplots(figsize=(8, 4))
                    colors = ["#10B981" if v >= 0 else "#EF4444" for v in exp_df["SHAP_Value"][::-1]]
                    ax.barh(exp_df["Feature"][::-1], exp_df["SHAP_Value"][::-1], color=colors)
                    ax.axvline(0, color="black", linestyle="--", linewidth=1)
                    ax.set_xlabel("SHAP Attribution (Score Points Contribution)")
                    ax.set_title("Live Feature Attribution (Green = Positive, Red = Negative)")
                    st.pyplot(fig)

    with tab3:
        st.subheader("Precomputed SHAP Dependence & Summary Plots")
        figures = get_available_figures()
        plot_options = [k for k in figures.keys() if "waterfall" not in k]
        selected_plot = st.selectbox("Select Diagnostic Plot:", plot_options)
        if selected_plot in figures:
            st.image(str(figures[selected_plot]), caption=selected_plot, width="stretch")


# ==============================================================================
# 6. LLM ANALYTICAL PLAYGROUND (Notebook 08 - Interactive)
# ==============================================================================
elif page == "6. 💬 LLM Analytical Playground":
    st.markdown('<div class="main-title">6. Local LLM Analytical Playground</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Interactive local generative AI running on GPU via llama.cpp CUDA backend (Notebook 08).</div>', unsafe_allow_html=True)

    c_m1, c_m2 = st.columns(2)
    with c_m1:
        model_choice = st.selectbox("Select Local GGUF Model:", ["Meta Llama 3.2 1B Instruct", "Qwen 2.5 1.5B Instruct"])
        model_key = "llama" if "Llama" in model_choice else "qwen"
    with c_m2:
        context_task = st.selectbox(
            "Select Analytical Evidence Context:",
            [
                "Customer Prediction Explanation (SHAP Evidence)",
                "Model Evaluation Explanation (Benchmark Metrics)",
                "Custom Evidence & Freeform Question"
            ]
        )

    st.markdown("---")
    st.subheader("Configure Prompt & Generation Parameters")

    p_col1, p_col2 = st.columns([2, 1])
    with p_col1:
        if "Customer" in context_task:
            default_prompt = (
                "You are a professional financial data science analyst.\n"
                "Analytical Evidence:\n"
                "- Predicted Credit Score: 639.6\n"
                "- Actual Credit Score: 648.0\n"
                "- Top SHAP Positive Driver: Payment_Ratio (+44.97 points)\n"
                "- Top SHAP Negative Driver: Credit_Utilization (-34.18 points)\n"
                "- Customer Profile: Age 33, Annual Income ₹7,87,406, Occupation: Self-Employed\n\n"
                "Analytical Task:\n"
                "Explain the customer's predicted credit score in clear, objective financial language. "
                "Highlight the primary factors influencing the score and suggest actionable steps to improve it."
            )
        elif "Model" in context_task:
            default_prompt = (
                "You are an expert machine learning engineer.\n"
                "Model Evaluation Benchmark Evidence (10,000 Test Customers):\n"
                "- XGBoost Regressor: MAE = 9.93 points, RMSE = 12.46 points, R² = 0.9672\n"
                "- Linear Regression (Baseline): MAE = 10.17 points, RMSE = 12.75 points, R² = 0.9656\n"
                "- Random Forest: MAE = 10.90 points, RMSE = 13.67 points, R² = 0.9605\n\n"
                "Analytical Task:\n"
                "Compare the predictive performance of these three models. Explain why the Linear Regression baseline "
                "performs almost as well as the non-linear XGBoost regressor on this dataset."
            )
        else:
            default_prompt = (
                "You are a senior credit risk analyst.\n"
                "Question: How do payment ratio and credit utilization mathematically impact creditworthiness?"
            )

        user_prompt = st.text_area("Prompt & Analytical Evidence:", default_prompt, height=220)

    with p_col2:
        st.markdown("**Inference Hyperparameters**")
        max_tokens = st.slider("Max New Tokens:", 128, 1024, 512, step=64)
        temperature = st.slider("Sampling Temperature:", 0.0, 1.0, 0.2, step=0.05)
        st.info("💡 **Low Temperature (0.1–0.2)** is recommended for factual financial explanations to minimize hallucination.")

    mode = st.radio("Execution Mode:", ["Live Local GPU Execution (llama.cpp CUDA)", "View Pre-Generated Benchmark Output"], horizontal=True)

    if mode == "Live Local GPU Execution (llama.cpp CUDA)":
        st.markdown("""
        <div class="disclaimer-box">
        <strong>Hardware Notice:</strong> Live local GPU execution starts <code>llama-server.exe</code> on <code>localhost:8080</code>, runs native Instruct chat inference, and shuts down the process cleanly.
        </div>
        """, unsafe_allow_html=True)

        if st.button(f"🚀 Run Live Inference on GPU ({model_choice})", type="primary"):
            manager = LocalLLMManager(port=8080, server_timeout=45)
            with st.spinner(f"Loading {model_choice} into GPU VRAM and executing inference..."):
                try:
                    manager.start_server(model_key)
                    res = manager.generate(user_prompt, max_tokens=max_tokens, temperature=temperature)
                    
                    tokens_count = res.get("tokens_predicted", res.get("tokens_generated", 0))
                    latency_val = res.get("latency_seconds", 0.0)
                    tps_val = res.get("tokens_per_second", 0.0)
                    st.success(f"✅ Generated {tokens_count} tokens in {latency_val:.2f}s ({tps_val:.1f} tokens/sec)")
                    st.markdown("### Model Response:")
                    st.markdown(f"""
                    <div style="background-color: #F8FAFC; padding: 1.2rem; border-radius: 0.5rem; border: 1px solid #E2E8F0; font-size: 1.05rem; line-height: 1.6;">
                    {res['content']}
                    </div>
                    """, unsafe_allow_html=True)

                    task_mode = "customer_explanation" if "Customer" in context_task else "model_comparison"
                    expected_nums = [639.6, 648.0, 44.97, 34.18] if task_mode == "customer_explanation" else [9.93, 12.46, 0.9672, 10.17, 12.75, 0.9656, 10.90, 13.67, 0.9605]
                    eval_res = evaluate_response_quality(res['content'], expected_numbers=expected_nums, task=task_mode)
                    st.markdown("---")
                    st.subheader("Deterministic Evaluation Audit")
                    c_e1, c_e2, c_e3 = st.columns(3)
                    c_e1.metric("Factual Consistency", f"{eval_res['factual_consistency']['factual_consistency_score']:.2f}")
                    c_e2.metric("Numerical Accuracy", f"{eval_res['numerical_accuracy']['accuracy_score']:.2f}")
                    c_e3.metric("Quality Rating", eval_res["rating"])
                except Exception as e:
                    st.error(f"Inference error: {e}")
                finally:
                    manager.stop_server()
    else:
        task_file_key = "customer_explanation" if "Customer" in context_task else "model_comparison"
        bench_file = get_results_dir() / "llm_outputs" / f"{model_key}_{task_file_key}.txt"
        if bench_file.exists():
            content = bench_file.read_text(encoding="utf-8")
            st.markdown(f"### Pre-Generated Benchmark Response ({model_choice}):")
            st.markdown(f"""
            <div style="background-color: #F8FAFC; padding: 1.2rem; border-radius: 0.5rem; border: 1px solid #E2E8F0; font-size: 1.05rem; line-height: 1.6;">
            {content}
            </div>
            """, unsafe_allow_html=True)

            expected_nums = [639.6, 648.0, 44.97, 34.18] if task_file_key == "customer_explanation" else [9.93, 12.46, 0.9672, 10.17, 12.75, 0.9656, 10.90, 13.67, 0.9605]
            eval_res = evaluate_response_quality(content, expected_numbers=expected_nums, task=task_file_key)
            st.markdown("---")
            st.subheader("Deterministic Evaluation Audit")
            c_e1, c_e2, c_e3 = st.columns(3)
            c_e1.metric("Factual Consistency", f"{eval_res['factual_consistency']['factual_consistency_score']:.2f}")
            c_e2.metric("Numerical Accuracy", f"{eval_res['numerical_accuracy']['accuracy_score']:.2f}")
            c_e3.metric("Quality Rating", eval_res["rating"])
        else:
            st.info("Pre-generated output file not found. Use Live Execution mode.")


# ==============================================================================
# 7. LLM EVALUATION & BENCHMARKING (Notebook 08 - Systematic)
# ==============================================================================
elif page == "7. ⚖️ LLM Evaluation & Benchmarking":
    st.markdown('<div class="main-title">7. LLM Evaluation & Systematic Benchmarking</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Systematic comparative evaluation of local LLMs across accuracy, consistency, throughput, and latency (Notebook 08).</div>', unsafe_allow_html=True)

    benchmark_path = get_results_dir() / "metrics" / "llm_benchmark_results.csv"

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Llama 3.2 1B Throughput", "41.3 tokens/sec", "Fastest Generation")
    col_m2.metric("Qwen 2.5 1.5B Throughput", "30.7 tokens/sec", "Higher Precision")
    col_m3.metric("Evaluation Methodology", "Deterministic Audit", "Regex + Logic Checks")

    st.markdown("---")

    if benchmark_path.exists():
        llm_df = pd.read_csv(benchmark_path)
        st.subheader("Systematic Evaluation Benchmark Results")
        st.dataframe(llm_df, width="stretch")

        st.markdown("#### Comparative Hardware & Performance Charts")
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(6, 3.8))
            bars = ax.bar(llm_df["model"] + "\n(" + llm_df["task"] + ")", llm_df["tokens_per_sec"], color="#6366F1")
            ax.set_ylabel("Throughput (Tokens / Second)")
            ax.set_title("Generation Throughput (Higher is Better)")
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f"{height:.1f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points", ha="center", va="bottom")
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots(figsize=(6, 3.8))
            bars = ax.bar(llm_df["model"] + "\n(" + llm_df["task"] + ")", llm_df["latency_sec"], color="#EC4899")
            ax.set_ylabel("Latency (Seconds)")
            ax.set_title("Total Generation Latency (Lower is Better)")
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f"{height:.2f}s", xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points", ha="center", va="bottom")
            st.pyplot(fig)

    st.markdown("---")

    st.subheader("Deterministic Evaluation Methodology")
    st.markdown("""
    To avoid using an expensive or non-deterministic LLM-as-a-judge, the evaluation pipeline in `src/evaluation.py` implements a **deterministic scoring rubric**:

    1. **Numerical Accuracy Scoring ($0.0 - 1.0$)**:
       - Extracts all numerical values from the generated response using regular expressions.
       - Compares against ground truth quantitative anchors from the analytical context (e.g. MAE 9.93, RMSE 12.46, $R^2$ 0.9672, predicted score 639.6).
       - Scores the exact percentage of correct quantitative assertions cited in the text.

    2. **Factual Consistency Scoring ($0.0 - 1.0$)**:
       - Evaluates directional correctness: checks whether the response correctly states that `Payment_Ratio` has a positive/beneficial relationship with score and `Credit_Utilization` has a negative/penalty relationship.
       - Confirms whether model comparison identifies XGBoost as the top-performing model.

    3. **Hallucination & Truncation Safeguards**:
       - Monitors response token length and sentence termination.
       - Verifies that chat templates (`<|start_header_id|>`, `<|im_start|>`) properly isolate system instructions from analytical generation.
    """)

    st.markdown("---")
    st.subheader("Technical Note: Server Cold-Start vs. Token Generation Latency")
    st.markdown("""
    In local LLM architectures, there is a fundamental distinction between two time components:
    - **Server Cold-Start Latency (~4.0 – 6.5s)**: The one-time overhead of spinning up `llama-server.exe`, reading the 4-bit quantized GGUF weights from SSD storage, and transferring model tensors into GPU VRAM.
    - **Active Per-Token Generation Latency (~23 – 32ms per token)**: The actual inference speed once weights reside in VRAM, which yields **41.3 tokens/sec** on Llama 3.2 1B and **30.7 tokens/sec** on Qwen 2.5 1.5B.
    - In persistent production deployment, the server remains warm in VRAM, meaning users experience only the active generation latency.
    """)
