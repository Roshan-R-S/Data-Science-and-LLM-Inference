"""Interactive Streamlit Application for Credit Score Prediction, Explainability, and Local LLM Inference.
Combines Data Science, Predictive ML, SHAP Interpretability, and local GGUF models.
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
    get_models_ml_dir
)
from src.explainability import (
    load_shap_feature_importance,
    load_shap_final_summary,
    load_sample_explanation,
    get_available_figures,
    get_results_dir
)
from src.llm import LocalLLMManager
from src.evaluation import evaluate_response_quality

# Page Configuration
st.set_page_config(
    page_title="Credit Score AI Platform",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0.2rem; }
    .subtitle { font-size: 1.1rem; color: #4B5563; margin-bottom: 1.5rem; }
    .metric-card { background-color: #F3F4F6; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #3B82F6; }
    .score-badge { font-size: 1.8rem; font-weight: 700; color: #10B981; }
</style>
""", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("💳 Navigation")
page = st.sidebar.radio(
    "Select Stage:",
    [
        "1. 📊 Data Explorer",
        "2. 🔍 Data Quality",
        "3. 📈 EDA Dashboard",
        "4. 🤖 ML Predictions",
        "5. 🔬 Explainability (XAI)",
        "6. 💬 LLM Explanations",
        "7. ⚖️ Model Comparison"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Architecture Highlights:**
- **ML Engine**: XGBoost Regressor ($R^2 = 0.9672$)
- **Explainability**: SHAP (PermutationExplainer)
- **Local LLMs**: Llama 3.2 1B & Qwen 2.5 1.5B (GGUF)
- **Inference Runtime**: llama.cpp CUDA
""")


# ==============================================================================
# 1. DATA EXPLORER
# ==============================================================================
if page == "1. 📊 Data Explorer":
    st.markdown('<div class="main-title">1. Customer Dataset Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Inspect and explore raw and processed credit card customer records.</div>', unsafe_allow_html=True)

    dataset_choice = st.radio(
        "Select Dataset Version:",
        ["Raw Dataset (data/raw)", "Cleaned Dataset (data/processed)", "Feature-Engineered Dataset (data/processed)"],
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

    st.subheader("Data Preview")
    rows_to_show = st.slider("Rows to display:", 5, 50, 10)
    st.dataframe(df.head(rows_to_show), width="stretch")

    with st.expander("Column Names and Data Types"):
        dtypes_df = pd.DataFrame({"Data Type": df.dtypes.astype(str), "Non-Null Count": df.notnull().sum()})
        st.dataframe(dtypes_df, width="stretch")

    with st.expander("Descriptive Statistics"):
        st.dataframe(df.describe().T, width="stretch")


# ==============================================================================
# 2. DATA QUALITY
# ==============================================================================
elif page == "2. 🔍 Data Quality":
    st.markdown('<div class="main-title">2. Data Quality & Validation Audit</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Systematic validation of missing values, duplicate records, and financial domain ranges.</div>', unsafe_allow_html=True)

    df_raw = load_raw_data()
    validation_report = run_full_validation(df_raw)

    col1, col2, col3 = st.columns(3)
    col1.metric("Missing Values", validation_report["total_missing_values"])
    col2.metric("Duplicate Records", validation_report["total_duplicates"])
    col3.metric("Zero-Variance Columns", len(validation_report["constant_columns"]))

    st.subheader("Domain Range Validation Checks")
    st.markdown("Verifies that credit indicators comply with standard domain boundaries:")
    range_df = pd.DataFrame(validation_report["range_violations"])
    st.dataframe(range_df, width='stretch')

    if validation_report["is_clean"]:
        st.success("✅ Complete dataset successfully passed all quality checks. Zero anomalies, missing values, or boundary violations detected.")


# ==============================================================================
# 3. EDA DASHBOARD
# ==============================================================================
elif page == "3. 📈 EDA Dashboard":
    st.markdown('<div class="main-title">3. Exploratory Data Analysis (EDA)</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Statistical patterns, feature distributions, skewness, and correlations with Credit Score.</div>', unsafe_allow_html=True)

    df = load_cleaned_data()
    numeric_cols, cat_cols = get_column_types(df)

    tab1, tab2, tab3 = st.tabs(["Target Correlations", "Feature Distributions", "Category Breakdowns"])

    with tab1:
        st.subheader("Numerical Correlations with Credit Score")
        corr_series = compute_correlations_with_target(df, target_col="Credit_Score", top_n=12)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        corr_series.sort_values().plot(kind="barh", ax=ax, color="#2563EB")
        ax.set_title("Top Correlated Features with Credit Score")
        ax.set_xlabel("Pearson Correlation Coefficient")
        st.pyplot(fig)

        st.info("💡 **Key Finding**: Payment Ratio (+0.767) and Credit Utilization (-0.599) dominate all linear relationships with Credit Score.")

    with tab2:
        st.subheader("Distribution Viewer")
        selected_col = st.selectbox("Select Numerical Feature:", numeric_cols)
        
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df[selected_col], bins=30, color="#3B82F6", edgecolor="black", alpha=0.7)
        ax.set_title(f"Histogram: {selected_col}")
        ax.set_xlabel(selected_col)
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

        _, highly_skewed = compute_skewness_analysis(df)
        st.markdown("**Features with High Skewness ($|skew| > 1.0$):**")
        st.dataframe(highly_skewed, width='stretch')

    with tab3:
        st.subheader("Average Credit Score by Categorical Attribute")
        selected_cat = st.selectbox("Select Categorical Feature:", cat_cols)
        cat_summary = compute_category_target_summary(df, cat_col=selected_cat)
        
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(cat_summary[selected_cat], cat_summary["mean"], color="#10B981")
        ax.set_ylabel("Mean Credit Score")
        ax.set_title(f"Average Credit Score by {selected_cat}")
        plt.xticks(rotation=30)
        st.pyplot(fig)
        st.dataframe(cat_summary, width='stretch')


# ==============================================================================
# 4. ML PREDICTIONS
# ==============================================================================
elif page == "4. 🤖 ML Predictions":
    st.markdown('<div class="main-title">4. Real-Time Credit Score Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Predict credit scores from raw customer attributes using the saved preprocessing pipeline and trained XGBoost model.</div>', unsafe_allow_html=True)

    preprocessor = CustomerPreprocessor.load()
    model = load_xgb_model()

    st.subheader("Input Customer Attributes")

    preset = st.selectbox(
        "Load Customer Sample Preset:",
        ["Custom Input", "Sample 0: Moderate Score (CC100000)", "Sample 1: Prime Customer (CC100003)"]
    )

    if preset == "Sample 0: Moderate Score (CC100000)":
        default_age, default_income, default_limit = 33, 787406.0, 220000.0
        default_spend, default_tx = 36733.49, 83
        default_pr, default_cu = 0.46, 0.70
        default_gender, default_occ, default_card = "M", "Self-Employed", "Silver"
    elif preset == "Sample 1: Prime Customer (CC100003)":
        default_age, default_income, default_limit = 42, 712664.0, 211000.0
        default_spend, default_tx = 21804.26, 64
        default_pr, default_cu = 0.95, 0.16
        default_gender, default_occ, default_card = "M", "Private Employee", "Silver"
    else:
        default_age, default_income, default_limit = 35, 1000000.0, 300000.0
        default_spend, default_tx = 45000.0, 50
        default_pr, default_cu = 0.75, 0.30
        default_gender, default_occ, default_card = "F", "Private Employee", "Gold"

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider("Age", 18, 85, default_age)
        income = st.number_input("Annual Income (INR)", 100000.0, 5000000.0, float(default_income), step=50000.0)
        credit_limit = st.number_input("Credit Limit (INR)", 50000.0, 2000000.0, float(default_limit), step=25000.0)
        gender = st.selectbox("Gender", ["M", "F"], index=0 if default_gender == "M" else 1)
    with col2:
        monthly_spend = st.number_input("Monthly Spending (INR)", 5000.0, 500000.0, float(default_spend), step=5000.0)
        tx_count = st.number_input("Monthly Transactions", 5, 200, int(default_tx))
        payment_ratio = st.slider("Payment Ratio", 0.0, 1.0, float(default_pr), step=0.01)
        occupation = st.selectbox("Occupation", ["Private Employee", "Government Employee", "Self-Employed", "Business Owner", "Retired", "Student"], index=0)
    with col3:
        credit_utilization = st.slider("Credit Utilization", 0.0, 1.0, float(default_cu), step=0.01)
        emi_count = st.number_input("Active EMI Count", 0, 10, 2)
        card_type = st.selectbox("Card Type", ["Basic", "Silver", "Gold", "Platinum", "Signature"], index=1)
        mobile_logins = st.number_input("Monthly App Logins", 0, 100, 25)

    if st.button("Generate Credit Score Prediction", type="primary"):
        # Construct raw input dict
        input_data = {
            "Age": age,
            "Annual_Income": income,
            "Credit_Limit": credit_limit,
            "Card_Age_Months": 36,
            "Monthly_Spending": monthly_spend,
            "Monthly_Transactions": tx_count,
            "Avg_Transaction_Value": monthly_spend / max(1, tx_count),
            "Online_Shopping_Spending": monthly_spend * 0.3,
            "Grocery_Spending": monthly_spend * 0.25,
            "Fuel_Spending": monthly_spend * 0.1,
            "Dining_Spending": monthly_spend * 0.15,
            "Travel_Spending": monthly_spend * 0.1,
            "Entertainment_Spending": monthly_spend * 0.05,
            "Utility_Bill_Spending": monthly_spend * 0.05,
            "Outstanding_Balance": credit_limit * credit_utilization,
            "Statement_Balance": credit_limit * credit_utilization * 1.1,
            "Payment_Amount": (credit_limit * credit_utilization) * payment_ratio,
            "Payment_Ratio": payment_ratio,
            "Credit_Utilization": credit_utilization,
            "Cash_Advance_Amount": 0.0,
            "EMI_Count": emi_count,
            "International_Transactions": 0,
            "Reward_Points_Earned": int(monthly_spend * 0.05),
            "Reward_Points_Redeemed": int(monthly_spend * 0.01),
            "Mobile_App_Login": mobile_logins,
            "Gender": gender,
            "Occupation": occupation,
            "Card_Type": card_type
        }

        with st.spinner("Processing through preprocessing pipeline and XGBoost regressor..."):
            pred_score = predict_credit_score(input_data, model=model, preprocessor=preprocessor)

        st.markdown("---")
        res_col1, res_col2 = st.columns([1, 2])
        with res_col1:
            st.metric("Predicted Credit Score", f"{pred_score:.1f} / 850")
            if pred_score >= 750:
                st.success("🌟 Tier: Excellent (Prime)")
            elif pred_score >= 670:
                st.info("👍 Tier: Good")
            elif pred_score >= 580:
                st.warning("⚠️ Tier: Fair")
            else:
                st.error("🚨 Tier: Poor (Subprime)")
        with res_col2:
            st.markdown(f"""
            **Prediction Breakdown:**
            - **Payment Ratio ({payment_ratio:.2f})**: Strong positive contribution.
            - **Credit Utilization ({credit_utilization:.2f})**: Primary risk penalty.
            - Model used: Trained `XGBRegressor` (100 estimators, max depth 6).
            """)


# ==============================================================================
# 5. EXPLAINABILITY (XAI)
# ==============================================================================
elif page == "5. 🔬 Explainability (XAI)":
    st.markdown('<div class="main-title">5. Explainable AI (SHAP Interpretability)</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Investigate feature importance, contribution directions, and individual prediction waterfall decompositions.</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Global Feature Importance", "Individual Waterfall Breakdown", "SHAP Diagnostic Plots"])

    with tab1:
        st.subheader("Top Global Feature Drivers")
        shap_summary = load_shap_final_summary()
        st.dataframe(shap_summary, width='stretch')

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(shap_summary["Feature"][::-1], shap_summary["Mean_Absolute_SHAP"][::-1], color="#2563EB")
        ax.set_xlabel("Mean |SHAP Value| (Impact on Credit Score points)")
        ax.set_title("Global SHAP Feature Importance")
        st.pyplot(fig)

    with tab2:
        st.subheader("Sample 0 Individual Customer Explanation")
        sample_0 = load_sample_explanation(0)
        st.dataframe(sample_0.head(10), width='stretch')

        figures = get_available_figures()
        if "shap_waterfall_sample_0" in figures:
            st.image(str(figures["shap_waterfall_sample_0"]), caption="Sample 0 SHAP Waterfall Plot", width='stretch')

    with tab3:
        st.subheader("Precomputed SHAP Dependence Plots")
        figures = get_available_figures()
        plot_options = [k for k in figures.keys() if "waterfall" not in k]
        selected_plot = st.selectbox("Select SHAP Plot:", plot_options)
        if selected_plot in figures:
            st.image(str(figures[selected_plot]), caption=selected_plot, width='stretch')


# ==============================================================================
# 6. LLM EXPLANATIONS
# ==============================================================================
elif page == "6. 💬 LLM Explanations":
    st.markdown('<div class="main-title">6. Local LLM Analytical Explanations</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">AI-generated evidence-based explanations running locally on your GPU via llama.cpp.</div>', unsafe_allow_html=True)

    model_choice = st.selectbox("Select Local Model:", ["Meta Llama 3.2 1B Instruct", "Qwen 2.5 1.5B Instruct"])
    model_key = "llama" if "Llama" in model_choice else "qwen"

    task_choice = st.radio("Select Analytical Task:", ["Customer Prediction Explanation", "Model Comparison Explanation"], horizontal=True)
    task_key = "customer_explanation" if "Customer" in task_choice else "model_comparison"

    exec_mode = st.radio("Mode:", ["View Pre-Generated Benchmark Output", "Execute Live on Local GPU Server"], horizontal=True)

    outputs_dir = get_results_dir() / "llm_outputs"
    output_file = outputs_dir / f"{model_key}_{task_key}.txt"

    if exec_mode == "View Pre-Generated Benchmark Output":
        if output_file.exists():
            with open(output_file, "r", encoding="utf-8") as f:
                content = f.read()
            st.markdown("### AI Analytical Response:")
            st.markdown(f"""
            <div style="background-color: #F9FAFB; padding: 1.5rem; border-radius: 0.5rem; border: 1px solid #E5E7EB; font-family: monospace;">
            {content}
            </div>
            """, unsafe_allow_html=True)

            eval_res = evaluate_response_quality(content)
            st.markdown("---")
            st.subheader("Deterministic Evaluation Audit:")
            col1, col2, col3 = st.columns(3)
            col1.metric("Factual Consistency", f"{eval_res['factual_consistency']['factual_consistency_score']:.2f}")
            col2.metric("Numerical Accuracy", f"{eval_res['numerical_accuracy']['accuracy_score']:.2f}")
            col3.metric("Quality Rating", eval_res["rating"])
        else:
            st.warning(f"No output found at {output_file.name}. Run the benchmark first.")
    else:
        st.info("Live local GPU execution requires the model server process. Make sure port 8080 is available.")
        if st.button(f"Generate Live Explanation via {model_choice}", type="primary"):
            manager = LocalLLMManager(port=8080, server_timeout=45)
            with st.spinner(f"Initializing {model_choice} on GPU..."):
                try:
                    manager.start_server(model_key)
                    prompt_file = get_project_root() / "prompts" / ("analysis.txt" if task_key == "customer_explanation" else "comparison.txt")
                    with open(prompt_file, "r", encoding="utf-8") as f:
                        prompt_tmpl = f.read()
                    
                    if task_key == "customer_explanation":
                        prompt = prompt_tmpl.format(
                            predicted_score="639.6",
                            actual_score="648.0",
                            top_features_summary="- Payment_Ratio: +44.97 impact\n- Credit_Utilization: -34.18 impact",
                            customer_highlights="- Age: 33 | Gender: Male | Self-Employed"
                        )
                    else:
                        prompt = prompt_tmpl

                    res = manager.generate(prompt, max_tokens=300)
                    st.success(f"Generated in {res['latency_seconds']}s ({res['tokens_per_second']} tokens/sec)")
                    st.write(res["content"])
                except Exception as e:
                    st.error(f"Inference error: {e}")
                finally:
                    manager.stop_server()


# ==============================================================================
# 7. MODEL COMPARISON
# ==============================================================================
elif page == "7. ⚖️ Model Comparison":
    st.markdown('<div class="main-title">7. Model & LLM Benchmarks</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Comparative evaluation across both predictive machine learning models and local generative language models.</div>', unsafe_allow_html=True)

    st.subheader("Machine Learning Algorithm Comparison (Credit Score Regression)")
    model_comp_path = get_results_dir() / "model_comparison.csv"
    if model_comp_path.exists():
        ml_df = pd.read_csv(model_comp_path)
        st.dataframe(ml_df, width='stretch')

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(ml_df["Model"], ml_df["R2"], color=["#10B981", "#3B82F6", "#F59E0B"])
        ax.set_ylabel("R² (Test Set Variance Explained)")
        ax.set_ylim(0.95, 0.97)
        ax.set_title("Regression Model Comparison (R²)")
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("Local LLM Comparative Benchmark (Llama 3.2 1B vs. Qwen 2.5 1.5B)")
    benchmark_path = get_results_dir() / "metrics" / "llm_benchmark_results.csv"
    if benchmark_path.exists():
        llm_df = pd.read_csv(benchmark_path)
        st.dataframe(llm_df, width='stretch')

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(llm_df["model"] + " (" + llm_df["task"] + ")", llm_df["tokens_per_sec"], color="#6366F1")
            ax.set_ylabel("Tokens / Second")
            ax.set_title("LLM Throughput Comparison")
            plt.xticks(rotation=30, ha="right")
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(llm_df["model"] + " (" + llm_df["task"] + ")", llm_df["latency_sec"], color="#EC4899")
            ax.set_ylabel("Latency (Seconds)")
            ax.set_title("LLM Response Latency Comparison")
            plt.xticks(rotation=30, ha="right")
            st.pyplot(fig)
    else:
        st.info("LLM benchmark metrics will appear here after running Phase 3.")
