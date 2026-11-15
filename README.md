# Data Science + Local LLM Inference Pipeline

An end-to-end data science and local-LLM intelligence system that analyses synthetic credit-card customer behaviour, predicts credit scores using machine learning, explains model predictions with explainable AI (XAI), and deploys locally hosted large language models (LLMs) to interpret analytical results in natural language.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
- [Pipeline Architecture](#pipeline-architecture)
- [Dataset](#dataset)
- [Machine Learning Results](#machine-learning-results)
- [Explainable AI (SHAP)](#explainable-ai-shap)
- [Local LLM Inference & Benchmarking](#local-llm-inference--benchmarking)
- [Repository Structure](#repository-structure)
- [Technology Stack](#technology-stack)
- [Getting Started](#getting-started)
- [Running the Notebooks](#running-the-notebooks)
- [Running the Streamlit Application](#running-the-streamlit-application)
- [Running Unit and Integration Tests](#running-unit-and-integration-tests)
- [Current Status](#current-status)
- [Limitations & Reliability Audit](#limitations--reliability-audit)
- [Disclaimer](#disclaimer)

---

## Project Overview

This project bridges three major disciplines in modern applied AI:

1. **Data Science and Machine Learning**: Data validation, automated cleaning, exploratory data analysis, feature engineering, cross-validation, and multi-model regression.
2. **Explainable AI (XAI)**: Quantifying global feature importance and individual prediction waterfall attributions using SHAP.
3. **Local LLM Inference & Evaluation**: Deploying quantized instruction-tuned LLMs (`Llama 3.2 1B` and `Qwen 2.5 1.5B`) on local GPU runtimes via `llama.cpp` and evaluating their analytical explanations against numerical ground truth.

The system is encapsulated within a modular Python codebase, a 32-test automated test suite, and an interactive **Streamlit** analytical dashboard.

---

## Problem Statement

A financial institution managing 50,000 credit-card customers needs more than raw records and black-box predictions:

- **Data Quality**: What anomalies, missing values, or invalid ranges exist?
- **Feature Significance**: Which behavioural patterns associate most strongly with creditworthiness?
- **Predictive Accuracy**: Can customer credit scores be accurately predicted with machine learning?
- **Explainability**: Why did the model generate a specific credit score for a particular customer?
- **AI Interpretation**: Can a local LLM generate human-understandable, factually consistent explanations strictly grounded in the numerical evidence without hallucinating?

---

## Objectives

- Ingest, inspect, and validate raw customer behaviour records (50,000 rows × 30 columns).
- Engineer domain-specific credit utilisation and spending deviation indicators.
- Train, evaluate, and serialize multiple regression models (Linear Regression, Random Forest, XGBoost).
- Persist an end-to-end preprocessing pipeline (`preprocessor.joblib`, `feature_columns.json`) to handle raw customer inference.
- Explain predictions globally and locally using SHAP values.
- Host local GGUF models on CUDA via `llama.cpp` server processes.
- Objectively benchmark Llama 3.2 vs. Qwen 2.5 on factual consistency, numerical accuracy, latency, and throughput.
- Deliver an interactive 7-stage analytical dashboard via Streamlit.

---

## Pipeline Architecture

```text
              Raw Customer Dataset (50,000 × 30)
                              |
                              v
                Data Quality & Range Validation
                              |
                              v
                        Data Cleaning
                              |
                              v
                  Exploratory Data Analysis
                              |
                              v
                     Feature Engineering
                  (+4 calculated variables)
                              |
                              v
             Saved Preprocessor Pipeline (39 features)
              (Encoding, Transformations, Column Order)
                              |
                              v
                       Machine Learning
                 /            |             \
                v             v              v
         Linear Regression  Random Forest   XGBoost (Winner: R²=0.9672)
                \             |              /
                 \            |             /
                              v
                     Explainable AI (SHAP)
                (Global Importance & Waterfalls)
                              |
                              v
                     Analytical Evidence
           (Metrics, Scores, Top Feature Attributions)
                              |
                              v
              Local LLM Inference (llama.cpp CUDA)
                 /                          \
                v                            v
          Llama 3.2 1B                 Qwen 2.5 1.5B
                \                            /
                 \                          /
                              v
           Deterministic LLM Evaluation & Benchmarking
          (Factual Consistency, Accuracy, Tokens/Sec)
                              |
                              v
                Interactive Streamlit Application
             (7 Stages: Data -> Model -> XAI -> LLM)
```

---

## Dataset

The project uses a synthetic credit-card customer dataset stored at:
`data/raw/synthetic_credit_card_customer_behavior_dataset.csv`

### Processing Dimensions Matrix:

| Stage                          |   Rows | Columns | Description                                              |
| ------------------------------ | -----: | ------: | -------------------------------------------------------- |
| **Raw dataset**                | 50,000 |      30 | Original customer records                                |
| **Cleaned dataset**            | 50,000 |      30 | Deduplicated and range-verified                          |
| **Feature-engineered dataset** | 50,000 |      34 | +4 domain spending and utilisation ratios                |
| **Encoded model matrix ($X$)** | 50,000 |      39 | 10 one-hot dummies (`Gender`, `Occupation`, `Card_Type`) |

### Key Metadata:

- **Target Variable**: `Credit_Score` (Range: 300 to 850)
- **Identifier**: `Customer_ID`
- **Predictive features**: 28 in raw dataset $\rightarrow$ 32 after feature engineering $\rightarrow$ 39 after one-hot encoding
- **Training records**: 40,000 (80%)
- **Test records**: 10,000 (20%)
- **Validation Audit**: 0 missing values, 0 duplicate records, 0 invalid domain ranges.

---

## Machine Learning Results

Three regression algorithms were trained on the identical 80/20 train/test split (`random_state=42`):

| Algorithm               |      MAE |      RMSE |   R² Score | Key Role                 |
| ----------------------- | -------: | --------: | ---------: | ------------------------ |
| **XGBoost Regressor**   | **9.93** | **12.46** | **0.9672** | **Production Champion**  |
| Linear Regression       |    10.17 |     12.75 |     0.9656 | High-performing baseline |
| Random Forest Regressor |    10.90 |     13.67 |     0.9605 | Non-linear ensemble      |

### Saved Preprocessing & Model Artifacts:

```text
models/ml/
├── xgb_model.json           # Trained XGBoost weights & booster configuration
├── preprocessor.joblib      # Fitted CustomerPreprocessor pipeline
├── feature_columns.json     # Ordered list of all 39 model feature names
├── xgb_X_test.csv          # Held-out 10,000 test feature matrix
└── xgb_y_test.csv          # Held-out 10,000 test ground-truth labels
```

---

## Explainable AI (SHAP)

Model interpretability was performed using `shap.Explainer` (PermutationExplainer) on the test cohort:

### Top Feature Drivers:

| Feature Name             | Mean Absolute SHAP | Direction    | Analytical Meaning                                            |
| ------------------------ | -----------------: | ------------ | ------------------------------------------------------------- |
| **`Payment_Ratio`**      |      **44.97 pts** | **Positive** | Higher payment ratios substantially elevate credit scores.    |
| **`Credit_Utilization`** |      **34.18 pts** | **Negative** | Heavy credit utilisation heavily penalises credit scores.     |
| `Cash_Advance_Amount`    |           4.41 pts | Negative     | Cash advances signal credit distress.                         |
| `EMI_Count`              |           4.38 pts | Negative     | High ongoing debt commitments lower creditworthiness.         |
| `Mobile_App_Login`       |           3.13 pts | Positive     | Engaged customer digital activity associates with lower risk. |

Saved visualizations and CSVs are located under `results/figures/` and `results/`.

---

## Local LLM Inference & Benchmarking

The project deploys compact GGUF instruction-tuned models locally via `llama.cpp` (`runtime/llama-cuda/llama-server.exe`) on dedicated CUDA GPU hardware:

- **Deployment Hardware**: NVIDIA GeForce RTX 4050 Laptop GPU (6,141 MiB VRAM), Driver 621.24
- **Meta Llama 3.2 1B Instruct** (`Llama-3.2-1B-Instruct-Q4_K_M.gguf` — 0.75 GB)
- **Qwen 2.5 1.5B Instruct** (`qwen2.5-1.5b-instruct-q4_k_m.gguf` — 1.04 GB)

### Verified Benchmark Results:

#### 1. Systematic Batch Evaluation Benchmark (300-token fixed generation):

| Model             | Task                 |    Latency | Benchmark Throughput | Factual Consistency | Numerical Accuracy | Quality Score | Rating   |
| ----------------- | -------------------- | ---------: | -------------------: | ------------------: | -----------------: | ------------: | -------- |
| **Llama 3.2 1B**  | Customer Explanation | **7.26 s** |        **41.34 t/s** |            **1.00** |           **1.00** |      **1.00** | **High** |
| **Qwen 2.5 1.5B** | Customer Explanation |     9.77 s |            30.72 t/s |            **1.00** |           **1.00** |      **1.00** | **High** |
| **Llama 3.2 1B**  | Model Comparison     | **7.25 s** |        **41.37 t/s** |                0.47 |               0.56 |          0.60 | Moderate |
| **Qwen 2.5 1.5B** | Model Comparison     |     9.23 s |            32.52 t/s |            **0.60** |           **1.00** |      **0.80** | **High** |

#### 2. Interactive Real-Time Inference Performance (Streamlit UI & API):

- **Meta Llama 3.2 1B Instruct**: **45.1 – 50.8 tokens/sec** (Server spin-up: 3.55s, response latency: 0.69 – 1.08s for 50–60 tokens).
- **Qwen 2.5 1.5B Instruct**: **38.2 – 43.1 tokens/sec** (Server spin-up: 2.57s, response latency: 0.80 – 3.92s for 60–150 tokens).

_Note on Throughput Variation: Higher interactive throughput (48–51 t/s) vs. batch benchmarks (31–41 t/s) is expected: shorter conversational generations take advantage of GPU boost clocks and lower KV-cache memory overhead._

_Benchmark artifacts are persisted in `results/metrics/llm_benchmark_results.csv` and `results/llm_outputs/`._

---

## Repository Structure

```text
data-science-llm-inference/
├── app/
│   └── streamlit_app.py              # 7-stage interactive dashboard
│
├── data/
│   ├── raw/
│   │   └── synthetic_credit_card_customer_behavior_dataset.csv
│   └── processed/
│       ├── cleaned_data.csv
│       └── feature_engineered_data.csv
│
├── models/
│   ├── llm/
│   │   ├── llama/                    # Llama 3.2 1B GGUF
│   │   └── qwen/                     # Qwen 2.5 1.5B GGUF
│   └── ml/
│       ├── xgb_model.json            # Trained XGBoost model
│       ├── preprocessor.joblib       # Fitted preprocessing artifact
│       ├── feature_columns.json      # 39-feature schema contract
│       ├── xgb_X_test.csv            # Test set features
│       └── xgb_y_test.csv            # Test set labels
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_quality.ipynb
│   ├── 03_data_cleaning.ipynb
│   ├── 04_eda.ipynb
│   ├── 05_feature_engineering.ipynb
│   ├── 06_modeling.ipynb
│   ├── 07_explainability.ipynb
│   └── 08_llm_evaluation.ipynb
│
├── prompts/
│   ├── analysis.txt                  # Structured single-customer prompt template
│   └── comparison.txt                # Multi-model evaluation prompt template
│
├── results/
│   ├── figures/                      # SHAP bar, scatter, summary, and waterfall plots
│   ├── llm_outputs/                  # Saved generated model explanations (.txt)
│   ├── metrics/
│   │   └── llm_benchmark_results.csv # Factual consistency, latency, and tokens/sec
│   ├── model_comparison.csv          # Linear vs RF vs XGBoost
│   ├── shap_feature_importance.csv
│   ├── shap_final_summary.csv
│   └── shap_sample_0_explanation.csv
│
├── runtime/
│   └── llama-cuda/                   # Local llama.cpp CUDA binaries
│
├── src/
│   ├── __init__.py                   # Package initialisation
│   ├── data_loader.py                # Dataset loaders and path resolution
│   ├── validation.py                 # Data quality and range validation rules
│   ├── cleaning.py                   # Automated deduplication and cleaning
│   ├── analysis.py                   # EDA summaries, skewness, and correlations
│   ├── feature_engineering.py        # Ratio and deviation feature formulas
│   ├── modeling.py                   # Preprocessing pipeline and XGBoost inference
│   ├── explainability.py             # SHAP loaders and driver summaries
│   ├── llm.py                        # LocalLLMManager server client
│   └── evaluation.py                 # Deterministic LLM evaluation rubrics
│
├── tests/
│   ├── test_validation.py            # Quality check unit tests
│   ├── test_cleaning.py              # Cleaning unit tests
│   ├── test_modeling.py              # Preprocessing, residual & model reproduction unit tests
│   ├── test_explainability.py        # SHAP attribution & live explanation unit tests
│   ├── test_llm.py                   # Deterministic evaluation unit tests
│   └── test_integration.py          # End-to-end pipeline integration tests
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Technology Stack

| Category                  | Tools & Libraries                                                |
| ------------------------- | ---------------------------------------------------------------- |
| **Programming Language**  | Python 3.10                                                      |
| **Data Manipulation**     | Pandas, NumPy                                                    |
| **Data Visualisation**    | Matplotlib, Seaborn                                              |
| **Machine Learning**      | Scikit-learn, XGBoost, Joblib                                    |
| **Explainable AI (XAI)**  | SHAP (TreeExplainer & PermutationExplainer)                      |
| **Local LLMs**            | Meta Llama 3.2 1B Instruct, Qwen 2.5 1.5B Instruct (GGUF Q4_K_M) |
| **Inference Engine**      | `llama.cpp` (CUDA-accelerated)                                   |
| **Interactive Dashboard** | Streamlit                                                        |
| **Testing & CI**          | Pytest (32 automated tests)                                      |

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/data-science-llm-inference.git
cd data-science-llm-inference
```

### 2. Activate Virtual Environment

PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Running the Streamlit Application

Launch the dashboard:

```powershell
streamlit run app/streamlit_app.py
```

Features available in the application:

1. **Data Explorer**: View raw, cleaned, and feature-engineered datasets with target distribution and customer identifier cardinality checks.
2. **Data Quality & Preparation**: Live validation checks, cleaning changes audit, mathematical formulas for all 4 engineered features, and one-hot encoding architecture.
3. **EDA Dashboard**: Target correlations, bivariate scatter plots, feature distributions, and skewness analysis without causal conflation.
4. **ML Workbench**: Real-time customer prediction, multi-model evaluation benchmarks, documented credit tiers, and residual analysis.
5. **Explainable AI (XAI)**: Global SHAP importance, direction impact, reference waterfall plots, and live SHAP explanations for custom customer inputs.
6. **LLM Analytical Playground**: Interactive local inference for Llama 3.2 1B and Qwen 2.5 1.5B with editable prompts, evidence injection, and sampling controls.
7. **LLM Evaluation & Benchmarking**: Systematic comparative benchmark table, performance charts, deterministic scoring rubrics, and cold-start vs inference latency analysis.

---

## Running Unit and Integration Tests

Run the full automated test suite:

```powershell
python -m pytest tests/ -v
```

All 32 tests validate data cleaning, range checks, single-row categorical alignment, model reproducibility, portable SHAP reconstruction, and deterministic LLM evaluation scoring.

---

## Current Status

- [x] Phase 0: Implementation & Reproducibility Audit completed.
- [x] Phase 1: Foundation (`.gitignore`, `requirements.txt`) configured.
- [x] Phase 2: Source modules (`src/`) implemented and covered by unit tests.
- [x] Phase 2: Preprocessor artifact (`preprocessor.joblib`, `feature_columns.json`) created.
- [x] Phase 3: Prompt engineering & local LLM benchmarking completed.
- [x] Phase 3: Notebook 08 updated with modular evaluation pipeline.
- [x] Phase 4: Streamlit interactive application implemented.
- [x] Phase 5: End-to-end integration test suite implemented (32/32 passing).
- [x] Phase 6: Documentation and metrics verified.

---

## Limitations & Reliability Audit

1. **Synthetic Nature**: The dataset is synthetic. While metrics are high ($R^2 \approx 0.9672$), performance reflects underlying synthetic generation relationships rather than real-world credit risk.
2. **Correlation vs. Causality**: SHAP values explain what features influenced the model; they do not establish economic causality.
3. **Local LLM Capabilities**: Compact 1B–1.5B models generate rapid inferences (30–45 tokens/sec) but require structured evidence prompts to prevent hallucination.

---

## Disclaimer

This is an educational and portfolio project demonstrating applied data science, explainable AI, and local LLM engineering. It is **not** a credit decisioning platform, financial advisory tool, or commercial lending service.
