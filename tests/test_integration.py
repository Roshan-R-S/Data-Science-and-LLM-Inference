"""End-to-end integration tests for the Credit Score & LLM analytical pipeline.
"""

import pandas as pd
import numpy as np
import pytest

from src.data_loader import load_raw_data, load_cleaned_data
from src.cleaning import clean_customer_data
from src.feature_engineering import engineer_features
from src.modeling import CustomerPreprocessor, load_xgb_model, predict_credit_score
from src.explainability import load_shap_final_summary, load_sample_explanation


def test_end_to_end_pipeline_single_customer():
    """Verify raw customer data flows through cleaning, feature engineering, and model inference."""
    # 1. Take a raw customer row from the real raw dataset
    raw_df = load_raw_data(nrows=5)
    sample_raw = raw_df.iloc[0].to_dict()

    # 2. Predict credit score directly from raw dictionary
    pred_score = predict_credit_score(sample_raw)

    assert isinstance(pred_score, float)
    assert 300.0 <= pred_score <= 850.0

    # Verify prediction is close to ground-truth actual Credit_Score
    actual_score = float(sample_raw["Credit_Score"])
    # Our XGBoost model has MAE ~ 9.9 points on this dataset
    assert abs(pred_score - actual_score) < 30.0


def test_end_to_end_pipeline_batch():
    """Verify batch processing through preprocessor and XGBoost model."""
    raw_df = load_raw_data(nrows=20)
    preprocessor = CustomerPreprocessor.load()
    model = load_xgb_model()

    X_transformed = preprocessor.transform(raw_df)
    assert X_transformed.shape == (20, 39)

    predictions = model.predict(X_transformed)
    assert len(predictions) == 20
    assert all(300 <= p <= 850 for p in predictions)


def test_shap_artifacts_consistency():
    """Verify SHAP summary table matches expected drivers."""
    shap_summary = load_shap_final_summary()
    assert len(shap_summary) >= 5
    assert shap_summary.iloc[0]["Feature"] == "Payment_Ratio"
    assert shap_summary.iloc[0]["Direction"] == "Positive"
    assert shap_summary.iloc[1]["Feature"] == "Credit_Utilization"
    assert shap_summary.iloc[1]["Direction"] == "Negative"

    sample_0 = load_sample_explanation(0)
    assert "Feature" in sample_0.columns
    assert "SHAP_Value" in sample_0.columns
