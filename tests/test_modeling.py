"""Unit tests for src/modeling.py.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import pytest

from src.modeling import (
    CustomerPreprocessor,
    load_xgb_model,
    predict_credit_score,
    compute_regression_metrics,
    get_models_ml_dir
)


@pytest.fixture
def raw_customer_sample() -> dict:
    """Fixture providing a raw customer profile dictionary."""
    return {
        "Customer_ID": "CC999999",
        "Age": 35,
        "Gender": "M",
        "Annual_Income": 850000.0,
        "Occupation": "Private Employee",
        "Card_Type": "Silver",
        "Credit_Limit": 250000.0,
        "Card_Age_Months": 24,
        "Monthly_Spending": 35000.0,
        "Monthly_Transactions": 50,
        "Avg_Transaction_Value": 700.0,
        "Online_Shopping_Spending": 12000.0,
        "Grocery_Spending": 8000.0,
        "Fuel_Spending": 3000.0,
        "Dining_Spending": 4000.0,
        "Travel_Spending": 2000.0,
        "Entertainment_Spending": 3000.0,
        "Utility_Bill_Spending": 3000.0,
        "Outstanding_Balance": 45000.0,
        "Statement_Balance": 40000.0,
        "Payment_Amount": 70000.0,
        "Payment_Ratio": 0.85,
        "Credit_Utilization": 0.25,
        "Cash_Advance_Amount": 0.0,
        "EMI_Count": 1,
        "International_Transactions": 0,
        "Reward_Points_Earned": 1200,
        "Reward_Points_Redeemed": 300,
        "Mobile_App_Login": 20
    }


def test_customer_preprocessor_raw_dict(raw_customer_sample):
    """Test that preprocessor transforms a raw customer dictionary to exact 39 model columns."""
    preprocessor = CustomerPreprocessor.load()
    X_transformed = preprocessor.transform(raw_customer_sample)

    assert isinstance(X_transformed, pd.DataFrame)
    assert X_transformed.shape == (1, 39)
    assert "calculated_monthly_spending" in X_transformed.columns
    assert "calculated_utilization" in X_transformed.columns
    assert "Gender_M" in X_transformed.columns
    assert X_transformed.isnull().sum().sum() == 0


def test_predict_credit_score(raw_customer_sample):
    """Test end-to-end credit score prediction on raw input."""
    score = predict_credit_score(raw_customer_sample)
    assert isinstance(score, float)
    assert 300 <= score <= 850


def test_model_reproducibility():
    """Verify that the loaded XGBoost model produces the exact recorded metrics on test data."""
    ml_dir = get_models_ml_dir()
    X_test = pd.read_csv(ml_dir / "xgb_X_test.csv")
    y_test = pd.read_csv(ml_dir / "xgb_y_test.csv").squeeze()

    model = load_xgb_model()
    preds = model.predict(X_test)
    metrics = compute_regression_metrics(y_test, preds)

    # Validate against model_comparison.csv ground truth
    assert pytest.approx(metrics["MAE"], abs=1e-3) == 9.9335
    assert pytest.approx(metrics["RMSE"], abs=1e-3) == 12.4552
    assert pytest.approx(metrics["R2"], abs=1e-4) == 0.9672
