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
    get_test_predictions_and_residuals,
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


def test_get_test_predictions_and_residuals():
    """Verify residual generation and structure."""
    df = get_test_predictions_and_residuals(sample_size=200)
    assert len(df) == 200
    assert set(df.columns) == {"Actual", "Predicted", "Residual", "Abs_Residual"}
    # Residuals mean should be close to 0
    assert abs(df["Residual"].mean()) < 5.0
    # Abs residual mean should be close to MAE (~9.9)
    assert 7.0 < df["Abs_Residual"].mean() < 13.0


def test_single_row_categorical_encoding(raw_customer_sample):
    """Verify that single-record categorical inputs activate exact dummy columns."""
    preprocessor = CustomerPreprocessor.load()

    # 1. Test Gender M vs F
    cust_m = raw_customer_sample.copy()
    cust_m["Gender"] = "M"
    X_m = preprocessor.transform(cust_m)
    assert X_m["Gender_M"].iloc[0] == 1.0

    cust_f = raw_customer_sample.copy()
    cust_f["Gender"] = "F"
    X_f = preprocessor.transform(cust_f)
    assert X_f["Gender_M"].iloc[0] == 0.0

    # 2. Test Occupation encoding
    cust_occ = raw_customer_sample.copy()
    cust_occ["Occupation"] = "Self-Employed"
    X_occ = preprocessor.transform(cust_occ)
    assert X_occ["Occupation_Self-Employed"].iloc[0] == 1.0
    assert X_occ["Occupation_Private Employee"].iloc[0] == 0.0
    assert X_occ["Occupation_Government Employee"].iloc[0] == 0.0

    # Reference Occupation (Business Owner) should have all occupation dummies as 0.0
    cust_ref_occ = raw_customer_sample.copy()
    cust_ref_occ["Occupation"] = "Business Owner"
    X_ref_occ = preprocessor.transform(cust_ref_occ)
    occ_dummies = [c for c in X_ref_occ.columns if c.startswith("Occupation_")]
    assert all(X_ref_occ[c].iloc[0] == 0.0 for c in occ_dummies)

    # 3. Test Card_Type encoding
    cust_card = raw_customer_sample.copy()
    cust_card["Card_Type"] = "Platinum"
    X_card = preprocessor.transform(cust_card)
    assert X_card["Card_Type_Platinum"].iloc[0] == 1.0
    assert X_card["Card_Type_Gold"].iloc[0] == 0.0

    # 4. Test Unknown Category handles safely without error
    cust_unk = raw_customer_sample.copy()
    cust_unk["Occupation"] = "Astronaut"
    cust_unk["Card_Type"] = "Diamond"
    X_unk = preprocessor.transform(cust_unk)
    assert X_unk.shape == (1, 39)
    assert all(X_unk[c].iloc[0] == 0.0 for c in occ_dummies)


def test_batch_vs_single_row_consistency():
    """Verify that batch transform and single-row transform produce identical feature values."""
    from src.data_loader import load_raw_data

    raw_batch = load_raw_data(nrows=5)
    preprocessor = CustomerPreprocessor.load()

    batch_transformed = preprocessor.transform(raw_batch)

    for i in range(len(raw_batch)):
        single_row = raw_batch.iloc[[i]].to_dict(orient="records")[0]
        single_transformed = preprocessor.transform(single_row)

        np.testing.assert_allclose(
            batch_transformed.iloc[i].values,
            single_transformed.iloc[0].values,
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"Discrepancy between batch and single-row transform for row {i}"
        )
