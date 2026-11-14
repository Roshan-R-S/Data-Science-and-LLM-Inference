"""Unit tests for src/validation.py.
"""

import pandas as pd
import numpy as np
import pytest

from src.validation import (
    check_missing_values,
    check_duplicates,
    validate_ranges,
    check_constant_columns,
    run_full_validation
)


@pytest.fixture
def sample_valid_df() -> pd.DataFrame:
    """Fixture providing clean and valid synthetic customer rows."""
    return pd.DataFrame({
        "Customer_ID": ["CC1000", "CC1001", "CC1002"],
        "Age": [25, 45, 60],
        "Gender": ["M", "F", "M"],
        "Payment_Ratio": [0.25, 0.85, 0.99],
        "Credit_Utilization": [0.15, 0.40, 0.75],
        "EMI_Count": [0, 2, 5],
        "Credit_Score": [550, 720, 680],
        "Annual_Income": [500000, 1200000, 800000]
    })


def test_check_missing_values(sample_valid_df):
    """Test missing value calculation on clean and dirty data."""
    summary = check_missing_values(sample_valid_df)
    assert summary["missing_count"].sum() == 0
    assert summary["missing_percentage"].sum() == 0.0

    # Inject missing value
    dirty_df = sample_valid_df.copy()
    dirty_df.loc[0, "Age"] = np.nan
    dirty_summary = check_missing_values(dirty_df)
    assert dirty_summary.loc["Age", "missing_count"] == 1
    assert dirty_summary.loc["Age", "missing_percentage"] > 0


def test_check_duplicates(sample_valid_df):
    """Test duplicate detection."""
    assert check_duplicates(sample_valid_df) == 0

    # Inject duplicate row
    duplicated_df = pd.concat([sample_valid_df, sample_valid_df.iloc[[0]]], ignore_index=True)
    assert check_duplicates(duplicated_df) == 1


def test_validate_ranges(sample_valid_df):
    """Test domain range validation for financial columns."""
    valid_summary = validate_ranges(sample_valid_df)
    assert valid_summary["invalid_records"].sum() == 0

    # Inject boundary violations
    dirty_df = sample_valid_df.copy()
    dirty_df.loc[0, "Age"] = 15          # Invalid: < 18
    dirty_df.loc[1, "Payment_Ratio"] = 1.5 # Invalid: > 1
    dirty_df.loc[2, "EMI_Count"] = -1      # Invalid: < 0

    dirty_summary = validate_ranges(dirty_df).set_index("check")
    assert dirty_summary.loc["Age", "invalid_records"] == 1
    assert dirty_summary.loc["Payment Ratio", "invalid_records"] == 1
    assert dirty_summary.loc["EMI Count", "invalid_records"] == 1


def test_check_constant_columns(sample_valid_df):
    """Test constant column detection."""
    assert len(check_constant_columns(sample_valid_df)) == 0

    # Add constant column
    sample_valid_df["Currency"] = "INR"
    constants = check_constant_columns(sample_valid_df)
    assert "Currency" in constants


def test_run_full_validation(sample_valid_df):
    """Test full validation suite dictionary result."""
    result = run_full_validation(sample_valid_df)
    assert result["total_rows"] == 3
    assert result["total_missing_values"] == 0
    assert result["total_duplicates"] == 0
    assert result["is_clean"] is True
