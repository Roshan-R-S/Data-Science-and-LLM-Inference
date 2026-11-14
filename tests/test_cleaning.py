"""Unit tests for src/cleaning.py.
"""

from pathlib import Path
import pandas as pd
import pytest

from src.cleaning import clean_customer_data, save_cleaned_data


@pytest.fixture
def raw_df_with_duplicates() -> pd.DataFrame:
    """Fixture providing customer DataFrame with duplicated rows."""
    base_df = pd.DataFrame({
        "Customer_ID": ["CC1001", "CC1002"],
        "Age": [30, 45],
        "Gender": ["M", "F"],
        "Payment_Ratio": [0.5, 0.9],
        "Credit_Utilization": [0.3, 0.2],
        "EMI_Count": [1, 3],
        "Credit_Score": [600, 750]
    })
    # Concatenate to create a duplicate
    return pd.concat([base_df, base_df.iloc[[0]]], ignore_index=True)


def test_clean_customer_data_removes_duplicates(raw_df_with_duplicates):
    """Test that clean_customer_data removes duplicate rows."""
    assert len(raw_df_with_duplicates) == 3
    cleaned_df, report = clean_customer_data(raw_df_with_duplicates)
    assert len(cleaned_df) == 2
    assert report["total_duplicates"] == 0
    assert report["is_clean"] is True


def test_clean_customer_data_does_not_mutate_input(raw_df_with_duplicates):
    """Ensure the original DataFrame is not modified in-place."""
    original_len = len(raw_df_with_duplicates)
    _ = clean_customer_data(raw_df_with_duplicates)
    assert len(raw_df_with_duplicates) == original_len


def test_save_cleaned_data(tmp_path, raw_df_with_duplicates):
    """Test saving cleaned data to a file."""
    cleaned_df, _ = clean_customer_data(raw_df_with_duplicates)
    target_path = tmp_path / "cleaned_test.csv"
    saved_path = save_cleaned_data(cleaned_df, output_path=target_path)
    assert saved_path.exists()

    loaded_df = pd.read_csv(saved_path)
    assert len(loaded_df) == 2
