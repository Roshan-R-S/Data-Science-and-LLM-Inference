"""Feature engineering transformations for credit card customer analytical pipeline.
Reproduces exact feature definitions from Notebook 05.
"""

from pathlib import Path
from typing import Optional, List
import pandas as pd
import numpy as np

from src.data_loader import get_feature_engineered_data_path

ENGINEERED_FEATURE_NAMES: List[str] = [
    "calculated_monthly_spending",
    "spending_difference",
    "calculated_utilization",
    "spending_relative_difference"
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create four domain-specific financial features from raw/cleaned columns.

    Exact Transformations from Notebook 05:
    1. calculated_monthly_spending = Annual_Income / 12
    2. spending_difference = Monthly_Spending - calculated_monthly_spending
    3. calculated_utilization = Monthly_Spending / Credit_Limit
    4. spending_relative_difference = spending_difference / calculated_monthly_spending

    Parameters
    ----------
    df : pd.DataFrame
        Input customer DataFrame containing Annual_Income, Monthly_Spending, and Credit_Limit.

    Returns
    -------
    pd.DataFrame
        New DataFrame containing the 4 added engineered features.
    """
    df_fe = df.copy()

    # Required columns check
    required_cols = ["Annual_Income", "Monthly_Spending", "Credit_Limit"]
    missing_cols = [c for c in required_cols if c not in df_fe.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for feature engineering: {missing_cols}")

    # 1. Monthly benchmark derived from annual income
    df_fe["calculated_monthly_spending"] = df_fe["Annual_Income"] / 12.0

    # 2. Absolute spending difference relative to monthly income
    df_fe["spending_difference"] = (
        df_fe["Monthly_Spending"] - df_fe["calculated_monthly_spending"]
    )

    # 3. Monthly spending as a proportion of total credit limit
    # Safe division: if Credit_Limit is 0, replace with 0.0
    credit_limit_safe = df_fe["Credit_Limit"].replace(0, np.nan)
    df_fe["calculated_utilization"] = (
        df_fe["Monthly_Spending"] / credit_limit_safe
    ).fillna(0.0)

    # 4. Relative spending difference normalized by monthly income
    # Safe division: if calculated_monthly_spending is 0, replace with 0.0
    calc_spending_safe = df_fe["calculated_monthly_spending"].replace(0, np.nan)
    df_fe["spending_relative_difference"] = (
        df_fe["spending_difference"] / calc_spending_safe
    ).fillna(0.0)

    return df_fe


def save_feature_engineered_data(
    df_fe: pd.DataFrame,
    output_path: Optional[Path] = None
) -> Path:
    """Save feature-engineered DataFrame to CSV.

    Parameters
    ----------
    df_fe : pd.DataFrame
        DataFrame with engineered features.
    output_path : Path, optional
        Custom path to save file. Defaults to data/processed/feature_engineered_data.csv.

    Returns
    -------
    Path
        Destination path where file was saved.
    """
    if output_path is None:
        output_path = get_feature_engineered_data_path()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_fe.to_csv(output_path, index=False)
    return output_path
