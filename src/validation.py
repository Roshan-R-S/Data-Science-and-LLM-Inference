"""Data quality checks and validation rules for credit card customer datasets.
Reproduces the exact validation checks from Notebook 02.
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np


def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze missing values count and percentage per column.

    Parameters
    ----------
    df : pd.DataFrame
        Input customer DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by column with 'missing_count' and 'missing_percentage',
        sorted descending by missing percentage.
    """
    summary = pd.DataFrame({
        "missing_count": df.isnull().sum(),
        "missing_percentage": (df.isnull().mean() * 100).round(4)
    })
    return summary.sort_values(by="missing_percentage", ascending=False)


def check_duplicates(df: pd.DataFrame) -> int:
    """Return count of fully duplicated rows in the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input customer DataFrame.

    Returns
    -------
    int
        Total count of duplicated rows.
    """
    return int(df.duplicated().sum())


def validate_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """Validate numerical value ranges for key credit features.

    Rules inspected:
    - Age: between 18 and 100
    - Payment_Ratio: between 0.0 and 1.0
    - Credit_Utilization: between 0.0 and 1.0
    - EMI_Count: non-negative (>= 0)
    - Credit_Score: between 300 and 850 (standard FICO/CIBIL range)

    Parameters
    ----------
    df : pd.DataFrame
        Input customer DataFrame.

    Returns
    -------
    pd.DataFrame
        Summary table with check name and count of invalid records.
    """
    checks = []
    
    # Age check
    if "Age" in df.columns:
        invalid_age = len(df[(df["Age"] < 18) | (df["Age"] > 100)])
        checks.append({"check": "Age", "invalid_records": invalid_age, "rule": "18 <= Age <= 100"})
    
    # Payment Ratio check
    if "Payment_Ratio" in df.columns:
        invalid_pr = len(df[(df["Payment_Ratio"] < 0) | (df["Payment_Ratio"] > 1)])
        checks.append({"check": "Payment Ratio", "invalid_records": invalid_pr, "rule": "0 <= Payment_Ratio <= 1"})
    
    # Credit Utilization check
    if "Credit_Utilization" in df.columns:
        invalid_cu = len(df[(df["Credit_Utilization"] < 0) | (df["Credit_Utilization"] > 1)])
        checks.append({"check": "Credit Utilization", "invalid_records": invalid_cu, "rule": "0 <= Credit_Utilization <= 1"})
    
    # EMI Count check
    if "EMI_Count" in df.columns:
        invalid_emi = len(df[df["EMI_Count"] < 0])
        checks.append({"check": "EMI Count", "invalid_records": invalid_emi, "rule": "EMI_Count >= 0"})
    
    # Credit Score check
    if "Credit_Score" in df.columns:
        invalid_cs = len(df[(df["Credit_Score"] < 300) | (df["Credit_Score"] > 850)])
        checks.append({"check": "Credit Score", "invalid_records": invalid_cs, "rule": "300 <= Credit_Score <= 850"})
    
    return pd.DataFrame(checks)


def check_constant_columns(df: pd.DataFrame) -> List[str]:
    """Identify columns with zero variance (single unique value).

    Parameters
    ----------
    df : pd.DataFrame
        Input customer DataFrame.

    Returns
    -------
    List[str]
        List of column names with nunique <= 1.
    """
    return [str(col) for col in df.columns if df[col].nunique() <= 1]


def run_full_validation(df: pd.DataFrame) -> Dict[str, Any]:
    """Execute complete data validation suite and return dictionary summary.

    Parameters
    ----------
    df : pd.DataFrame
        Input customer DataFrame.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing counts of rows, columns, nulls, duplicates,
        range violations, and constant columns.
    """
    missing_df = check_missing_values(df)
    range_df = validate_ranges(df)
    
    is_clean = bool(
        int(missing_df["missing_count"].sum()) == 0
        and check_duplicates(df) == 0
        and (range_df["invalid_records"].sum() == 0 if not range_df.empty else True)
    )
    
    return {
        "total_rows": int(len(df)),
        "total_columns": int(len(df.columns)),
        "total_missing_values": int(missing_df["missing_count"].sum()),
        "total_duplicates": check_duplicates(df),
        "constant_columns": check_constant_columns(df),
        "range_violations": range_df.to_dict(orient="records"),
        "is_clean": is_clean
    }
