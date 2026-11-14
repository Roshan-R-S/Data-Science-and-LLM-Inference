"""Exploratory Data Analysis (EDA) functions and statistical summaries.
Reproduces analytical routines from Notebook 04.
"""

from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np


def get_column_types(df: pd.DataFrame, id_col: str = "Customer_ID") -> Tuple[List[str], List[str]]:
    """Partition columns into numerical and categorical lists.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    id_col : str, optional
        Column name to exclude from categorical list. Default is 'Customer_ID'.

    Returns
    -------
    Tuple[List[str], List[str]]
        (numeric_columns, categorical_columns)
    """
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if id_col in categorical_cols:
        categorical_cols.remove(id_col)

    return numeric_cols, categorical_cols


def compute_skewness_analysis(df: pd.DataFrame, threshold: float = 1.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate feature skewness and identify highly skewed variables.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    threshold : float, optional
        Absolute skewness threshold. Default is 1.0.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (all_skewness_df, highly_skewed_df)
    """
    numeric_cols, _ = get_column_types(df)
    skewness_series = df[numeric_cols].skew().sort_values(ascending=False)
    
    skew_df = pd.DataFrame({"skewness": skewness_series})
    highly_skewed = skew_df[skew_df["skewness"].abs() > threshold]
    
    return skew_df, highly_skewed


def compute_correlations_with_target(
    df: pd.DataFrame,
    target_col: str = "Credit_Score",
    top_n: Optional[int] = 10
) -> pd.Series:
    """Calculate correlation of all numerical features with the target variable.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    target_col : str, optional
        Target column name. Default is 'Credit_Score'.
    top_n : int, optional
        Number of top correlated features to return. If None, returns all.

    Returns
    -------
    pd.Series
        Correlations sorted by absolute magnitude descending.
    """
    numeric_cols, _ = get_column_types(df)
    if target_col not in numeric_cols:
        raise ValueError(f"Target column '{target_col}' not found in numeric columns.")

    corrs = (
        df[numeric_cols]
        .corr()[target_col]
        .drop(target_col)
        .sort_values(key=abs, ascending=False)
    )

    if top_n is not None:
        return corrs.head(top_n)
    return corrs


def compute_category_target_summary(
    df: pd.DataFrame,
    cat_col: str,
    target_col: str = "Credit_Score"
) -> pd.DataFrame:
    """Calculate mean, median, std, and customer counts for target grouped by categorical feature.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    cat_col : str
        Categorical grouping column.
    target_col : str, optional
        Target column name. Default is 'Credit_Score'.

    Returns
    -------
    pd.DataFrame
        Aggregated summary sorted by mean target score descending.
    """
    summary = (
        df.groupby(cat_col)[target_col]
        .agg(count="count", mean="mean", median="median", std="std")
        .sort_values(by="mean", ascending=False)
        .reset_index()
    )
    return summary
