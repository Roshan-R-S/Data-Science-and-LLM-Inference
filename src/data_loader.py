"""Data loading and path resolution utilities for the Credit Score analytical pipeline.
"""

from pathlib import Path
from typing import Optional
import pandas as pd


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    # Since this file is in src/, project root is its parent directory
    return Path(__file__).resolve().parent.parent


def get_data_dir() -> Path:
    """Return the absolute path to the data directory."""
    return get_project_root() / "data"


def get_raw_data_path() -> Path:
    """Return path to the raw customer behaviour dataset."""
    return get_data_dir() / "raw" / "synthetic_credit_card_customer_behavior_dataset.csv"


def get_cleaned_data_path() -> Path:
    """Return path to the cleaned customer behaviour dataset."""
    return get_data_dir() / "processed" / "cleaned_data.csv"


def get_feature_engineered_data_path() -> Path:
    """Return path to the feature-engineered dataset."""
    return get_data_dir() / "processed" / "feature_engineered_data.csv"


def load_raw_data(nrows: Optional[int] = None) -> pd.DataFrame:
    """Load the raw customer behaviour dataset.

    Parameters
    ----------
    nrows : int, optional
        Number of rows to load. Default is all rows.

    Returns
    -------
    pd.DataFrame
        Raw dataset DataFrame.
    """
    path = get_raw_data_path()
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found at {path}")
    return pd.read_csv(path, nrows=nrows)


def load_cleaned_data(nrows: Optional[int] = None) -> pd.DataFrame:
    """Load the cleaned customer behaviour dataset.

    Parameters
    ----------
    nrows : int, optional
        Number of rows to load. Default is all rows.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset DataFrame.
    """
    path = get_cleaned_data_path()
    if not path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at {path}")
    return pd.read_csv(path, nrows=nrows)


def load_feature_engineered_data(nrows: Optional[int] = None) -> pd.DataFrame:
    """Load the feature-engineered customer behaviour dataset.

    Parameters
    ----------
    nrows : int, optional
        Number of rows to load. Default is all rows.

    Returns
    -------
    pd.DataFrame
        Feature-engineered dataset DataFrame.
    """
    path = get_feature_engineered_data_path()
    if not path.exists():
        raise FileNotFoundError(f"Feature-engineered dataset not found at {path}")
    return pd.read_csv(path, nrows=nrows)
