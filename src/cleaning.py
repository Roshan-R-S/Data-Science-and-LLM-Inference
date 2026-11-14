"""Data cleaning pipeline for credit card customer dataset.
Reproduces the exact cleaning transformations from Notebook 03.
"""

from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import pandas as pd

from src.validation import run_full_validation
from src.data_loader import get_cleaned_data_path


def clean_customer_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean the raw customer dataset by removing duplicates and ensuring schema integrity.

    Parameters
    ----------
    df : pd.DataFrame
        Input raw customer DataFrame.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        Cleaned DataFrame and final validation summary dictionary.
    """
    df_clean = df.copy()

    # Drop duplicate records
    df_clean = df_clean.drop_duplicates()

    # Compile validation metrics
    validation_report = run_full_validation(df_clean)

    return df_clean, validation_report


def save_cleaned_data(df_clean: pd.DataFrame, output_path: Optional[Path] = None) -> Path:
    """Save cleaned DataFrame to CSV.

    Parameters
    ----------
    df_clean : pd.DataFrame
        Cleaned DataFrame to save.
    output_path : Path, optional
        Custom path to save file. Defaults to data/processed/cleaned_data.csv.

    Returns
    -------
    Path
        Destination path where file was saved.
    """
    if output_path is None:
        output_path = get_cleaned_data_path()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_path, index=False)
    return output_path
