"""Explainable AI (XAI) and SHAP interpretation utilities.
Reproduces SHAP analysis routines and artifact loaders from Notebook 07.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np

from src.data_loader import get_project_root


def get_results_dir() -> Path:
    """Return path to results directory."""
    return get_project_root() / "results"


def get_figures_dir() -> Path:
    """Return path to results/figures directory."""
    return get_results_dir() / "figures"


def load_shap_feature_importance() -> pd.DataFrame:
    """Load the precomputed global SHAP feature importance table.

    Returns
    -------
    pd.DataFrame
        Table with 'Feature' and 'Mean_Absolute_SHAP', sorted descending.
    """
    path = get_results_dir() / "shap_feature_importance.csv"
    if not path.exists():
        raise FileNotFoundError(f"SHAP feature importance not found at {path}")
    return pd.read_csv(path)


def load_shap_final_summary() -> pd.DataFrame:
    """Load the top-10 SHAP summary table with feature contribution direction.

    Returns
    -------
    pd.DataFrame
        Table with 'Feature', 'Mean_Absolute_SHAP', and 'Direction' ('Positive' / 'Negative').
    """
    path = get_results_dir() / "shap_final_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"SHAP summary not found at {path}")
    return pd.read_csv(path)


def load_sample_explanation(sample_idx: int = 0) -> pd.DataFrame:
    """Load individual customer SHAP explanation table.

    Returns
    -------
    pd.DataFrame
        Table with 'Feature', 'Feature_Value', 'SHAP_Value', 'Absolute_SHAP'.
    """
    path = get_results_dir() / f"shap_sample_{sample_idx}_explanation.csv"
    if not path.exists():
        # Fallback to sample 0
        path = get_results_dir() / "shap_sample_0_explanation.csv"
    return pd.read_csv(path)


def get_top_drivers_summary(top_n: int = 5) -> List[Dict[str, Any]]:
    """Return structured list of top SHAP drivers for LLM prompt ingestion.

    Parameters
    ----------
    top_n : int, optional
        Number of top features. Default is 5.

    Returns
    -------
    List[Dict[str, Any]]
        List of dicts with feature name, mean SHAP magnitude, and impact direction.
    """
    summary_df = load_shap_final_summary().head(top_n)
    drivers = []
    for _, row in summary_df.iterrows():
        drivers.append({
            "feature": str(row["Feature"]),
            "importance": float(row["Mean_Absolute_SHAP"]),
            "direction": str(row["Direction"])
        })
    return drivers


def get_available_figures() -> Dict[str, Path]:
    """Return dictionary of available precomputed SHAP visualization paths."""
    fig_dir = get_figures_dir()
    figures = {}
    for plot_file in fig_dir.glob("*.png"):
        figures[plot_file.stem] = plot_file
    return figures
