"""Explainable AI (XAI) and SHAP interpretation utilities.
Reproduces SHAP analysis routines and artifact loaders from Notebook 07.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np

from src.data_loader import get_project_root
from src.modeling import load_xgb_model, CustomerPreprocessor


def _patch_shap_xgboost_loader() -> bool:
    """Defensive runtime compatibility patch for SHAP 0.49+ and XGBoost 3.x.

    In XGBoost 3.x, learner_model_param['base_score'] is serialized in UBJSON as
    '[6.28973E2]' (a string formatted with brackets). SHAP's XGBTreeModelLoader
    attempts float(learner_model_param['base_score']), which raises:
    ValueError: could not convert string to float: '[6.28973E2]'.

    This function safely wraps decode_ubjson_buffer in shap.explainers._tree to
    strip bracket wrappers from the base_score parameter at runtime without
    requiring manual file edits inside the virtual environment.
    """
    try:
        import shap.explainers._tree as shap_tree
        if hasattr(shap_tree, "decode_ubjson_buffer"):
            orig_decode = shap_tree.decode_ubjson_buffer
            if getattr(orig_decode, "_is_xgboost_compat_patched", False):
                return True

            def patched_decode(fd):
                jmodel = orig_decode(fd)
                try:
                    lmp = jmodel.get("learner", {}).get("learner_model_param", {})
                    if "base_score" in lmp and isinstance(lmp["base_score"], str):
                        lmp["base_score"] = lmp["base_score"].strip("[]")
                except Exception:
                    pass
                return jmodel

            patched_decode._is_xgboost_compat_patched = True
            shap_tree.decode_ubjson_buffer = patched_decode
            return True
    except Exception:
        pass
    return False


# Automatically apply compatibility patch on module import
_patch_shap_xgboost_loader()


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


def explain_custom_prediction(
    customer_input: Union[pd.DataFrame, Dict[str, Any]],
    model: Optional[Any] = None,
    preprocessor: Optional[CustomerPreprocessor] = None,
    top_n: int = 15
) -> Dict[str, Any]:
    """Calculate live SHAP feature attributions for a single customer prediction.

    Parameters
    ----------
    customer_input : Union[pd.DataFrame, Dict[str, Any]]
        Single customer features (raw or engineered).
    model : Optional[Any]
        Loaded XGBoost model. If None, loaded via load_xgb_model().
    preprocessor : Optional[CustomerPreprocessor]
        Fitted preprocessor. If None, loaded via CustomerPreprocessor().
    top_n : int, optional
        Number of top features to return in driver summary. Default is 15.

    Returns
    -------
    Dict[str, Any]
        Dictionary with:
        - 'base_value': float (expected value / baseline prediction)
        - 'prediction': float (model predicted credit score)
        - 'explanation_df': pd.DataFrame with Feature, Feature_Value, SHAP_Value, Absolute_SHAP, Direction
        - 'top_drivers': List[Dict] of top contributors
    """
    import shap

    if model is None:
        model = load_xgb_model()
    if preprocessor is None:
        preprocessor = CustomerPreprocessor()

    if isinstance(customer_input, dict):
        df_input = pd.DataFrame([customer_input])
    else:
        df_input = customer_input.copy()

    X = preprocessor.transform(df_input)

    # Use TreeExplainer for fast, exact tree attribution
    explainer = shap.TreeExplainer(model)
    shap_obj = explainer(X)

    raw_base = explainer.expected_value
    base_value = float(np.squeeze(raw_base))

    shap_values = shap_obj.values[0]
    feature_values = X.iloc[0].values
    prediction = float(model.predict(X)[0])

    explanation_df = pd.DataFrame({
        "Feature": X.columns,
        "Feature_Value": feature_values,
        "SHAP_Value": shap_values,
        "Absolute_SHAP": np.abs(shap_values)
    })
    explanation_df["Direction"] = explanation_df["SHAP_Value"].apply(
        lambda v: "Positive" if v >= 0 else "Negative"
    )
    explanation_df = explanation_df.sort_values(
        by="Absolute_SHAP", ascending=False
    ).reset_index(drop=True)

    top_drivers = []
    for _, row in explanation_df.head(top_n).iterrows():
        top_drivers.append({
            "feature": str(row["Feature"]),
            "value": float(row["Feature_Value"]),
            "shap_value": float(row["SHAP_Value"]),
            "importance": float(row["Absolute_SHAP"]),
            "direction": str(row["Direction"])
        })

    return {
        "base_value": base_value,
        "prediction": prediction,
        "explanation_df": explanation_df,
        "top_drivers": top_drivers
    }
