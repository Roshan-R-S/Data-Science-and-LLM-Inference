"""Unit tests for Explainable AI (SHAP) functions in src/explainability.py."""

import pytest
import pandas as pd
from src.explainability import (
    load_shap_feature_importance,
    load_shap_final_summary,
    load_sample_explanation,
    get_top_drivers_summary,
    get_available_figures,
    explain_custom_prediction,
)
from src.data_loader import load_raw_data


def test_load_shap_feature_importance():
    df = load_shap_feature_importance()
    assert isinstance(df, pd.DataFrame)
    assert "Feature" in df.columns
    assert "Mean_Absolute_SHAP" in df.columns
    assert len(df) == 39
    assert df["Mean_Absolute_SHAP"].iloc[0] >= df["Mean_Absolute_SHAP"].iloc[-1]


def test_load_shap_final_summary():
    df = load_shap_final_summary()
    assert isinstance(df, pd.DataFrame)
    assert "Feature" in df.columns
    assert "Mean_Absolute_SHAP" in df.columns
    assert "Direction" in df.columns
    assert len(df) == 10


def test_load_sample_explanation():
    df = load_sample_explanation(0)
    assert isinstance(df, pd.DataFrame)
    assert "Feature" in df.columns
    assert "SHAP_Value" in df.columns
    assert len(df) == 39


def test_get_top_drivers_summary():
    drivers = get_top_drivers_summary(top_n=5)
    assert len(drivers) == 5
    assert "feature" in drivers[0]
    assert "importance" in drivers[0]
    assert "direction" in drivers[0]


def test_get_available_figures():
    figures = get_available_figures()
    assert isinstance(figures, dict)
    assert "shap_summary_plot" in figures
    assert "shap_bar_plot" in figures
    assert "shap_waterfall_sample_0" in figures


def test_explain_custom_prediction():
    raw_sample = load_raw_data(nrows=1)
    cust_dict = raw_sample.iloc[0].to_dict()
    result = explain_custom_prediction(cust_dict, top_n=5)

    assert "base_value" in result
    assert "prediction" in result
    assert "explanation_df" in result
    assert "top_drivers" in result

    assert isinstance(result["base_value"], float)
    assert isinstance(result["prediction"], float)
    assert len(result["top_drivers"]) == 5

    # Check reconstruction property: base_value + sum(shap_values) ~= prediction
    total_shap = result["explanation_df"]["SHAP_Value"].sum()
    reconstructed = result["base_value"] + total_shap
    assert abs(reconstructed - result["prediction"]) < 0.1


def test_explain_custom_prediction_reconstruction_tolerance():
    """Verify that Base Value + Sum(SHAP) equals Model Prediction within 0.001 points."""
    raw_sample = load_raw_data(nrows=1)
    cust_dict = raw_sample.iloc[0].to_dict()
    cust_dict["Gender"] = "M"
    cust_dict["Occupation"] = "Self-Employed"
    cust_dict["Card_Type"] = "Platinum"

    result = explain_custom_prediction(cust_dict, top_n=10)
    total_shap = result["explanation_df"]["SHAP_Value"].sum()
    reconstructed = result["base_value"] + total_shap
    diff = abs(reconstructed - result["prediction"])
    assert diff < 0.001

    # Verify categorical feature values in explanation DataFrame
    exp_df = result["explanation_df"]
    assert exp_df.loc[exp_df["Feature"] == "Gender_M", "Feature_Value"].values[0] == 1.0
    assert exp_df.loc[exp_df["Feature"] == "Occupation_Self-Employed", "Feature_Value"].values[0] == 1.0
    assert exp_df.loc[exp_df["Feature"] == "Card_Type_Platinum", "Feature_Value"].values[0] == 1.0
