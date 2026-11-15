"""Unit tests for src/llm.py and src/evaluation.py.
Uses mock responses for LLM evaluation logic to keep test execution fast and deterministic.
"""

import pytest
from src.llm import LocalLLMManager
from src.evaluation import (
    extract_numbers_from_text,
    evaluate_factual_consistency,
    evaluate_numerical_accuracy,
    evaluate_response_quality
)


def test_extract_numbers_from_text():
    """Test extracting integer and float numbers from analytical prose."""
    text = "The predicted credit score is 680 with an R2 of 0.967 and MAE of 9.93."
    numbers = extract_numbers_from_text(text)
    assert 680.0 in numbers
    assert 0.967 in numbers
    assert 9.93 in numbers


def test_evaluate_factual_consistency_positive():
    """Test factual consistency when key drivers and directions are accurately cited."""
    good_text = (
        "The model predicts a high credit score primarily due to a strong payment ratio, "
        "which significantly increases the score. In contrast, higher credit utilization "
        "imposes a penalty and reduces the score."
    )
    result = evaluate_factual_consistency(
        good_text,
        top_features=["Payment_Ratio", "Credit_Utilization"]
    )
    assert "Payment_Ratio" in result["cited_features"]
    assert "Credit_Utilization" in result["cited_features"]
    assert result["factual_consistency_score"] >= 0.8


def test_evaluate_factual_consistency_missing_features():
    """Test factual consistency score drops when features are omitted."""
    poor_text = "The customer has a nice profile and will probably repay their loan."
    result = evaluate_factual_consistency(
        poor_text,
        top_features=["Payment_Ratio", "Credit_Utilization"]
    )
    assert len(result["cited_features"]) == 0
    assert result["factual_consistency_score"] == 0.0


def test_evaluate_numerical_accuracy():
    """Test matching extracted numbers against expected anchors."""
    text = "The predicted credit score is 645.2 points with MAE of 9.93."
    result = evaluate_numerical_accuracy(
        text,
        expected_numbers=[645.0, 9.93],
        tolerance=0.01
    )
    assert result["matched_count"] == 2
    assert result["accuracy_score"] == 1.0


def test_evaluate_response_quality_composite():
    """Test overall composite score calculation."""
    text = (
        "Based on the XGBoost model results, the customer's predicted credit score is 650. "
        "The primary positive factor is their payment ratio, which increases the score. "
        "On the other hand, a higher credit utilization reduces the score."
    )
    result = evaluate_response_quality(
        text,
        expected_numbers=[650.0],
        top_features=["Payment_Ratio", "Credit_Utilization"]
    )
    assert result["composite_quality_score"] > 0.7
    assert result["rating"] in ["Moderate", "High"]


def test_llm_manager_model_paths():
    """Test that LocalLLMManager correctly resolves known model paths."""
    manager = LocalLLMManager()
    llama_path = manager.get_model_path("llama")
    qwen_path = manager.get_model_path("qwen")
    assert llama_path.exists()
    assert qwen_path.exists()

    with pytest.raises(ValueError):
        manager.get_model_path("non_existent_model")


def test_evaluate_factual_consistency_model_comparison():
    """Test task-specific factual consistency scoring for model comparison."""
    good_comp = (
        "The XGBoost Regressor is the top-performing model with the lowest MAE of 9.93 "
        "and highest R2 of 0.9672. The Linear Regression baseline achieved a competitive "
        "MAE of 10.17, while Random Forest achieved an MAE of 10.90."
    )
    res_good = evaluate_factual_consistency(good_comp, task="model_comparison")
    assert res_good["task"] == "model_comparison"
    assert res_good["correct_winner_identified"] is True
    assert len(res_good["cited_models"]) == 3
    assert res_good["factual_consistency_score"] >= 0.8

    # Hallucinated model comparison: falsely asserts Random Forest is the best
    bad_comp = (
        "The evaluation results indicate that the Random Forest Regressor model performed the best "
        "in terms of Mean Absolute Error (MAE) and R2. XGBoost and Linear Regression followed behind."
    )
    res_bad = evaluate_factual_consistency(bad_comp, task="model_comparison")
    assert res_bad["correct_winner_identified"] is False
    assert res_bad["factual_consistency_score"] < res_good["factual_consistency_score"]


def test_llm_manager_response_contract():
    """Verify that LocalLLMManager format_chat_prompt formats templates for both llama and qwen."""
    manager = LocalLLMManager()
    llama_prompt = manager.format_chat_prompt("Hello", model_key="llama")
    assert "<|start_header_id|>system<|end_header_id|>" in llama_prompt
    assert "<|eot_id|>" in llama_prompt

    qwen_prompt = manager.format_chat_prompt("Hello", model_key="qwen")
    assert "<|im_start|>system" in qwen_prompt
    assert "<|im_end|>" in qwen_prompt
