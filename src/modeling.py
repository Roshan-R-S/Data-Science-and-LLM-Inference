"""Machine learning modeling, preprocessing pipeline, evaluation, and inference utilities.
Reproduces model training, evaluation, and serialization from Notebook 06.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import json
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.data_loader import get_project_root, load_feature_engineered_data
from src.feature_engineering import engineer_features, ENGINEERED_FEATURE_NAMES


def get_models_ml_dir() -> Path:
    """Return path to ML models directory."""
    return get_project_root() / "models" / "ml"


def get_feature_columns_path() -> Path:
    """Return path to feature_columns.json."""
    return get_models_ml_dir() / "feature_columns.json"


def get_preprocessor_path() -> Path:
    """Return path to preprocessor.joblib."""
    return get_models_ml_dir() / "preprocessor.joblib"


def get_xgb_model_path() -> Path:
    """Return path to xgb_model.json."""
    return get_models_ml_dir() / "xgb_model.json"


class CustomerPreprocessor:
    """End-to-end preprocessor converting raw customer records into encoded model features.

    Handles:
    1. Validation of required input features.
    2. Automatic computation of engineered features if not already present.
    3. One-hot encoding of categorical columns ('Gender', 'Occupation', 'Card_Type').
    4. Exact alignment of columns against trained model feature names.
    """

    CATEGORICAL_COLS = ["Gender", "Occupation", "Card_Type"]
    ID_COL = "Customer_ID"
    TARGET_COL = "Credit_Score"

    # Fixed training-derived category levels with alphabetical reference (dropped first)
    CATEGORICAL_CATEGORIES: Dict[str, List[str]] = {
        "Gender": ["F", "M"],
        "Occupation": [
            "Business Owner",
            "Government Employee",
            "Private Employee",
            "Retired",
            "Self-Employed",
            "Student"
        ],
        "Card_Type": [
            "Basic",
            "Gold",
            "Platinum",
            "Signature",
            "Silver"
        ]
    }

    def __init__(self, feature_columns: Optional[List[str]] = None):
        """Initialize preprocessor with expected feature column order.

        Parameters
        ----------
        feature_columns : List[str], optional
            Exact ordered list of feature names. If None, loads from feature_columns.json.
        """
        if feature_columns is not None:
            self.feature_columns = feature_columns
        else:
            json_path = get_feature_columns_path()
            if json_path.exists():
                with open(json_path, "r", encoding="utf-8") as f:
                    self.feature_columns = json.load(f)
            else:
                self.feature_columns = []

    def fit(self, X: pd.DataFrame) -> "CustomerPreprocessor":
        """Learn encoded feature column names from training data.

        Parameters
        ----------
        X : pd.DataFrame
            Feature-engineered training DataFrame.

        Returns
        -------
        CustomerPreprocessor
            Fitted preprocessor instance.
        """
        # Exclude ID and target if present
        cols_to_drop = [c for c in [self.ID_COL, self.TARGET_COL] if c in X.columns]
        X_clean = X.drop(columns=cols_to_drop)

        # One-hot encode with drop_first=True
        X_encoded = pd.get_dummies(X_clean, columns=self.CATEGORICAL_COLS, drop_first=True)
        self.feature_columns = X_encoded.columns.tolist()
        return self

    def transform(self, customer_input: Union[Dict[str, Any], pd.DataFrame]) -> pd.DataFrame:
        """Transform raw customer input into aligned model features.

        Parameters
        ----------
        customer_input : Union[Dict[str, Any], pd.DataFrame]
            Single customer dictionary or DataFrame of customer records.

        Returns
        -------
        pd.DataFrame
            Preprocessed feature matrix ready for XGBoost prediction.
        """
        if isinstance(customer_input, dict):
            df = pd.DataFrame([customer_input])
        else:
            df = customer_input.copy()

        # Step 1: Compute engineered features if missing
        has_fe = all(f in df.columns for f in ENGINEERED_FEATURE_NAMES)
        if not has_fe:
            df = engineer_features(df)

        # Drop ID and target columns if present
        cols_to_drop = [c for c in [self.ID_COL, self.TARGET_COL] if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        # Step 2: One-hot encode categoricals using fixed training-time category levels
        existing_cats = [c for c in self.CATEGORICAL_COLS if c in df.columns]
        for c in existing_cats:
            known_cats = self.CATEGORICAL_CATEGORIES.get(c)
            if known_cats is not None:
                # Using pd.Categorical guarantees all category levels exist even for a single row
                df[c] = pd.Categorical(df[c], categories=known_cats)

        df_encoded = pd.get_dummies(df, columns=existing_cats, drop_first=True)

        # Step 3: Reindex to ensure identical feature columns and order as trained model
        if self.feature_columns:
            df_aligned = df_encoded.reindex(columns=self.feature_columns, fill_value=0.0)
        else:
            df_aligned = df_encoded

        # Convert to float
        return df_aligned.astype(float)

    def save(self, output_path: Optional[Path] = None) -> Path:
        """Save fitted preprocessor instance to disk."""
        if output_path is None:
            output_path = get_preprocessor_path()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, output_path)

        # Also save feature_columns.json for transparency
        json_path = output_path.parent / "feature_columns.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.feature_columns, f, indent=2)

        return output_path

    @classmethod
    def load(cls, model_path: Optional[Path] = None) -> "CustomerPreprocessor":
        """Load fitted preprocessor from disk."""
        if model_path is None:
            model_path = get_preprocessor_path()
        if not model_path.exists():
            # Fallback to initializing with feature_columns.json
            return cls()
        return joblib.load(model_path)


def load_xgb_model(model_path: Optional[Path] = None) -> XGBRegressor:
    """Load the trained XGBoost credit score regressor from JSON format.

    Parameters
    ----------
    model_path : Path, optional
        Path to xgb_model.json. Defaults to models/ml/xgb_model.json.

    Returns
    -------
    XGBRegressor
        Loaded XGBoost model instance.
    """
    if model_path is None:
        model_path = get_xgb_model_path()
    if not model_path.exists():
        raise FileNotFoundError(f"Trained XGBoost model not found at {model_path}")

    model = XGBRegressor()
    model.load_model(str(model_path))
    return model


def predict_credit_score(
    customer_input: Union[Dict[str, Any], pd.DataFrame],
    model: Optional[XGBRegressor] = None,
    preprocessor: Optional[CustomerPreprocessor] = None
) -> Union[float, np.ndarray]:
    """End-to-end prediction of credit score for raw customer input.

    Parameters
    ----------
    customer_input : Union[Dict[str, Any], pd.DataFrame]
        Raw customer feature dictionary or DataFrame.
    model : XGBRegressor, optional
        Loaded XGBoost model. If None, loads from default path.
    preprocessor : CustomerPreprocessor, optional
        Fitted preprocessor. If None, loads from default path.

    Returns
    -------
    Union[float, np.ndarray]
        Predicted credit score (single float if dict input, array if DataFrame).
    """
    if preprocessor is None:
        preprocessor = CustomerPreprocessor.load()
    if model is None:
        model = load_xgb_model()

    X_transformed = preprocessor.transform(customer_input)
    predictions = model.predict(X_transformed)

    if isinstance(customer_input, dict) or len(predictions) == 1:
        return float(predictions[0])
    return predictions


def compute_regression_metrics(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray]
) -> Dict[str, float]:
    """Calculate MAE, RMSE, and R2 regression metrics.

    Parameters
    ----------
    y_true : array-like
        Ground truth target values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    Dict[str, float]
        Dictionary with 'MAE', 'RMSE', and 'R2'.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2)
    }


def get_test_predictions_and_residuals(
    sample_size: Optional[int] = 1000,
    random_state: int = 42
) -> pd.DataFrame:
    """Generate actual vs predicted test set samples and compute residuals.

    Parameters
    ----------
    sample_size : Optional[int], optional
        Number of test samples to return (for fast visualization). Default is 1000.
        If None, returns all 10,000 test samples.
    random_state : int, optional
        Random seed for subsampling. Default is 42.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'Actual', 'Predicted', 'Residual', 'Abs_Residual'.
    """
    ml_dir = get_models_ml_dir()
    x_test_path = ml_dir / "xgb_X_test.csv"
    y_test_path = ml_dir / "xgb_y_test.csv"

    if x_test_path.exists() and y_test_path.exists():
        X_test = pd.read_csv(x_test_path)
        y_test = pd.read_csv(y_test_path).squeeze()
    else:
        fe_data = load_feature_engineered_data()
        y = fe_data["Credit_Score"]
        preprocessor = CustomerPreprocessor.load()
        X = preprocessor.transform(fe_data)
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )

    if sample_size is not None and sample_size < len(X_test):
        idx = np.random.RandomState(random_state).choice(len(X_test), size=sample_size, replace=False)
        X_sub = X_test.iloc[idx]
        y_sub = y_test.iloc[idx]
    else:
        X_sub = X_test
        y_sub = y_test

    model = load_xgb_model()
    y_pred = model.predict(X_sub)

    res_df = pd.DataFrame({
        "Actual": y_sub.values,
        "Predicted": y_pred,
        "Residual": y_sub.values - y_pred,
        "Abs_Residual": np.abs(y_sub.values - y_pred)
    })
    return res_df
