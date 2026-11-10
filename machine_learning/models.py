import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV


def get_model_pipeline(model_type: str = "xgboost", params: Dict[str, Any] = None):
    """
    Instantiates an XGBoost or LightGBM regressor model with default or custom parameters.
    
    Args:
        model_type (str): Type of model ('xgboost' or 'lightgbm').
        params (Dict[str, Any]): Hyperparameters for model initialization.
        
    Returns:
        Regressor model instance.
    """
    if params is None:
        params = {}

    if model_type.lower() == "xgboost":
        default_params = {
            "n_estimators": 100,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1
        }
        default_params.update(params)
        return XGBRegressor(**default_params)

    elif model_type.lower() == "lightgbm":
        default_params = {
            "n_estimators": 100,
            "learning_rate": 0.05,
            "max_depth": 6,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1
        }
        default_params.update(params)
        return LGBMRegressor(**default_params)

    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Choose 'xgboost' or 'lightgbm'.")


def tune_time_series_hyperparameters(X_train: pd.DataFrame, 
                                     y_train: pd.Series, 
                                     model_type: str = "xgboost", 
                                     n_splits: int = 3) -> Tuple[Any, Dict[str, Any]]:
    """
    Performs Time-Series Cross-Validation Grid Search to tune hyperparameters cleanly without temporal leakage.
    
    Args:
        X_train (pd.DataFrame): Training feature matrix.
        y_train (pd.Series): Training target values.
        model_type (str): Model to tune ('xgboost' or 'lightgbm').
        n_splits (int): Number of expanding window temporal splits.
        
    Returns:
        Tuple[Any, Dict[str, Any]]: Best estimator instance and dictionary of best parameters.
    """
    base_model = get_model_pipeline(model_type=model_type)
    
    # Expanding window time series cross-validator
    tscv = TimeSeriesSplit(n_splits=n_splits)

    if model_type.lower() == "xgboost":
        param_grid = {
            "n_estimators": [50, 100],
            "learning_rate": [0.03, 0.1],
            "max_depth": [4, 6]
        }
    else:  # LightGBM
        param_grid = {
            "n_estimators": [50, 100],
            "learning_rate": [0.03, 0.1],
            "max_depth": [4, 6],
            "num_leaves": [15, 31]
        }

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=tscv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        verbose=0
    )

    grid_search.fit(X_train, y_train)
    
    print(f"✅ Best {model_type.upper()} Parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_, grid_search.best_params_


def save_trained_model(model: Any, 
                       model_name: str, 
                       artifact_dir: str = "machine_learning/artifacts/models") -> str:
    """
    Saves trained model binary artifact using joblib.
    """
    os.makedirs(artifact_dir, exist_ok=True)
    file_path = os.path.join(artifact_dir, f"{model_name}.joblib")
    joblib.dump(model, file_path)
    print(f"📦 Saved model checkpoint to: {file_path}")
    return file_path


if __name__ == "__main__":
    print("Supervised Models module initialized successfully.")