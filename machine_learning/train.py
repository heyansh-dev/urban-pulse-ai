import json
import os
import pandas as pd
import numpy as np
from typing import Dict, Tuple


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculates standard regression metrics: MAE, RMSE, and MAPE.
    
    Args:
        y_true (np.ndarray): Actual target values.
        y_pred (np.ndarray): Predicted target values.
        
    Returns:
        Dict[str, float]: Dictionary containing MAE, RMSE, and MAPE metrics.
    """
    # Filter out zeros to prevent division by zero in MAPE calculation
    mask = y_true != 0
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = float(np.mean(np.abs((y_true_clean - y_pred_clean) / y_true_clean)) * 100)

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE_percent": round(mape, 4)
    }


def evaluate_naive_persistence(test_df: pd.DataFrame, 
                                target_cols: list = ['pm10', 'pm25'], 
                                horizons: list = [1, 24]) -> Dict[str, Dict]:
    """
    Evaluates a Naive Persistence Baseline Model (y_hat_{t+h} = y_t) across specified horizons.
    
    Args:
        test_df (pd.DataFrame): Test dataset indexed by time.
        target_cols (list): Targets to evaluate.
        horizons (list): Forecasting horizons in hours (e.g., 1-hour ahead, 24-hour ahead).
        
    Returns:
        Dict[str, Dict]: Nested dictionary containing evaluation metrics for each target and horizon.
    """
    metrics_results = {}

    for target in target_cols:
        if target not in test_df.columns:
            continue
            
        metrics_results[target] = {}
        
        for horizon in horizons:
            # Naive persistence formula: predict target value from 'horizon' steps prior
            y_true = test_df[target].iloc[horizon:].values
            y_pred = test_df[target].shift(horizon).iloc[horizon:].values

            # Drop any NaN pairs resulting from shifting
            valid_mask = ~np.isnan(y_pred) & ~np.isnan(y_true)
            
            target_metrics = calculate_metrics(y_true[valid_mask], y_pred[valid_mask])
            metrics_results[target][f"horizon_{horizon}h"] = target_metrics
            
            print(f"📊 [{target.upper()} - {horizon}h Horizon Persistence] "
                  f"MAE: {target_metrics['MAE']} | RMSE: {target_metrics['RMSE']} | MAPE: {target_metrics['MAPE_percent']}%")

    return metrics_results


def save_baseline_artifacts(metrics: Dict[str, Dict], 
                            artifact_dir: str = "machine_learning/artifacts", 
                            filename: str = "baseline_metrics.json") -> str:
    """
    Saves baseline metric outputs into a structured JSON artifact file.
    
    Args:
        metrics (Dict): Generated baseline metrics.
        artifact_dir (str): Destination folder path.
        filename (str): Name of the json file.
        
    Returns:
        str: Absolute path of saved file.
    """
    os.makedirs(artifact_dir, exist_ok=True)
    file_path = os.path.join(artifact_dir, filename)

    with open(file_path, "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"\n✅ Artifact successfully saved to: {file_path}")
    return file_path


if __name__ == "__main__":
    print("Initializing Naive Persistence Baseline Execution...")
    
    # Simple synthetic test run validation
    date_range = pd.date_range(start="2026-01-01", periods=100, freq="1h")
    mock_data = pd.DataFrame({
        "pm10": np.sin(np.linspace(0, 10, 100)) * 20 + 50 + np.random.normal(0, 2, 100),
        "pm25": np.cos(np.linspace(0, 10, 100)) * 10 + 25 + np.random.normal(0, 1, 100)
    }, index=date_range)
    
    print("\nRunning smoke-test evaluation on synthetic series:")
    results = evaluate_naive_persistence(mock_data)
    save_baseline_artifacts(results)