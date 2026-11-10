import pandas as pd
import numpy as np
from typing import Tuple, List

def create_supervised_matrices(df: pd.DataFrame, 
                               target_col: str, 
                               horizon: int = 1) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Constructs feature matrix X and target vector y for a specific forecast horizon h.
    
    Args:
        df (pd.DataFrame): Engineered feature dataframe (output from feature_engineering.py).
        target_col (str): Target variable to predict (e.g., 'pm10' or 'pm25').
        horizon (int): Forecasting horizon in hours (e.g., 1 for t+1, 24 for t+24).
        
    Returns:
        Tuple[pd.DataFrame, pd.Series]: (X, y) aligned chronologically.
    """
    df_temp = df.copy()
    
    # Target y is shifted backward by the forecasting horizon offset h
    y = df_temp[target_col].shift(-horizon)
    
    # Drop non-feature target columns or metadata if multi-city column structure exists
    X = df_temp.copy()
    
    # Align indices after shifting target
    valid_mask = ~y.isna()
    X = X.loc[valid_mask]
    y = y.loc[valid_mask]
    
    return X, y

if __name__ == "__main__":
    print("Model utilities module initialized successfully.")