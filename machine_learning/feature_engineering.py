import pandas as pd
import numpy as np
from typing import List, Optional, Tuple


def add_cyclical_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts time features (hour, day of week, month) and computes sine/cosine 
    cyclical encodings to represent periodic temporal behavior cleanly.
    
    Args:
        df (pd.DataFrame): Dataframe with a DatetimeIndex.
        
    Returns:
        pd.DataFrame: Dataframe augmented with temporal features.
    """
    df_out = df.copy()
    
    if not isinstance(df_out.index, pd.DatetimeIndex):
        raise TypeError("Dataframe index must be a DatetimeIndex to extract temporal features.")

    # Raw temporal features
    df_out['hour'] = df_out.index.hour
    df_out['dayofweek'] = df_out.index.dayofweek
    df_out['month'] = df_out.index.month

    # Cyclical Sine/Cosine encodings
    # Hour of day (24-hour cycle)
    df_out['sin_hour'] = np.sin(2 * np.pi * df_out['hour'] / 24.0)
    df_out['cos_hour'] = np.cos(2 * np.pi * df_out['hour'] / 24.0)

    # Day of week (7-day cycle)
    df_out['sin_dayofweek'] = np.sin(2 * np.pi * df_out['dayofweek'] / 7.0)
    df_out['cos_dayofweek'] = np.cos(2 * np.pi * df_out['dayofweek'] / 7.0)

    # Month of year (12-month cycle)
    df_out['sin_month'] = np.sin(2 * np.pi * df_out['month'] / 12.0)
    df_out['cos_month'] = np.cos(2 * np.pi * df_out['month'] / 12.0)

    return df_out


def create_lag_features(df: pd.DataFrame, 
                        target_cols: List[str], 
                        lags: List[int] = [1, 2, 3, 6, 12, 24]) -> pd.DataFrame:
    """
    Generates historical lag features for specified target columns.
    
    Args:
        df (pd.DataFrame): Input time-series dataframe.
        target_cols (List[str]): Columns to create lag features for.
        lags (List[int]): List of lag offsets in hours (default: [1, 2, 3, 6, 12, 24]).
        
    Returns:
        pd.DataFrame: Dataframe containing lag columns.
    """
    df_out = df.copy()
    
    for col in target_cols:
        if col in df_out.columns:
            for lag in lags:
                df_out[f'{col}_lag_{lag}h'] = df_out[col].shift(lag)
                
    return df_out


def create_rolling_features(df: pd.DataFrame, 
                           target_cols: List[str], 
                           windows: List[int] = [3, 6, 12, 24]) -> pd.DataFrame:
    """
    Generates rolling statistical aggregations (mean, standard deviation, min, max)
    for target columns without data leakage.
    
    Args:
        df (pd.DataFrame): Input time-series dataframe.
        target_cols (List[str]): Target columns to calculate rolling statistics for.
        windows (List[int]): List of rolling window sizes in hours (default: [3, 6, 12, 24]).
        
    Returns:
        pd.DataFrame: Dataframe with added rolling feature columns.
    """
    df_out = df.copy()
    
    for col in target_cols:
        if col in df_out.columns:
            for window in windows:
                # Shift by 1 prior to rolling to prevent inclusion of current timestep value
                shifted_series = df_out[col].shift(1)
                
                df_out[f'{col}_roll_mean_{window}h'] = shifted_series.rolling(window=window).mean()
                df_out[f'{col}_roll_std_{window}h'] = shifted_series.rolling(window=window).std()
                df_out[f'{col}_roll_min_{window}h'] = shifted_series.rolling(window=window).min()
                df_out[f'{col}_roll_max_{window}h'] = shifted_series.rolling(window=window).max()
                
    return df_out


def build_feature_pipeline(df: pd.DataFrame, 
                           target_cols: List[str], 
                           lags: List[int] = [1, 2, 3, 6, 12, 24], 
                           windows: List[int] = [3, 6, 12, 24],
                           drop_na: bool = True) -> pd.DataFrame:
    """
    Executes full feature engineering pipeline sequentially.
    
    Args:
        df (pd.DataFrame): Cleaned, hourly resampled dataframe with DatetimeIndex.
        target_cols (List[str]): List of target pollutant columns (e.g., ['pm10', 'pm25']).
        lags (List[int]): Lag steps to generate.
        windows (List[int]): Rolling window sizes to generate.
        drop_na (bool): Whether to drop initial NaN rows created by shifting/rolling.
        
    Returns:
        pd.DataFrame: Fully engineered feature set ready for model training.
    """
    # Step 1: Temporal & Cyclical features
    df_features = add_cyclical_temporal_features(df)
    
    # Step 2: Historical Lag features
    df_features = create_lag_features(df_features, target_cols=target_cols, lags=lags)
    
    # Step 3: Rolling Window statistics
    df_features = create_rolling_features(df_features, target_cols=target_cols, windows=windows)
    
    # Step 4: Handle initial window NaN values safely
    if drop_na:
        df_features = df_features.dropna().copy()
        
    return df_features


if __name__ == "__main__":
    print("Feature Engineering module initialized successfully.")