import pandas as pd
import numpy as np
from typing import Tuple, List, Optional

def load_and_clean_data(file_path: str, target_pollutants: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Loads raw air quality data, standardizes timestamps, and filters target pollutants.
    
    Args:
        file_path (str): Path to the raw CSV dataset.
        target_pollutants (List[str], optional): Target pollutants to retain (default: ['pm10', 'pm25']).
        
    Returns:
        pd.DataFrame: Cleaned dataframe with UTC DatetimeIndex.
    """
    if target_pollutants is None:
        target_pollutants = ['pm10', 'pm25']
        
    df = pd.read_csv(file_path)
    
    # Ensure column names match data contract specifications
    df.columns = df.columns.str.lower().str.strip()
    
    # Parse timestamps to UTC
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    elif 'datetime' in df.columns:
        df['timestamp'] = pd.to_datetime(df['datetime'], utc=True)
        df = df.rename(columns={'datetime': 'timestamp'})
    else:
        raise KeyError("Dataframe must contain a 'timestamp' or 'datetime' column.")
        
    # Filter target pollutants if parameter column exists
    if 'parameter' in df.columns:
        df['parameter'] = df['parameter'].str.lower().str.strip()
        df = df[df['parameter'].isin(target_pollutants)].copy()
        
    return df


def resample_to_hourly_grid(df: pd.DataFrame, city_col: str = 'city', 
                             param_col: str = 'parameter', 
                             val_col: str = 'value') -> pd.DataFrame:
    """
    Resamples irregularly sampled time-series data to a continuous hourly grid per city and parameter.
    
    Args:
        df (pd.DataFrame): Cleaned air quality dataframe.
        city_col (str): Column name for city identifiers.
        param_col (str): Column name for pollutant parameters.
        val_col (str): Column name for sensor readings.
        
    Returns:
        pd.DataFrame: Pivot table indexed by continuous hourly timestamp with (city, parameter) columns.
    """
    # Ensure timestamp is set as datetime index for resampling
    df_temp = df.copy()
    if not isinstance(df_temp.index, pd.DatetimeIndex):
        df_temp = df_temp.set_index('timestamp')
        
    # Pivot to wide format to handle multi-city multi-pollutant grids cleanly
    pivoted = df_temp.pivot_table(
        index=df_temp.index,
        columns=[city_col, param_col],
        values=val_col,
        aggfunc='mean'
    )
    
    # Resample each series to continuous hourly frequency
    hourly_df = pivoted.resample('1h').mean()
    
    # Linear interpolation with max limit of 3 consecutive missing hours to prevent data fabrication
    hourly_df = hourly_df.interpolate(method='time', limit=3)
    
    return hourly_df


def train_test_chronological_split(df: pd.DataFrame, 
                                    test_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs a strict chronological train/test split to prevent temporal data leakage.
    
    Args:
        df (pd.DataFrame): Prepared hourly time-series dataframe.
        test_ratio (float): Fraction of temporal data reserved for testing (default: 0.2).
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (train_df, test_df)
    """
    split_index = int(len(df) * (1 - test_ratio))
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()
    
    return train_df, test_df


if __name__ == "__main__":
    print("Preprocessing module initialized successfully.")