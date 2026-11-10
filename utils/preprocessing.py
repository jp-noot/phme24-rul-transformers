"""
Data Preprocessing Utilities for NASA C-MAPSS Dataset and RUL Estimation.
Includes target censoring, sliding window generation, and statistical feature extraction.
"""

from typing import Tuple, List, Optional
import random
import numpy as np
import pandas as pd
from sklearn import preprocessing, linear_model

"""
Data preprocessing pipeline for NASA C-MAPSS dataset (FD001 to FD004).
Handles raw text file parsing, column mapping, invariant feature removal, RUL calculation,
and optional data censoring simulation.
"""

from typing import List, Optional, Tuple
import os
import random
import numpy as np
import pandas as pd


# Default features to drop for FD001 (flat/constant sensor readings)
FD001_DROP_COLS = [
    "operational_setting_3",
    "sensor_1",
    "sensor_5",
    "sensor_10",
    "sensor_16",
    "sensor_18",
    "sensor_19",
    "sensor_6",
]


def load_raw_cmapss(file_path: str) -> pd.DataFrame:
    """
    Load raw space-separated C-MAPSS text files into a clean Pandas DataFrame.

    Args:
        file_path: Path to raw train_FD00x.txt or test_FD00x.txt file.

    Returns:
        pd.DataFrame: Formatted DataFrame with standard column headers.
    """
    df = pd.read_csv(file_path, sep=r"\s+", header=None)

    # Base column headers
    col_names = ["unit_number", "time_cycle"]
    col_names += [f"operational_setting_{i}" for i in range(1, 4)]
    col_names += [f"sensor_{i}" for i in range(1, 22)]

    # Map column headers to the loaded dataframe slicing valid columns
    df = df.iloc[:, : len(col_names)]
    df.columns = col_names

    return df


def calculate_rul(df: pd.DataFrame, max_rul_cap: Optional[int] = None) -> pd.DataFrame:
    """
    Calculate Remaining Useful Life (RUL) for each engine trajectory.

    Args:
        df: DataFrame containing 'unit_number' and 'time_cycle'.
        max_rul_cap: Optional upper cap for piece-wise linear RUL target (e.g., 125 or 130).

    Returns:
        pd.DataFrame: DataFrame updated with 'rul' column.
    """
    df_out = df.copy()
    max_cycles = df_out.groupby("unit_number")["time_cycle"].transform("max")
    df_out["rul"] = max_cycles - df_out["time_cycle"]

    if max_rul_cap is not None:
        df_out["rul"] = df_out["rul"].clip(upper=max_rul_cap)

    return df_out


def add_binary_target_horizons(
    df: pd.DataFrame, horizons: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Add binary classification targets for discrete temporal horizons.

    Args:
        df: DataFrame with existing 'rul' column.
        horizons: List of cycle thresholds (default: [10, 20, ..., 130]).

    Returns:
        pd.DataFrame: DataFrame augmented with target_{horizon} columns.
    """
    if horizons is None:
        horizons = [step * 10 for step in range(1, 14)]

    df_out = df.copy()
    for threshold in horizons:
        col_name = f"target_{threshold}"
        df_out[col_name] = (df_out["rul"] <= threshold).astype(int)

    return df_out


def apply_censoring(
    df: pd.DataFrame,
    num_censored_units: int,
    horizons: Optional[List[int]] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate data censoring on a subset of engine units by cutting off trajectories 
    at random historical cycles.

    Args:
        df: Input DataFrame with RUL targets.
        num_censored_units: Number of unique units to censor.
        horizons: Target horizon thresholds.
        seed: Random seed for reproducibility.

    Returns:
        pd.DataFrame: Censored DataFrame with updated target columns.
    """
    random.seed(seed)
    if horizons is None:
        horizons = [step * 10 for step in range(1, 14)]

    df_censored = df.copy()
    df_censored["censored"] = 0

    unique_units = list(df_censored["unit_number"].unique())
    sampled_units = random.sample(unique_units, min(num_censored_units, len(unique_units)))

    filtered_rows = []
    for unit_id, group in df_censored.groupby("unit_number"):
        if unit_id in sampled_units:
            # Pick a random cycle before final failure
            available_cycles = group["time_cycle"].iloc[:-1].tolist()
            if available_cycles:
                cutoff_cycle = random.choice(available_cycles)
                group = group[group["time_cycle"] <= cutoff_cycle].copy()
                group["censored"] = 1
        filtered_rows.append(group)

    df_censored = pd.concat(filtered_rows, ignore_index=True)

    # Adjust horizon targets for censored samples
    for threshold in horizons:
        col_name = f"target_{threshold}"
        df_censored[col_name] = 0
        
        # Valid uncensored targets
        uncensored_mask = (df_censored["rul"] <= threshold) & (df_censored["censored"] == 0)
        df_censored.loc[uncensored_mask, col_name] = 1

        # Mask out censored units exceeding threshold as NaN
        censored_mask = (df_censored["rul"] <= threshold) & (df_censored["censored"] == 1)
        df_censored.loc[censored_mask, col_name] = np.nan

    return df_censored


def preprocess_cmapss_file(
    raw_txt_path: str,
    output_csv_path: str,
    cols_to_drop: Optional[List[str]] = None,
    num_censored_units: int = 0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Execute complete end-to-end processing pipeline on a single C-MAPSS text file.

    Args:
        raw_txt_path: Path to raw input .txt file.
        output_csv_path: Path where preprocessed .csv will be saved.
        cols_to_drop: List of column names to remove (e.g. invariant sensors).
        num_censored_units: Number of units to subject to censoring simulation (0 to disable).
        seed: Random seed.

    Returns:
        pd.DataFrame: Fully processed DataFrame.
    """
    print(f"Processing raw data from: {raw_txt_path}")
    df = load_raw_cmapss(raw_txt_path)

    if cols_to_drop:
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    df = calculate_rul(df)
    df = add_binary_target_horizons(df)

    if num_censored_units > 0:
        df = apply_censoring(df, num_censored_units=num_censored_units, seed=seed)

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    print(f"Saved processed dataset to: {output_csv_path}")

    return df


def create_censored_target(df: pd.DataFrame, num_censored_units: int) -> pd.DataFrame:
    """
    Simulate data censoring by randomly selecting unit trajectories and cutting 
    them before failure. Adds binary target columns for discrete RUL intervals.

    Args:
        df: Input DataFrame containing 'unit_number', 'time_cycle', and 'rul'.
        num_censored_units: Number of unique units to censor randomly.

    Returns:
        pd.DataFrame with added 'censored' column and target horizon columns.
    """
    censored_units = random.sample(list(df['unit_number'].unique()), num_censored_units)
    df_censored = df.copy()
    df_censored['censored'] = 0

    for unit_id in censored_units:
        unit_data = df.loc[df['unit_number'] == unit_id]
        # Randomly select a cycle index prior to the final cycle for censoring
        cutoff_cycle = random.sample(list(unit_data['time_cycle'].iloc[:-1]), 1)[0]
        
        # Filter rows out after cutoff cycle for the selected unit
        keep_condition = (
            ((df_censored['unit_number'] == unit_id) & (df_censored['time_cycle'] <= cutoff_cycle)) | 
            (df_censored['unit_number'] != unit_id)
        )
        df_censored = df_censored.loc[keep_condition].reset_index(drop=True)
        df_censored.loc[df_censored['unit_number'] == unit_id, 'censored'] = 1

    # Create binned target horizons (intervals of 10 cycles up to 130)
    for horizon_step in range(1, 14):
        col_name = f'target_{horizon_step * 10}'
        df_censored[col_name] = 0.0
        
        # Target = 1 if RUL <= threshold and trajectory is fully uncensored
        df_censored.loc[
            (df_censored['rul'] <= (10 * horizon_step)) & (df_censored['censored'] == 0), 
            col_name
        ] = 1.0
        
        # Target = NaN for censored units exceeding threshold
        df_censored.loc[
            (df_censored['rul'] <= (10 * horizon_step)) & (df_censored['censored'] == 1), 
            col_name
        ] = np.nan

    return df_censored


def create_sliding_windows(
    df: pd.DataFrame, 
    qualitative_cols: List[str], 
    quantitative_cols: List[str], 
    target_col: str, 
    window_size: int,
    mean: Optional[pd.Series] = None,
    std: Optional[pd.Series] = None
) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
    """
    Standardize quantitative features and construct 3D sliding window arrays for time-series models.

    Args:
        df: Input DataFrame.
        qualitative_cols: List of categorical/unit identifier column names.
        quantitative_cols: List of numerical feature column names (sensors, operating conditions).
        target_col: Column name of the target variable (RUL).
        window_size: Temporal length of each sliding window sequence.
        mean: Optional pre-computed mean for normalization (useful for test set scaling).
        std: Optional pre-computed std for normalization (useful for test set scaling).

    Returns:
        X: Sequence tensor of shape (num_samples, window_size, num_features).
        y: Target array of shape (num_samples, 1).
        mean: Computed or applied mean series.
        std: Computed or applied std series.
    """
    df_scaled = df.copy()

    if mean is None or std is None:
        mean = df_scaled[quantitative_cols].mean()
        std = df_scaled[quantitative_cols].std()

    # Z-score normalization
    df_scaled[quantitative_cols] = (df_scaled[quantitative_cols] - mean) / std
    combined_df = pd.concat([df_scaled[quantitative_cols], df_scaled[qualitative_cols], df_scaled[target_col]], axis=1)

    X, y = [], []
    feature_cols = quantitative_cols + qualitative_cols

    for unit_id in combined_df['unit_number'].unique():
        unit_data = combined_df.loc[combined_df['unit_number'] == unit_id]
        n_samples = len(unit_data)
        
        if n_samples >= window_size:
            for step in range(n_samples - window_size + 1):
                X.append(unit_data.iloc[step : step + window_size][feature_cols].values)
                y.append(unit_data.iloc[step + window_size - 1][target_col])

    return np.array(X), np.array(y), mean, std


def extract_slope_feature(data: np.ndarray) -> List[float]:
    """Extract linear regression slope for each feature over time window."""
    regr = linear_model.LinearRegression()
    slopes = []
    time_indices = np.arange(data.shape[0]).reshape(-1, 1)
    
    for feat_idx in range(data.shape[1]):
        regr.fit(time_indices, data[:, feat_idx])
        slopes.extend(regr.coef_)
    return slopes


def extract_mean_feature(data: np.ndarray) -> List[float]:
    """Extract mean value for each feature over time window."""
    return [np.mean(data[:, feat_idx]) for feat_idx in range(data.shape[1])]


def extract_statistical_features(X: np.ndarray, window_size: int, num_quant_features: int) -> np.ndarray:
    """
    Enrich raw sliding window sequences with statistical trend features (slope and window mean).
    Expands sequence length from window_size to window_size + 2.

    Args:
        X: Raw sequence array of shape (N, window_size, total_features).
        window_size: Temporal length of window sequences.
        num_quant_features: Number of continuous quantitative features.

    Returns:
        np.ndarray: Enhanced feature array of shape (N, window_size + 2, total_features).
    """
    num_qualitative = X.shape[2] - num_quant_features
    slopes_list, means_list = [], []

    for sequence in X:
        quant_data = sequence[:, :num_quant_features]
        slopes_list.append(extract_slope_feature(quant_data))
        means_list.append(extract_mean_feature(quant_data))

    scaler_slopes = preprocessing.MinMaxScaler()
    scaled_slopes = scaler_slopes.fit_transform(slopes_list)

    scaler_means = preprocessing.MinMaxScaler()
    scaled_means = scaler_means.fit_transform(means_list)

    X_enhanced = []
    for i in range(len(X)):
        # 1. Traitement des variables quantitatives (dimensions: (window_size, num_quant_features))
        quant_base = X[i, :, :num_quant_features]
        slope_row = np.reshape(scaled_slopes[i], (1, num_quant_features))  # Forme 2D: (1, num_quant_features)
        mean_row = np.reshape(scaled_means[i], (1, num_quant_features))    # Forme 2D: (1, num_quant_features)
        
        # Empilement vertical sur l'axe temporel (résultat: (window_size + 2, num_quant_features))
        combined_quant = np.vstack((quant_base, slope_row, mean_row))
        
        # 2. Traitement des variables qualitatives (unit_number, time_cycle, RUL)
        qual_base = X[i, :, num_quant_features : num_quant_features + num_qualitative]
        qual_padded = np.vstack((qual_base, qual_base[0:2]))  # Remplissage pour correspondre à window_size + 2
        
        # 3. Fusion horizontale des colonnes quantitatives et qualitatives
        final_sample = np.hstack((combined_quant, qual_padded))
        X_enhanced.append(final_sample)

    return np.array(X_enhanced)