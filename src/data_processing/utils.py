"""
Utility functions for NIR tomato spectroscopy data processing.

This module contains functions for processing NIR spectroscopy data:

Functions:
- identify_spectral_columns: Identifies wavelength columns vs metadata columns in datasets
- fix_column_names: Corrects column names if needed
- remove_duplicate_rows: Removes exact duplicate rows from the dataset
- add_sample_identifier: Creates a unique identifier for each tomato sample
- detect_outliers_zscore: Identifies outliers using Z-score statistical method
- detect_outliers_pca: Identifies outliers using PCA decomposition and statistics
- remove_constant_and_empty_columns: Removes uninformative columns from dataset
- split_data: Splits data into train, validation, and test sets
- filter_numeric_features: Filters dataset to keep only numeric features suitable for ML

These utilities support the transformers and provide core functionality for 
preprocessing NIR tomato spectroscopy datasets before modeling.
"""

import re
import logging
from typing import Tuple, List, Dict, Any, Optional, Union
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from .constants import (
    DEFAULT_PCA_COMPONENTS,
    DEFAULT_WAVELENGTH_PERCENT,
    DEFAULT_OUTLIER_THRESHOLD
)

def identify_spectral_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Identify wavelength (spectral) columns vs metadata columns.
    
    Args:
        df: DataFrame with spectral data
        
    Returns:
        Tuple containing (spectral_columns, non_spectral_columns)
    """
    # Pattern to identify wavelength columns (numbers with decimal points)
    spectral_pattern = re.compile(r'^(\d+[,\.]\d+|\d+)$')
    
    spectral_cols: List[str] = []
    for col in df.columns:
        col_str = str(col)
        if spectral_pattern.match(col_str):
            # Validates the string is a number
            try:
                float(col_str.replace(',', '.'))
                spectral_cols.append(col)
            except ValueError:
                pass
            
    non_spectral_cols: List[str] = [col for col in df.columns if col not in spectral_cols]
    
    return spectral_cols, non_spectral_cols

def fix_column_names(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Fix column names if needed.
    
    Args:
        df: DataFrame with NIR spectroscopy data
        verbose: Whether to log processing information
        
    Returns:
        DataFrame with corrected column names
    """
    # Configure logging
    logger = logging.getLogger(__name__)
    
    df_fixed = df.copy()
    
    # Example fix if needed
    # Currently just returns the original dataframe
    
    return df_fixed

def remove_duplicate_rows(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Identify and remove completely duplicate rows in the dataset.
    
    Args:
        df: DataFrame with NIR spectroscopy data
        verbose: Whether to log processing information
        
    Returns:
        DataFrame with duplicates removed
    """
    # Configure logging
    logger = logging.getLogger(__name__)
    
    # Search for completely duplicate rows (across all columns)
    duplicated = df.duplicated(keep=False)
    duplicate_count = duplicated.sum()
    
    if duplicate_count > 0:
        if verbose:
            logger.info(f"Found {duplicate_count} rows with exact duplicates")
        
        # Remove only the duplicates (keep first occurrences)
        df_cleaned = df.drop_duplicates(keep='first')
        removed_count = len(df) - len(df_cleaned)
        if verbose:
            logger.info(f"Removed {removed_count} duplicate rows. Remaining rows: {len(df_cleaned)}")
        
        return df_cleaned
    else:
        if verbose:
            logger.info("No duplicate rows found")
        return df

def add_sample_identifier(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a sample identifier column for easier grouping.
    
    Args:
        df: DataFrame with NIR spectroscopy data
        
    Returns:
        DataFrame with added sample identifier column
    """
    df_with_id = df.copy()
    
    # Create sample identifier column based on relevant columns
    if all(col in df.columns for col in ['SAMPLE NO', 'plant', 'wetlab ID']):
        df_with_id['Sample_ID'] = (
            df_with_id['SAMPLE NO'].astype(str) + '_' + 
            df_with_id['plant'].astype(str) + '_' + 
            df_with_id['wetlab ID'].astype(str)
        )
    else:
        # If those columns don't exist, create a simple index-based ID
        df_with_id['Sample_ID'] = [f"sample_{i}" for i in range(len(df))]
    
    return df_with_id

def detect_outliers_zscore(
    spectral_data: np.ndarray,
    threshold: float = DEFAULT_OUTLIER_THRESHOLD,
    wavelength_percent: float = DEFAULT_WAVELENGTH_PERCENT
) -> np.ndarray:
    """
    Detect outliers using Z-score method.
    
    Args:
        spectral_data: Array of spectral data (n_samples, n_wavelengths)
        threshold: Z-score threshold for outlier detection
        wavelength_percent: Percent of wavelengths that must be outliers
            
    Returns:
        Boolean mask indicating outlier samples
    """
    # Calculate Z-scores for each wavelength
    z_scores = np.abs(stats.zscore(spectral_data, axis=0))
    
    # Count wavelengths where each sample exceeds threshold
    outlier_counts = np.sum(z_scores > threshold, axis=1)
    
    # A sample is an outlier if it's an outlier in at least wavelength_percent of wavelengths
    min_outlier_wavelengths = int(spectral_data.shape[1] * wavelength_percent)
    outlier_mask = outlier_counts >= min_outlier_wavelengths
    
    return outlier_mask

def detect_outliers_pca(
    spectral_data: np.ndarray,
    threshold: float = DEFAULT_OUTLIER_THRESHOLD,
    n_components: int = DEFAULT_PCA_COMPONENTS
) -> np.ndarray:
    """
    Detect outliers using PCA decomposition and Hotelling's T2 statistic.
    
    Args:
        spectral_data: Array of spectral data (n_samples, n_wavelengths)
        threshold: Threshold for outlier detection (standard deviations)
        n_components: Number of PCA components to use
            
    Returns:
        Boolean mask indicating outlier samples
    """
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(spectral_data)
    
    # Apply PCA
    pca = PCA(n_components=min(n_components, X_scaled.shape[1], X_scaled.shape[0]))
    X_pca = pca.fit_transform(X_scaled)
    
    # Calculate Hotelling's T2 statistic
    t2 = np.sum((X_pca ** 2) / pca.explained_variance_, axis=1)
    
    # Identify outliers
    t2_mean = np.mean(t2)
    t2_std = np.std(t2)
    outlier_mask = t2 > (t2_mean + threshold * t2_std)
    
    return outlier_mask

def remove_constant_and_empty_columns(
    df: pd.DataFrame, 
    remove_empty: bool = True, 
    remove_constant: bool = True,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Remove constant and empty columns from the dataset.
    
    Args:
        df: DataFrame with NIR spectroscopy data
        remove_empty: Whether to remove columns with all missing values
        remove_constant: Whether to remove columns with a single unique value
        verbose: Whether to log processing information
            
    Returns:
        DataFrame with constant and empty columns removed
    """
    # Configure logging
    logger = logging.getLogger(__name__)
    
    df_cleaned = df.copy()
    original_columns = df_cleaned.columns.tolist()
    removed_columns = []
    
    # Remove empty columns
    if remove_empty:
        empty_cols = [col for col in df_cleaned.columns if df_cleaned[col].isna().all()]
        if empty_cols:
            df_cleaned = df_cleaned.drop(columns=empty_cols)
            removed_columns.extend(empty_cols)
            if verbose:
                logger.info(f"Removed {len(empty_cols)} empty columns")
    
    # Remove constant columns
    if remove_constant:
        constant_cols = [col for col in df_cleaned.columns 
                        if df_cleaned[col].nunique(dropna=False) <= 1]
        if constant_cols:
            df_cleaned = df_cleaned.drop(columns=constant_cols)
            removed_columns.extend(constant_cols)
            if verbose:
                logger.info(f"Removed {len(constant_cols)} constant columns")
    
    if verbose:
        logger.info(f"Total columns removed: {len(removed_columns)}")
        logger.info(f"Remaining columns: {len(df_cleaned.columns)}")
    
    return df_cleaned

def split_data(
    X: pd.DataFrame, 
    y: pd.Series,
    test_size: float = 0.2, 
    val_size: Optional[float] = 0.2,
    random_state: int = 42,
    stratify: Optional[pd.Series] = None
) -> Dict[str, Any]:
    """
    Split data into training, validation, and test sets.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        test_size: Proportion of data for test set
        val_size: Proportion of data for validation set (from remaining after test)
        random_state: Random seed for reproducibility
        stratify: Optional Series to use for stratified splitting
            
    Returns:
        Dictionary with data splits and indices
    """
    # First split: separate test set
    if stratify is not None:
        X_temp, X_test, y_temp, y_test, strat_temp, strat_test = train_test_split(
            X, y, stratify, test_size=test_size, random_state=random_state
        )
    else:
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        strat_temp = strat_test = None
    
    # Second split: separate validation set if requested
    if val_size:
        # Calculate effective validation size from remaining data
        effective_val_size = val_size / (1 - test_size)
        
        if stratify is not None and strat_temp is not None:
            X_train, X_val, y_train, y_val, _, _ = train_test_split(
                X_temp, y_temp, strat_temp, 
                test_size=effective_val_size, 
                random_state=random_state
            )
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, 
                test_size=effective_val_size, 
                random_state=random_state
            )
    else:
        # No validation set
        X_train, y_train = X_temp, y_temp
        X_val, y_val = pd.DataFrame(), pd.Series(dtype=float)
    
    # Get indices from original dataframe
    train_indices = X_train.index
    val_indices = X_val.index if val_size else pd.Index([])
    test_indices = X_test.index
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'train_indices': train_indices,
        'val_indices': val_indices,
        'test_indices': test_indices
    }

def filter_numeric_features(
    df: pd.DataFrame,
    exclude_columns: Optional[List[str]] = None,
    keep_columns: Optional[List[str]] = None,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Filter dataframe to keep only numeric features suitable for machine learning.
    Optionally exclude specific columns or explicitly keep certain columns.
    
    Args:
        df: DataFrame with features
        exclude_columns: List of column names to explicitly exclude
        keep_columns: List of column names to explicitly keep (overrides numeric filtering)
        verbose: Whether to log processing information
            
    Returns:
        DataFrame with only numeric columns or explicitly kept columns
    """
    # Configure logging
    logger = logging.getLogger(__name__)
    
    exclude_columns = exclude_columns or []
    keep_columns = keep_columns or []
    
    # Start with a copy
    df_filtered = df.copy()
    
    # First exclude specific columns
    if exclude_columns:
        columns_to_drop = [col for col in exclude_columns if col in df.columns]
        if columns_to_drop:
            df_filtered = df_filtered.drop(columns=columns_to_drop)
            if verbose:
                logger.info(f"Excluded {len(columns_to_drop)} specified columns: {columns_to_drop}")
    
    # If keep_columns is specified, keep only those columns
    if keep_columns:
        keep_columns = [col for col in keep_columns if col in df_filtered.columns]
        if keep_columns:
            df_filtered = df_filtered[keep_columns]
            if verbose:
                logger.info(f"Keeping only {len(keep_columns)} specified columns")
            return df_filtered
    
    # Otherwise, filter to keep only numeric columns
    initial_columns = df_filtered.columns.tolist()
    df_numeric = df_filtered.select_dtypes(include=['number'])
    
    # Log results if verbose
    if verbose:
        removed_columns = set(initial_columns) - set(df_numeric.columns)
        if removed_columns:
            logger.info(f"Removed {len(removed_columns)} non-numeric columns: {list(removed_columns)}")
        logger.info(f"Kept {len(df_numeric.columns)} numeric columns")
    
    return df_numeric 