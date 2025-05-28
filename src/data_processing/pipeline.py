"""
Functions for creating and managing NIR tomato spectroscopy data processing pipelines.

This module provides tools to build preprocessing pipelines and prepare data for modeling:

Functions:
- create_preprocessing_pipeline: Creates a preprocessing pipeline for NIR data
- process_and_save_data: Processes data using a pipeline and saves results
- prepare_data_for_regression: Prepares data for regression tasks with Brix prediction
- load_and_preprocess_data: Loads data and applies basic preprocessing
"""

import os
import pickle
import joblib
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional, Union
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin

from .constants import (
    TARGET_COLUMN,
    DEFAULT_DATASET_PATH,
    AVAILABLE_PREPROCESSING_METHODS
)
from .transformers import (
    SNVTransformer,
    SavGolTransformer,
    MSCTransformer,
    OutlierDetector,
    PCATransformer,
    create_preprocessing_pipelines
)
from .utils import (
    identify_spectral_columns,
    fix_column_names,
    remove_duplicate_rows,
    add_sample_identifier,
    remove_constant_and_empty_columns,
    split_data,
    detect_outliers_zscore,
    detect_outliers_pca,
    filter_numeric_features
)

def create_preprocessing_pipeline(
    preprocessing_method: str = 'raw',
    outlier_detection: bool = True,
    outlier_method: str = 'zscore',
    outlier_threshold: float = 3.0,
    remove_outliers: bool = False,
    custom_steps: Optional[List[Tuple[str, Any]]] = None,
) -> Pipeline:
    """
    Create a preprocessing pipeline for NIR tomato data.
    
    Args:
        preprocessing_method: Method for preprocessing ('raw', 'snv', 'sg1', etc.)
        outlier_detection: Whether to include outlier detection
        outlier_method: Method for outlier detection ('zscore', 'pca', 'both')
        outlier_threshold: Threshold for outlier detection
        remove_outliers: Whether to remove detected outliers
        custom_steps: Additional custom steps to add to the pipeline
            
    Returns:
        Scikit-learn Pipeline for preprocessing
    """
    # Get available pipelines
    preprocessing_pipelines = create_preprocessing_pipelines()
    
    # Validate preprocessing method
    if preprocessing_method not in preprocessing_pipelines:
        valid_methods = list(preprocessing_pipelines.keys())
        raise ValueError(f"Invalid preprocessing method: {preprocessing_method}. Valid options: {valid_methods}")
    
    # Get base pipeline for the selected method
    base_pipeline = preprocessing_pipelines[preprocessing_method]
    
    # Create steps list from base pipeline
    steps = list(base_pipeline.steps)
    
    # Add outlier detection step if requested
    if outlier_detection:
        outlier_step = ('outlier_detector', OutlierDetector(
            method=outlier_method,
            threshold=outlier_threshold,
            remove_outliers=remove_outliers
        ))
        # Add as first step
        steps.insert(0, outlier_step)
    
    # Add any custom steps
    if custom_steps:
        steps.extend(custom_steps)
    
    # Create the final pipeline
    pipeline = Pipeline(steps)
    
    return pipeline

def process_and_save_data(
    data_path: str = DEFAULT_DATASET_PATH,
    output_dir: str = "data/processed",
    output_filename: str = "processed_tomato_nir_data",
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
    preprocessing_method: str = 'raw',
    outlier_detection: bool = True,
    remove_constant_columns: bool = True,
    impute_missing_values: bool = True,
    save_format: str = 'joblib',
    verbose: bool = False,
) -> Dict[str, str]:
    """
    Process NIR tomato data and save the results.
    
    Args:
        data_path: Path to the input data file
        output_dir: Directory to save processed data
        output_filename: Base filename for output files
        test_size: Proportion of data for test set
        val_size: Proportion of data for validation set
        random_state: Random seed for reproducibility
        preprocessing_method: Method for preprocessing
        outlier_detection: Whether to perform outlier detection
        remove_constant_columns: Whether to remove constant columns
        impute_missing_values: Whether to impute missing values
        save_format: Format to save data ('joblib', 'pickle', 'csv')
        verbose: Whether to log processing information
            
    Returns:
        Dictionary with paths to saved files
    """
    # Configure logging
    log_level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=log_level, format='%(message)s')
    logger = logging.getLogger(__name__)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and preprocess data
    logger.info(f"Loading data from {data_path}...")
    processed_data = load_and_preprocess_data(
        data_path=data_path,
        preprocessing_method=preprocessing_method,
        outlier_detection=outlier_detection,
        remove_constant_columns=remove_constant_columns,
        impute_missing_values=impute_missing_values,
        verbose=verbose
    )
    
    # Extract components
    X = processed_data['X']
    y = processed_data['y']
    preprocessing_info = processed_data['preprocessing_info']
    
    # Split data into train, validation, and test sets
    logger.info(f"Splitting data into train, validation, and test sets...")
    splits = split_data(
        X=X,
        y=y,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state
    )
    
    # Create dataset with all information
    dataset = {
        'X': X,
        'y': y,
        'data_splits': {
            'train_indices': splits['train_indices'],
            'val_indices': splits['val_indices'],
            'test_indices': splits['test_indices']
        },
        'preprocessing_info': preprocessing_info
    }
    
    # Save data based on specified format
    saved_files = {}
    
    if save_format == 'joblib':
        # Save as joblib file
        joblib_path = os.path.join(output_dir, f"{output_filename}.joblib")
        joblib.dump(dataset, joblib_path)
        saved_files['joblib'] = joblib_path
        logger.info(f"Saved dataset to {joblib_path}")
    
    elif save_format == 'pickle':
        # Save as pickle file
        pickle_path = os.path.join(output_dir, f"{output_filename}.pkl")
        with open(pickle_path, 'wb') as f:
            pickle.dump(dataset, f)
        saved_files['pickle'] = pickle_path
        logger.info(f"Saved dataset to {pickle_path}")
    
    elif save_format == 'csv':
        # Save as CSV files
        X_path = os.path.join(output_dir, f"{output_filename}_features.csv")
        y_path = os.path.join(output_dir, f"{output_filename}_target.csv")
        splits_path = os.path.join(output_dir, f"{output_filename}_splits.csv")
        
        # Save features
        X.to_csv(X_path, index=True)
        saved_files['X_csv'] = X_path
        
        # Save target
        y.to_frame().to_csv(y_path, index=True)
        saved_files['y_csv'] = y_path
        
        # Save splits
        splits_df = pd.DataFrame({
            'index': list(splits['train_indices']) + list(splits['val_indices']) + list(splits['test_indices']),
            'split': ['train'] * len(splits['train_indices']) + 
                    ['val'] * len(splits['val_indices']) + 
                    ['test'] * len(splits['test_indices'])
        })
        splits_df.to_csv(splits_path, index=False)
        saved_files['splits_csv'] = splits_path
        
        # Save preprocessing info
        info_path = os.path.join(output_dir, f"{output_filename}_preprocessing_info.json")
        # Convert numpy arrays and other non-serializable objects to lists
        info_serializable = {k: str(v) if isinstance(v, (np.ndarray, set)) else v 
                            for k, v in preprocessing_info.items()}
        pd.Series(info_serializable).to_json(info_path)
        saved_files['info_json'] = info_path
        
        logger.info(f"Saved dataset as CSV files in {output_dir}")
    
    else:
        raise ValueError(f"Invalid save format: {save_format}. Use 'joblib', 'pickle', or 'csv'.")
    
    return saved_files

def load_and_preprocess_data(
    data_path: str = DEFAULT_DATASET_PATH,
    preprocessing_method: str = 'raw',
    outlier_detection: bool = True,
    outlier_method: str = 'zscore',
    outlier_threshold: float = 3.0,
    remove_outliers: bool = False,
    remove_constant_columns: bool = True,
    impute_missing_values: bool = True,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Load and preprocess NIR tomato data.
    
    Args:
        data_path: Path to the input data file
        preprocessing_method: Method for preprocessing
        outlier_detection: Whether to perform outlier detection
        outlier_method: Method for outlier detection
        outlier_threshold: Threshold for outlier detection
        remove_outliers: Whether to remove detected outliers
        remove_constant_columns: Whether to remove constant columns
        impute_missing_values: Whether to impute missing values
        verbose: Whether to log processing information
            
    Returns:
        Dictionary with processed data and preprocessing information
    """
    # Configure logging
    log_level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=log_level, format='%(message)s')
    logger = logging.getLogger(__name__)
    
    # Load data
    logger.info(f"Loading data from {data_path}...")
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    elif data_path.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path}. Use CSV or Excel files.")
    
    logger.info(f"Loaded data shape: {df.shape}")
    
    # Basic preprocessing
    logger.info("Applying basic preprocessing...")
    
    # Fix column names if needed
    df = fix_column_names(df, verbose=verbose)
    
    # Remove duplicate rows
    df = remove_duplicate_rows(df, verbose=verbose)
    
    # Add sample identifier
    df = add_sample_identifier(df)
    
    # Remove constant and empty columns if requested
    if remove_constant_columns:
        df = remove_constant_and_empty_columns(df, verbose=verbose)
    
    # Check if target column exists
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in the dataset.")
    
    # Identify spectral columns
    spectral_cols, non_spectral_cols = identify_spectral_columns(df)
    logger.info(f"Identified {len(spectral_cols)} spectral columns and {len(non_spectral_cols)} non-spectral columns")
    
    # Separate target from features
    y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])
    
    # Impute missing values if requested
    if impute_missing_values and X.isna().any().any():
        logger.info("Imputing missing values...")
        # Use different imputation for numerical and categorical columns
        numeric_cols = X.select_dtypes(include=['number']).columns
        categorical_cols = X.select_dtypes(exclude=['number']).columns
        
        if not numeric_cols.empty:
            numeric_imputer = SimpleImputer(strategy='mean')
            X[numeric_cols] = numeric_imputer.fit_transform(X[numeric_cols])
        
        if not categorical_cols.empty:
            categorical_imputer = SimpleImputer(strategy='most_frequent')
            X[categorical_cols] = categorical_imputer.fit_transform(X[categorical_cols])
    
    # Create and apply preprocessing pipeline
    logger.info(f"Applying preprocessing method: {preprocessing_method}")
    pipeline = create_preprocessing_pipeline(
        preprocessing_method=preprocessing_method,
        outlier_detection=outlier_detection,
        outlier_method=outlier_method,
        outlier_threshold=outlier_threshold,
        remove_outliers=remove_outliers
    )
    
    # Transform the data using the pipeline
    # Only apply to spectral columns
    if spectral_cols:
        X_spectral = X[spectral_cols]
        X_spectral_processed = pipeline.fit_transform(X_spectral)
        
        # If output is a DataFrame, combine with non-spectral columns
        if isinstance(X_spectral_processed, pd.DataFrame):
            non_target_cols = [col for col in non_spectral_cols if col != TARGET_COLUMN]
            if non_target_cols:
                X_processed = pd.concat([X_spectral_processed, X[non_target_cols]], axis=1)
            else:
                X_processed = X_spectral_processed
        else:
            # Convert numpy array to DataFrame
            X_processed = pd.DataFrame(
                X_spectral_processed, 
                index=X.index,
                columns=[f"feature_{i}" for i in range(X_spectral_processed.shape[1])]
            )
            
            # Add non-spectral columns
            non_target_cols = [col for col in non_spectral_cols if col != TARGET_COLUMN]
            if non_target_cols:
                X_processed = pd.concat([X_processed, X[non_target_cols]], axis=1)
    else:
        # No spectral columns identified, just pass through the pipeline
        X_processed = pipeline.fit_transform(X)
    
    # Prepare preprocessing information
    preprocessing_info = {
        'original_shape': df.shape,
        'preprocessed_shape': X_processed.shape,
        'preprocessing_method': preprocessing_method,
        'outlier_detection': outlier_detection,
        'outlier_method': outlier_method if outlier_detection else None,
        'spectral_columns': spectral_cols,
        'non_spectral_columns': non_spectral_cols,
        'target_column': TARGET_COLUMN,
        'target_stats': {
            'min': y.min(),
            'max': y.max(),
            'mean': y.mean(),
            'std': y.std(),
            'median': y.median()
        }
    }
    
    return {
        'X': X_processed,
        'y': y,
        'preprocessing_info': preprocessing_info
    }

def prepare_data_for_regression(
    data_path: str = DEFAULT_DATASET_PATH,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
    preprocessing_method: str = 'raw',
    outlier_detection: bool = True,
    remove_constant_columns: bool = True,
    impute_missing_values: bool = True,
    additional_features: Optional[List[str]] = None,
    exclude_features: Optional[List[str]] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Prepare data for regression tasks (Brix prediction).
    
    Args:
        data_path: Path to the input data file
        test_size: Proportion of data for test set
        val_size: Proportion of data for validation set
        random_state: Random seed for reproducibility
        preprocessing_method: Method for preprocessing
        outlier_detection: Whether to perform outlier detection
        remove_constant_columns: Whether to remove constant columns
        impute_missing_values: Whether to impute missing values
        additional_features: Additional non-spectral features to include
        exclude_features: Features to exclude
        verbose: Whether to log processing information
            
    Returns:
        Dictionary with processed data splits and information
    """
    # Configure logging
    log_level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=log_level, format='%(message)s')
    logger = logging.getLogger(__name__)
    
    # Load and preprocess data
    processed_data = load_and_preprocess_data(
        data_path=data_path,
        preprocessing_method=preprocessing_method,
        outlier_detection=outlier_detection,
        remove_constant_columns=remove_constant_columns,
        impute_missing_values=impute_missing_values,
        verbose=verbose
    )
    
    # Extract components
    X = processed_data['X']
    y = processed_data['y']
    preprocessing_info = processed_data['preprocessing_info']
    
    # Filter features if specified
    if exclude_features:
        X = X.drop(columns=[col for col in exclude_features if col in X.columns])
        logger.info(f"Excluded {len(exclude_features)} features")
    
    if additional_features:
        # Keep only specified additional features from non-spectral columns
        spectral_cols = preprocessing_info['spectral_columns']
        keep_cols = spectral_cols + [col for col in additional_features if col in X.columns]
        X = X[keep_cols]
        logger.info(f"Keeping {len(spectral_cols)} spectral columns and {len(additional_features)} additional features")
    
    # Split data into train, validation, and test sets
    logger.info(f"Splitting data into train, validation, and test sets...")
    splits = split_data(
        X=X,
        y=y,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state
    )
    
    return {
        'X_train': splits['X_train'],
        'y_train': splits['y_train'],
        'X_val': splits['X_val'],
        'y_val': splits['y_val'],
        'X_test': splits['X_test'],
        'y_test': splits['y_test'],
        'preprocessing_info': preprocessing_info,
        'feature_names': list(X.columns)
    }

def preprocess_spectra(
    df: pd.DataFrame,
    target_column: str,
    transformers: List[BaseEstimator] = None,
    exclude_columns: List[str] = None,
    remove_outliers: bool = True,
    outlier_method: str = 'zscore',
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Complete preprocessing pipeline for NIR tomato spectroscopy data.
    
    Args:
        df: DataFrame with spectral data
        target_column: Column name of the target variable
        transformers: List of transformer objects to apply to spectral data
        exclude_columns: List of columns to exclude from processing
        remove_outliers: Whether to detect and remove outliers
        outlier_method: Method for outlier detection ('zscore' or 'pca')
        verbose: Whether to log processing information
        
    Returns:
        Dictionary containing:
            - X: Processed feature DataFrame
            - y: Target Series
            - spectral_columns: List of spectral column names
            - non_spectral_columns: List of non-spectral column names
            - preprocessing_pipeline: Fitted preprocessing pipeline
            - outlier_mask: Boolean mask of outliers (if remove_outliers=True)
    """
    # Configure logging
    logger = logging.getLogger(__name__)
    
    # Set default values
    exclude_columns = exclude_columns or []
    transformers = transformers or [SNVTransformer()]
    
    # Initialize return dictionary
    result = {}
    
    # Step 1: Fix column names if needed
    df_fixed = fix_column_names(df, verbose=verbose)
    
    # Step 2: Identify spectral and non-spectral columns
    spectral_columns, non_spectral_columns = identify_spectral_columns(df_fixed)
    result['spectral_columns'] = spectral_columns
    result['non_spectral_columns'] = non_spectral_columns
    
    if verbose:
        logger.info(f"Identified {len(spectral_columns)} spectral columns")
        logger.info(f"Identified {len(non_spectral_columns)} non-spectral columns")
    
    # Step 3: Remove duplicates
    df_no_duplicates = remove_duplicate_rows(df_fixed, verbose=verbose)
    
    # Step 4: Remove constant and empty columns
    df_cleaned = remove_constant_and_empty_columns(df_no_duplicates, verbose=verbose)
    
    # Step 5: Filter out non-numeric features that can't be used for modeling
    # Define columns to exclude (target and explicitly excluded columns)
    columns_to_exclude = exclude_columns.copy()
    # Don't exclude the target column yet
    if target_column in columns_to_exclude:
        columns_to_exclude.remove(target_column)
    
    df_numeric = filter_numeric_features(
        df_cleaned, 
        exclude_columns=columns_to_exclude,
        verbose=verbose
    )
    
    # Step 6: Create and apply preprocessing pipeline for spectral data
    spectral_columns = [col for col in spectral_columns if col in df_numeric.columns]
    
    # Create pipeline
    steps = []
    for i, transformer in enumerate(transformers):
        steps.append((f'transformer_{i}', transformer))
    
    preprocessing_pipeline = Pipeline(steps)
    
    # Extract spectral data
    X_spectral = df_numeric[spectral_columns].copy()
    
    # Apply preprocessing
    if verbose:
        logger.info(f"Applying preprocessing pipeline with {len(steps)} transformers")
    
    X_spectral_transformed = preprocessing_pipeline.fit_transform(X_spectral)
    
    # Convert back to DataFrame with column names
    X_spectral_df = pd.DataFrame(
        X_spectral_transformed, 
        columns=spectral_columns,
        index=X_spectral.index
    )
    
    # Step 7: Combine with non-spectral numeric features
    non_spectral_numeric_cols = [
        col for col in df_numeric.columns 
        if col not in spectral_columns and col != target_column
    ]
    
    if non_spectral_numeric_cols:
        if verbose:
            logger.info(f"Adding {len(non_spectral_numeric_cols)} non-spectral numeric features")
        X = pd.concat([X_spectral_df, df_numeric[non_spectral_numeric_cols]], axis=1)
    else:
        X = X_spectral_df
    
    # Step 8: Handle outliers if requested
    if remove_outliers:
        if outlier_method == 'zscore':
            outlier_mask = detect_outliers_zscore(X_spectral.values)
        elif outlier_method == 'pca':
            outlier_mask = detect_outliers_pca(X_spectral.values)
        else:
            raise ValueError(f"Unknown outlier method: {outlier_method}")
        
        outlier_count = np.sum(outlier_mask)
        if verbose:
            logger.info(f"Detected {outlier_count} outliers using {outlier_method} method")
        
        # Remove outliers
        if outlier_count > 0:
            X = X.loc[~outlier_mask]
        
        result['outlier_mask'] = outlier_mask
    
    # Step 9: Prepare target variable if available
    if target_column in df_numeric.columns:
        y = df_numeric[target_column].loc[X.index]
        result['y'] = y
    else:
        if verbose:
            logger.warning(f"Target column '{target_column}' not found in DataFrame")
        result['y'] = None
    
    # Add results to dictionary
    result['X'] = X
    result['preprocessing_pipeline'] = preprocessing_pipeline
    
    return result 