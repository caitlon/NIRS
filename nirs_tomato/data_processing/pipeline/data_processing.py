"""
Data Processing Module for NIR Tomato Spectroscopy

This module provides functions for processing NIR spectroscopy data, including
loading, preprocessing, and saving processed data.
"""

import logging
import os
import pickle
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from ..constants import DEFAULT_DATASET_PATH, TARGET_COLUMN
from ..transformers import MSCTransformer, SavGolTransformer, SNVTransformer
from ..utils import (
    add_sample_identifier,
    detect_outliers_pca,
    detect_outliers_zscore,
    filter_numeric_features,
    fix_column_names,
    identify_spectral_columns,
    remove_constant_and_empty_columns,
    remove_duplicate_rows,
    split_data,
)
from .preprocessing import create_preprocessing_pipeline


def get_wavelengths_from_columns(column_names: List[str]) -> List[int]:
    """
    Extract wavelength values from column names.

    Args:
        column_names: List of column names like 'wl_900', 'wl_901', etc.

    Returns:
        List of extracted wavelength values as integers
    """
    wavelengths = []
    for col in column_names:
        # Try to extract the wavelength number from the column name
        if col.startswith("wl_"):
            try:
                wl = int(col.split("_")[1])
                wavelengths.append(wl)
            except (ValueError, IndexError):
                continue
    return wavelengths


def apply_spectral_transformation(
    X: np.ndarray, transform_method: str = "none", **kwargs: Any
) -> np.ndarray:
    """
    Apply spectral transformation to data.

    Args:
        X: Spectral data array
        transform_method: Transformation method ('snv', 'msc', 'savgol', 'none')
        **kwargs: Additional arguments for transformers

    Returns:
        Transformed spectral data
    """
    if transform_method == "none":
        return X

    if transform_method == "snv":
        transformer = SNVTransformer()
    elif transform_method == "msc":
        transformer = MSCTransformer()
    elif transform_method == "savgol":
        window_length = kwargs.get("window_length", 15)
        polyorder = kwargs.get("polyorder", 2)
        deriv = kwargs.get("deriv", 0)
        transformer = SavGolTransformer(
            window_length=window_length, polyorder=polyorder, deriv=deriv
        )
    else:
        raise ValueError(f"Unknown transformation method: {transform_method}")

    return transformer.fit_transform(X)


def process_spectral_data(
    data: pd.DataFrame,
    target_column: str,
    exclude_columns: List[str] = None,
    transform_method: str = "none",
    apply_savgol: bool = False,
    window_length: int = 15,
    polyorder: int = 2,
    deriv: int = 0,
    feature_selection_method: str = "none",
    n_features: int = 20,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Process spectral data with transformations and optional feature selection.

    Args:
        data: DataFrame with spectral and non-spectral data
        target_column: Name of target column
        exclude_columns: Columns to exclude from processing
        transform_method: Transformation method ('snv', 'msc', 'none')
        apply_savgol: Whether to apply Savitzky-Golay filtering
        window_length: Window length for SG filter
        polyorder: Polynomial order for SG filter
        deriv: Derivative order for SG filter
        feature_selection_method: Method for feature selection ('vip', 'ga', 'cars', 'none')
        n_features: Number of features to select
        verbose: Whether to log processing details

    Returns:
        Dictionary with processed data and metadata
    """
    # Configure logging
    logger = logging.getLogger(__name__)
    exclude_columns = exclude_columns or []

    # Step 1: Identify spectral columns
    spectral_cols = identify_spectral_columns(data)

    # Step 2: Extract wavelengths from column names
    wavelengths = get_wavelengths_from_columns(spectral_cols)

    # Step 3: Extract features and target
    X = data[spectral_cols].copy()
    y = data[target_column].copy() if target_column in data.columns else None

    # Step 4: Apply spectral transformations
    X_values = X.values
    if transform_method != "none":
        if verbose:
            logger.info(f"Applying {transform_method.upper()} transformation")
        X_transformed = apply_spectral_transformation(
            X_values, transform_method=transform_method
        )
        X = pd.DataFrame(X_transformed, index=X.index, columns=X.columns)

    # Step 5: Apply Savitzky-Golay if requested
    if apply_savgol:
        if verbose:
            logger.info(
                f"Applying Savitzky-Golay filter (window={window_length}, "
                f"polyorder={polyorder}, deriv={deriv})"
            )
        X_sg = apply_spectral_transformation(
            X.values,
            transform_method="savgol",
            window_length=window_length,
            polyorder=polyorder,
            deriv=deriv,
        )
        X = pd.DataFrame(X_sg, index=X.index, columns=X.columns)

    # Step 6: Apply feature selection if requested
    selected_features = []
    if feature_selection_method != "none" and y is not None:
        if verbose:
            logger.info(
                f"Applying {feature_selection_method.upper()} feature selection"
            )

        from ..feature_selection import (
            CARSSelector,
            GeneticAlgorithmSelector,
            PLSVIPSelector,
        )

        if feature_selection_method == "vip":
            selector = PLSVIPSelector(n_features_to_select=n_features)
        elif feature_selection_method == "ga":
            selector = GeneticAlgorithmSelector(
                n_features_to_select=n_features
            )
        elif feature_selection_method == "cars":
            selector = CARSSelector(n_features_to_select=n_features)
        else:
            raise ValueError(
                f"Unknown feature selection method: {feature_selection_method}"
            )

        # Fit the selector
        selector.fit(X, y)

        # Get selected features
        if hasattr(selector, "selected_features_indices_"):
            selected_indices = selector.selected_features_indices_
            selected_features = [X.columns[i] for i in selected_indices]
            X = X[selected_features]

            if verbose:
                logger.info(f"Selected {len(selected_features)} features")

    # Prepare result
    result = {
        "data": data,
        "X": X,
        "y": y,
        "wavelengths": wavelengths,
        "spectral_columns": spectral_cols,
        "selected_features": selected_features,
    }

    return result


def process_and_save_data(
    data_path: str = DEFAULT_DATASET_PATH,
    output_dir: str = "data/processed",
    output_filename: str = "processed_tomato_nir_data",
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
    preprocessing_method: str = "raw",
    outlier_detection: bool = True,
    remove_constant_columns: bool = True,
    impute_missing_values: bool = True,
    save_format: str = "joblib",
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
    logging.basicConfig(level=log_level, format="%(message)s")
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
        verbose=verbose,
    )

    # Extract components
    X = processed_data["X"]
    y = processed_data["y"]
    preprocessing_info = processed_data["preprocessing_info"]

    # Split data into train, validation, and test sets
    logger.info("Splitting data into train, validation, and test sets...")
    splits = split_data(
        X=X,
        y=y,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
    )

    # Create dataset with all information
    dataset = {
        "X": X,
        "y": y,
        "data_splits": {
            "train_indices": splits["train_indices"],
            "val_indices": splits["val_indices"],
            "test_indices": splits["test_indices"],
        },
        "preprocessing_info": preprocessing_info,
    }

    # Save data based on specified format
    saved_files = {}

    if save_format == "joblib":
        # Save as joblib file
        joblib_path = os.path.join(output_dir, f"{output_filename}.joblib")
        joblib.dump(dataset, joblib_path)
        saved_files["joblib"] = joblib_path
        logger.info(f"Saved dataset to {joblib_path}")

    elif save_format == "pickle":
        # Save as pickle file
        pickle_path = os.path.join(output_dir, f"{output_filename}.pkl")
        with open(pickle_path, "wb") as f:
            pickle.dump(dataset, f)
        saved_files["pickle"] = pickle_path
        logger.info(f"Saved dataset to {pickle_path}")

    elif save_format == "csv":
        # Save as CSV files
        X_path = os.path.join(output_dir, f"{output_filename}_features.csv")
        y_path = os.path.join(output_dir, f"{output_filename}_target.csv")
        splits_path = os.path.join(output_dir, f"{output_filename}_splits.csv")

        # Save features
        X.to_csv(X_path, index=True)
        saved_files["X_csv"] = X_path

        # Save target
        y.to_frame().to_csv(y_path, index=True)
        saved_files["y_csv"] = y_path

        # Save splits
        splits_df = pd.DataFrame(
            {
                "index": list(splits["train_indices"])
                + list(splits["val_indices"])
                + list(splits["test_indices"]),
                "split": ["train"] * len(splits["train_indices"])
                + ["val"] * len(splits["val_indices"])
                + ["test"] * len(splits["test_indices"]),
            }
        )
        splits_df.to_csv(splits_path, index=False)
        saved_files["splits_csv"] = splits_path

        # Save preprocessing info
        info_path = os.path.join(
            output_dir, f"{output_filename}_preprocessing_info.json"
        )
        # Convert numpy arrays and other non-serializable objects to lists
        info_serializable = {
            k: str(v) if isinstance(v, (np.ndarray, set)) else v
            for k, v in preprocessing_info.items()
        }
        pd.Series(info_serializable).to_json(info_path)
        saved_files["info_json"] = info_path

        logger.info("Saved dataset as CSV files in " + output_dir)

    else:
        raise ValueError(
            f"Invalid save format: {
                save_format
            }. Use 'joblib', 'pickle', or 'csv'."
        )

    return saved_files


def load_and_preprocess_data(
    data_path: str = DEFAULT_DATASET_PATH,
    preprocessing_method: str = "raw",
    outlier_detection: bool = True,
    outlier_method: str = "zscore",
    outlier_threshold: float = 3.0,
    remove_outliers: bool = False,
    remove_constant_columns: bool = True,
    impute_missing_values: bool = True,
    verbose: bool = False,
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
    logging.basicConfig(level=log_level, format="%(message)s")
    logger = logging.getLogger(__name__)

    # Load data
    logger.info(f"Loading data from {data_path}...")
    if data_path.endswith(".csv"):
        df = pd.read_csv(data_path)
    elif data_path.endswith((".xlsx", ".xls")):
        df = pd.read_excel(data_path)
    else:
        raise ValueError(
            f"Unsupported file format: {data_path}. Use CSV or Excel files."
        )

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
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found in the dataset."
        )

    # Identify spectral columns
    spectral_cols, non_spectral_cols = identify_spectral_columns(df)
    logger.info(
        f"Identified {len(spectral_cols)} spectral columns and {
            len(non_spectral_cols)
        } non-spectral columns"
    )

    # Separate target from features
    y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])

    # Impute missing values if requested
    if impute_missing_values and X.isna().any().any():
        logger.info("Imputing missing values...")
        # Use different imputation for numerical and categorical columns
        numeric_cols = X.select_dtypes(include=["number"]).columns
        categorical_cols = X.select_dtypes(exclude=["number"]).columns

        if not numeric_cols.empty:
            numeric_imputer = SimpleImputer(strategy="mean")
            X[numeric_cols] = numeric_imputer.fit_transform(X[numeric_cols])

        if not categorical_cols.empty:
            categorical_imputer = SimpleImputer(strategy="most_frequent")
            X[categorical_cols] = categorical_imputer.fit_transform(
                X[categorical_cols]
            )

    # Create and apply preprocessing pipeline
    logger.info(f"Applying preprocessing method: {preprocessing_method}")
    pipeline = create_preprocessing_pipeline(
        preprocessing_method=preprocessing_method,
        outlier_detection=outlier_detection,
        outlier_method=outlier_method,
        outlier_threshold=outlier_threshold,
        remove_outliers=remove_outliers,
    )

    # Transform the data using the pipeline
    # Only apply to spectral columns
    if spectral_cols:
        X_spectral = X[spectral_cols]
        X_spectral_processed = pipeline.fit_transform(X_spectral)

        # If output is a DataFrame, combine with non-spectral columns
        if isinstance(X_spectral_processed, pd.DataFrame):
            non_target_cols = [
                col for col in non_spectral_cols if col != TARGET_COLUMN
            ]
            if non_target_cols:
                X_processed = pd.concat(
                    [X_spectral_processed, X[non_target_cols]], axis=1
                )
            else:
                X_processed = X_spectral_processed
        else:
            # Convert numpy array to DataFrame
            X_processed = pd.DataFrame(
                X_spectral_processed,
                index=X.index,
                columns=[
                    f"feature_{i}"
                    for i in range(X_spectral_processed.shape[1])
                ],
            )

            # Add non-spectral columns
            non_target_cols = [
                col for col in non_spectral_cols if col != TARGET_COLUMN
            ]
            if non_target_cols:
                X_processed = pd.concat(
                    [X_processed, X[non_target_cols]], axis=1
                )
    else:
        # No spectral columns identified, just pass through the pipeline
        X_processed = pipeline.fit_transform(X)

    # Prepare preprocessing information
    preprocessing_info = {
        "original_shape": df.shape,
        "preprocessed_shape": X_processed.shape,
        "preprocessing_method": preprocessing_method,
        "outlier_detection": outlier_detection,
        "outlier_method": outlier_method if outlier_detection else None,
        "spectral_columns": spectral_cols,
        "non_spectral_columns": non_spectral_cols,
        "target_column": TARGET_COLUMN,
        "target_stats": {
            "min": y.min(),
            "max": y.max(),
            "mean": y.mean(),
            "std": y.std(),
            "median": y.median(),
        },
    }

    return {"X": X_processed, "y": y, "preprocessing_info": preprocessing_info}


def preprocess_spectra(
    df: pd.DataFrame,
    target_column: str,
    transformers: List[BaseEstimator] = None,
    exclude_columns: List[str] = None,
    remove_outliers: bool = True,
    outlier_method: str = "zscore",
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
    spectral_columns, non_spectral_columns = identify_spectral_columns(
        df_fixed
    )
    result["spectral_columns"] = spectral_columns
    result["non_spectral_columns"] = non_spectral_columns

    if verbose:
        logger.info(f"Identified {len(spectral_columns)} spectral columns")
        logger.info(
            f"Identified {len(non_spectral_columns)} non-spectral columns"
        )

    # Step 3: Remove duplicates
    df_no_duplicates = remove_duplicate_rows(df_fixed, verbose=verbose)

    # Step 4: Remove constant and empty columns
    df_cleaned = remove_constant_and_empty_columns(
        df_no_duplicates, verbose=verbose
    )

    # Step 5: Filter out non-numeric features that can't be used for modeling
    # Define columns to exclude (target and explicitly excluded columns)
    columns_to_exclude = exclude_columns.copy()
    # Don't exclude the target column yet
    if target_column in columns_to_exclude:
        columns_to_exclude.remove(target_column)

    df_numeric = filter_numeric_features(
        df_cleaned, exclude_columns=columns_to_exclude, verbose=verbose
    )

    # Step 6: Create and apply preprocessing pipeline for spectral data
    spectral_columns = [
        col for col in spectral_columns if col in df_numeric.columns
    ]

    # Create pipeline
    steps = []
    for i, transformer in enumerate(transformers):
        steps.append((f"transformer_{i}", transformer))

    preprocessing_pipeline = Pipeline(steps)

    # Extract spectral data
    X_spectral = df_numeric[spectral_columns].copy()

    # Apply preprocessing
    if verbose:
        logger.info(
            f"Applying preprocessing pipeline with {len(steps)} transformers"
        )

    X_spectral_transformed = preprocessing_pipeline.fit_transform(X_spectral)

    # Convert back to DataFrame with column names
    X_spectral_df = pd.DataFrame(
        X_spectral_transformed,
        columns=spectral_columns,
        index=X_spectral.index,
    )

    # Step 7: Combine with non-spectral numeric features
    non_spectral_numeric_cols = [
        col
        for col in df_numeric.columns
        if col not in spectral_columns and col != target_column
    ]

    if non_spectral_numeric_cols:
        if verbose:
            logger.info(
                f"Adding {
                    len(non_spectral_numeric_cols)
                } non-spectral numeric features"
            )
        X = pd.concat(
            [X_spectral_df, df_numeric[non_spectral_numeric_cols]], axis=1
        )
    else:
        X = X_spectral_df

    # Step 8: Handle outliers if requested
    if remove_outliers:
        if outlier_method == "zscore":
            outlier_mask = detect_outliers_zscore(X_spectral.values)
        elif outlier_method == "pca":
            outlier_mask = detect_outliers_pca(X_spectral.values)
        else:
            raise ValueError(f"Unknown outlier method: {outlier_method}")

        outlier_count = np.sum(outlier_mask)
        if verbose:
            logger.info(
                f"Detected {outlier_count} outliers using {
                    outlier_method
                } method"
            )

        # Remove outliers
        if outlier_count > 0:
            X = X.loc[~outlier_mask]

        result["outlier_mask"] = outlier_mask

    # Step 9: Prepare target variable if available
    if target_column in df_numeric.columns:
        y = df_numeric[target_column].loc[X.index]
        result["y"] = y
    else:
        if verbose:
            logger.warning(
                f"Target column '{target_column}' not found in DataFrame"
            )
        result["y"] = None

    # Add results to dictionary
    result["X"] = X
    result["preprocessing_pipeline"] = preprocessing_pipeline

    return result
