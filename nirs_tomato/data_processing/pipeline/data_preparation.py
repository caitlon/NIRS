"""
Data Preparation Module for NIR Tomato Spectroscopy

This module provides functions for preparing data for regression tasks
with NIR spectroscopy data.
"""

import logging
from typing import Any, Dict, List, Optional

from ..constants import DEFAULT_DATASET_PATH
from ..utils import split_data
from .data_processing import load_and_preprocess_data


def prepare_data_for_regression(
    data_path: str = DEFAULT_DATASET_PATH,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
    preprocessing_method: str = "raw",
    outlier_detection: bool = True,
    remove_constant_columns: bool = True,
    impute_missing_values: bool = True,
    additional_features: Optional[List[str]] = None,
    exclude_features: Optional[List[str]] = None,
    verbose: bool = False,
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
    logging.basicConfig(level=log_level, format="%(message)s")
    logger = logging.getLogger(__name__)

    # Load and preprocess data
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

    # Filter features if specified
    if exclude_features:
        X = X.drop(
            columns=[col for col in exclude_features if col in X.columns]
        )
        logger.info(f"Excluded {len(exclude_features)} features")

    if additional_features:
        # Keep only specified additional features from non-spectral columns
        spectral_cols = preprocessing_info["spectral_columns"]
        keep_cols = spectral_cols + [
            col for col in additional_features if col in X.columns
        ]
        X = X[keep_cols]
        logger.info(
            f"Keeping {len(spectral_cols)} spectral columns and {len(additional_features)} additional features"
        )

    # Split data into train, validation, and test sets
    logger.info("Splitting data into train, validation, and test sets...")
    splits = split_data(
        X=X,
        y=y,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
    )

    return {
        "X_train": splits["X_train"],
        "y_train": splits["y_train"],
        "X_val": splits["X_val"],
        "y_val": splits["y_val"],
        "X_test": splits["X_test"],
        "y_test": splits["y_test"],
        "preprocessing_info": preprocessing_info,
        "feature_names": list(X.columns),
    }
