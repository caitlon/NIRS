"""
Data Processing Module

This module provides tools for processing NIR spectroscopy data.
"""

from .utils import (
    identify_spectral_columns,
    fix_column_names,
    remove_duplicate_rows,
    add_sample_identifier,
    detect_outliers_zscore,
    detect_outliers_pca,
    remove_constant_and_empty_columns,
    split_data
)

from .transformers import (
    SNVTransformer,
    SavGolTransformer,
    MSCTransformer,
    OutlierDetector,
    PCATransformer,
    create_preprocessing_pipelines
)

from .pipeline import (
    create_preprocessing_pipeline,
    process_and_save_data,
    load_and_preprocess_data,
    prepare_data_for_regression
)

from .constants import (
    TARGET_COLUMN,
    DEFAULT_DATASET_PATH,
    AVAILABLE_PREPROCESSING_METHODS,
    AVAILABLE_AGGREGATION_METHODS,
    REGRESSION_METRICS,
    REGRESSION_MODELS
) 