"""
Functions for creating and managing NIR tomato spectroscopy data processing pipelines.

This module has been refactored for better modularity.
The pipeline functions are now located in the pipeline/ directory.

This file is kept for backward compatibility.
"""  

from .pipeline.data_preparation import prepare_data_for_regression
from .pipeline.data_processing import (
    load_and_preprocess_data,
    preprocess_spectra,
    process_and_save_data,
)
from .pipeline.preprocessing import create_preprocessing_pipeline

__all__ = [
    "create_preprocessing_pipeline",
    "process_and_save_data",
    "load_and_preprocess_data",
    "prepare_data_for_regression",
    "preprocess_spectra",
]
