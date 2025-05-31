"""
Functions for creating and managing NIR tomato spectroscopy data processing pipelines.

This module has been refactored for better modularity.
The pipeline functions are now located in the pipeline/ directory.

This file is kept for backward compatibility.
"""

from .pipeline.preprocessing import create_preprocessing_pipeline
from .pipeline.data_processing import (
    process_and_save_data, 
    load_and_preprocess_data,
    preprocess_spectra
)
from .pipeline.data_preparation import prepare_data_for_regression

__all__ = [
    "create_preprocessing_pipeline",
    "process_and_save_data",
    "load_and_preprocess_data",
    "prepare_data_for_regression",
    "preprocess_spectra"
] 