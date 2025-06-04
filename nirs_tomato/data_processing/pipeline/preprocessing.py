"""
Preprocessing Pipeline Creation Module

This module provides tools to create preprocessing pipelines for NIR tomato spectroscopy data.
"""  

from typing import Any, List, Optional, Tuple

from sklearn.pipeline import Pipeline

from ..transformers import OutlierDetector, create_preprocessing_pipelines


def create_preprocessing_pipeline(
    preprocessing_method: str = "raw",
    outlier_detection: bool = True,
    outlier_method: str = "zscore",
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
        raise ValueError(
            f"Invalid preprocessing method: {
                preprocessing_method
            }. Valid options: {valid_methods}"
        )

    # Get base pipeline for the selected method
    base_pipeline = preprocessing_pipelines[preprocessing_method]

    # Create steps list from base pipeline
    steps = list(base_pipeline.steps)

    # Add outlier detection step if requested
    if outlier_detection:
        outlier_step = (
            "outlier_detector",
            OutlierDetector(
                method=outlier_method,
                threshold=outlier_threshold,
                remove_outliers=remove_outliers,
            ),
        )
        # Add as first step
        steps.insert(0, outlier_step)

    # Add any custom steps
    if custom_steps:
        steps.extend(custom_steps)

    # Create the final pipeline
    pipeline = Pipeline(steps)

    return pipeline
