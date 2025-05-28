"""
Modeling Module

This module provides tools for modeling NIR spectroscopy data
to predict Brix values in tomatoes.
"""

from .regression_models import (
    train_regression_model,
    evaluate_regression_model,
    hyperparameter_search,
    save_model,
    load_model,
    plot_regression_results,
    compare_models
) 