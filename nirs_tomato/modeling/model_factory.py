"""
Model factory functions for creating regression models.

This module provides factory functions for creating different types of
regression models with default parameter grids for hyperparameter tuning.
"""

from typing import Any, Dict, Tuple

import xgboost as xgb
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


def create_pls_model() -> Tuple[PLSRegression, Dict[str, Any]]:
    """
    Create a Partial Least Squares regression model.

    Returns:
        Tuple of (model, parameter grid)
    """
    model = PLSRegression(n_components=10, scale=False)

    param_grid = {"n_components": [2, 5, 10, 15, 20]}

    return model, param_grid


def create_svr_model() -> Tuple[SVR, Dict[str, Any]]:
    """
    Create a Support Vector Regression model.

    Returns:
        Tuple of (model, parameter grid)
    """
    model = SVR(kernel="rbf", C=1.0, gamma="scale")

    param_grid = {
        "C": [0.1, 1.0, 10.0, 100.0],
        "gamma": ["scale", "auto", 0.01, 0.1, 1.0],
        "kernel": ["rbf", "linear", "poly"],
    }

    return model, param_grid


def create_rf_model() -> Tuple[RandomForestRegressor, Dict[str, Any]]:
    """
    Create a Random Forest regression model.

    Returns:
        Tuple of (model, parameter grid)
    """
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    return model, param_grid


def create_xgb_model() -> Tuple[xgb.XGBRegressor, Dict[str, Any]]:
    """
    Create an XGBoost regression model.

    Returns:
        Tuple of (model, parameter grid)
    """
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )

    param_grid = {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 5, 7, 9],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
    }

    return model, param_grid
