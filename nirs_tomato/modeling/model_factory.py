"""
Model factory functions for creating regression models.

This module provides factory functions for creating different types of
regression models with default parameter grids for hyperparameter tuning.
"""

from typing import Any, Dict, Tuple, Union

import lightgbm as lgb
import xgboost as xgb
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR


def create_model(model_type: str, **kwargs) -> Union[
    PLSRegression,
    SVR,
    RandomForestRegressor,
    xgb.XGBRegressor,
    lgb.LGBMRegressor,
]:
    """
    Create a regression model of specified type with provided parameters.

    Args:
        model_type: Type of model to create ('pls', 'svr', 'rf', 'xgb', 'lgbm', 'mlp')
        **kwargs: Parameters to pass to the model constructor

    Returns:
        Instantiated regression model

    Raises:
        ValueError: If model_type is not recognized
    """
    if model_type == "pls":
        n_components = kwargs.get("n_components", 10)
        scale = kwargs.get("scale", False)
        return PLSRegression(n_components=n_components, scale=scale)

    elif model_type == "svr":
        kernel = kwargs.get("kernel", "rbf")
        C = kwargs.get("C", 1.0)
        gamma = kwargs.get("gamma", "scale")
        epsilon = kwargs.get("epsilon", 0.1)
        return SVR(kernel=kernel, C=C, gamma=gamma, epsilon=epsilon)

    elif model_type == "rf":
        n_estimators = kwargs.get("n_estimators", 100)
        max_depth = kwargs.get("max_depth", None)
        min_samples_split = kwargs.get("min_samples_split", 2)
        min_samples_leaf = kwargs.get("min_samples_leaf", 1)
        random_state = kwargs.get("random_state", 42)
        return RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )

    elif model_type == "xgb":
        n_estimators = kwargs.get("n_estimators", 100)
        learning_rate = kwargs.get("learning_rate", 0.1)
        max_depth = kwargs.get("max_depth", 3)
        subsample = kwargs.get("subsample", 0.8)
        colsample_bytree = kwargs.get("colsample_bytree", 0.8)
        random_state = kwargs.get("random_state", 42)
        return xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
        )

    elif model_type == "lgbm":
        n_estimators = kwargs.get("n_estimators", 100)
        learning_rate = kwargs.get("learning_rate", 0.1)
        max_depth = kwargs.get("max_depth", -1)
        num_leaves = kwargs.get("num_leaves", 31)
        random_state = kwargs.get("random_state", 42)
        return lgb.LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=num_leaves,
            random_state=random_state,
        )

    elif model_type == "mlp":
        hidden_layer_sizes = kwargs.get("hidden_layer_sizes", (100,))
        activation = kwargs.get("activation", "relu")
        solver = kwargs.get("solver", "adam")
        random_state = kwargs.get("random_state", 42)
        return MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            random_state=random_state,
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")


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
