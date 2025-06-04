"""
Regression models for NIR tomato spectroscopy Brix prediction.

This module contains functions for training and evaluating regression models
to predict Brix values from NIR spectral data of tomatoes.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.base import BaseEstimator
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

# Set plotting style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("notebook", font_scale=1.2)


def train_regression_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = "rf",
    model_params: Optional[Dict[str, Any]] = None,
    random_state: int = 42,
    verbose: bool = False,
) -> BaseEstimator:
    """
    Train a regression model for Brix prediction.

    Args:
        X_train: Training features
        y_train: Training target values
        model_type: Type of model to train ('rf', 'svr', 'elasticnet', 'xgb', 'lgbm', 'mlp', 'pls')
        model_params: Model hyperparameters
        random_state: Random seed for reproducibility
        verbose: Whether to print training information

    Returns:
        Trained regression model
    """
    # Configure logging
    logger = logging.getLogger(__name__)
    log_level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=log_level, format="%(message)s")

    # Set default parameters if not provided
    if model_params is None:
        model_params = {}

    # Create and train the model based on the specified type
    if model_type == "rf":
        # Random Forest
        default_params = {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "random_state": random_state,
        }
        # Update with provided parameters
        params = {**default_params, **model_params}
        model = RandomForestRegressor(**params)

    elif model_type == "svr":
        # Support Vector Regression
        default_params = {
            "kernel": "rbf",
            "C": 1.0,
            "epsilon": 0.1,
            "gamma": "scale",
        }
        params = {**default_params, **model_params}
        model = SVR(**params)

    elif model_type == "elasticnet":
        # Elastic Net
        default_params = {
            "alpha": 1.0,
            "l1_ratio": 0.5,
            "max_iter": 1000,
            "random_state": random_state,
        }
        params = {**default_params, **model_params}
        model = ElasticNet(**params)

    elif model_type == "xgb":
        # XGBoost
        default_params = {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 3,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": random_state,
        }
        params = {**default_params, **model_params}
        model = xgb.XGBRegressor(**params)

    elif model_type == "lgbm":
        # LightGBM
        default_params = {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": -1,
            "num_leaves": 31,
            "random_state": random_state,
            "verbose": -1,
        }
        params = {**default_params, **model_params}
        model = lgb.LGBMRegressor(**params)

    elif model_type == "mlp":
        # Multi-layer Perceptron
        default_params = {
            "hidden_layer_sizes": (100,),
            "activation": "relu",
            "solver": "adam",
            "alpha": 0.0001,
            "max_iter": 200,
            "early_stopping": True,
            "random_state": random_state,
        }
        params = {**default_params, **model_params}
        model = MLPRegressor(**params)

    elif model_type == "pls":
        # Partial Least Squares Regression
        default_params = {
            "n_components": 10,
            "scale": False,
            "max_iter": 500,
            "tol": 1e-6,
        }
        params = {**default_params, **model_params}
        model = PLSRegression(**params)

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Log training start
    if verbose:
        logger.info(f"Training {model_type.upper()} model with parameters:")
        for param, value in params.items():
            logger.info(f"  {param}: {value}")

    # Train the model
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()

    if verbose:
        training_time = end_time - start_time
        logger.info(f"Model training completed in {training_time:.2f} seconds")

    return model


def evaluate_regression_model(
    model: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: Optional[pd.DataFrame] = None,
    y_test: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    """
    Evaluate a trained regression model.

    Args:
        model: Trained regression model
        X_train: Training features
        y_train: Training target values
        X_val: Validation features
        y_val: Validation target values
        X_test: Test features (optional)
        y_test: Test target values (optional)

    Returns:
        Dictionary with evaluation metrics
    """
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    # Calculate metrics for training set
    train_metrics = {
        "rmse": np.sqrt(mean_squared_error(y_train, y_train_pred)),
        "mae": mean_absolute_error(y_train, y_train_pred),
        "r2": r2_score(y_train, y_train_pred),
        "explained_variance": explained_variance_score(y_train, y_train_pred),
    }

    # Calculate metrics for validation set
    val_metrics = {
        "rmse": np.sqrt(mean_squared_error(y_val, y_val_pred)),
        "mae": mean_absolute_error(y_val, y_val_pred),
        "r2": r2_score(y_val, y_val_pred),
        "explained_variance": explained_variance_score(y_val, y_val_pred),
    }

    # Calculate metrics for test set if provided
    test_metrics = None
    if X_test is not None and y_test is not None:
        y_test_pred = model.predict(X_test)
        test_metrics = {
            "rmse": np.sqrt(mean_squared_error(y_test, y_test_pred)),
            "mae": mean_absolute_error(y_test, y_test_pred),
            "r2": r2_score(y_test, y_test_pred),
            "explained_variance": explained_variance_score(
                y_test, y_test_pred
            ),
        }

    # Prepare results
    results = {
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "y_train_pred": y_train_pred,
        "y_val_pred": y_val_pred,
        "y_test_pred": y_test_pred if X_test is not None else None,
    }

    return results


def hyperparameter_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_type: str = "rf",
    param_grid: Optional[Dict[str, List[Any]]] = None,
    n_iter: int = 20,
    cv: int = 3,
    scoring: str = "neg_root_mean_squared_error",
    random_state: int = 42,
    verbose: bool = False,
) -> Tuple[BaseEstimator, Dict[str, Any]]:
    """
    Perform hyperparameter search for a regression model.

    Args:
        X_train: Training features
        y_train: Training target values
        X_val: Validation features
        y_val: Validation target values
        model_type: Type of model to train
        param_grid: Hyperparameter grid to search
        n_iter: Number of iterations for randomized search
        cv: Number of cross-validation folds
        scoring: Scoring metric for model selection
        random_state: Random seed for reproducibility
        verbose: Whether to print search information

    Returns:
        Tuple of (best_model, search_results)
    """
    # Configure logging
    logger = logging.getLogger(__name__)
    log_level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=log_level, format="%(message)s")

    # Set default parameter grid if not provided
    if param_grid is None:
        if model_type == "rf":
            param_grid = {
                "n_estimators": [50, 100, 200, 300],
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            }
        elif model_type == "svr":
            param_grid = {
                "kernel": ["linear", "poly", "rbf"],
                "C": [0.1, 1.0, 10.0, 100.0],
                "epsilon": [0.01, 0.1, 0.2],
                "gamma": ["scale", "auto", 0.1, 0.01],
            }
        elif model_type == "elasticnet":
            param_grid = {
                "alpha": [0.0001, 0.001, 0.01, 0.1, 1.0],
                "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
                "max_iter": [500, 1000, 2000],
            }
        elif model_type == "xgb":
            param_grid = {
                "n_estimators": [50, 100, 200, 300],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "max_depth": [3, 5, 7, 9],
                "subsample": [0.6, 0.8, 1.0],
                "colsample_bytree": [0.6, 0.8, 1.0],
            }
        elif model_type == "lgbm":
            param_grid = {
                "n_estimators": [50, 100, 200, 300],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "max_depth": [-1, 5, 10, 15],
                "num_leaves": [31, 50, 100, 200],
            }
        elif model_type == "mlp":
            param_grid = {
                "hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50)],
                "activation": ["relu", "tanh"],
                "alpha": [0.0001, 0.001, 0.01],
                "learning_rate": ["constant", "adaptive"],
                "max_iter": [200, 500],
            }

    # Create base model
    model = train_regression_model(
        X_train=X_train, y_train=y_train, model_type=model_type, verbose=False
    )

    # Create randomized search
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        random_state=random_state,
        verbose=1 if verbose else 0,
    )

    # Run search
    if verbose:
        logger.info(
            f"Starting hyperparameter search for {model_type.upper()} with {
                n_iter
            } iterations..."
        )

    start_time = time.time()
    search.fit(X_train, y_train)
    end_time = time.time()

    if verbose:
        search_time = end_time - start_time
        logger.info(
            f"Hyperparameter search completed in {search_time:.2f} seconds"
        )
        logger.info(f"Best parameters: {search.best_params_}")
        logger.info(f"Best score: {search.best_score_:.4f}")

    # Evaluate best model
    best_model = search.best_estimator_
    results = evaluate_regression_model(
        model=best_model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
    )

    if verbose:
        logger.info(f"Validation RMSE: {results['val_metrics']['rmse']:.4f}")
        logger.info(f"Validation R²: {results['val_metrics']['r2']:.4f}")

    # Prepare search results
    search_results = {
        "best_params": search.best_params_,
        "best_score": search.best_score_,
        "cv_results": search.cv_results_,
        "model_evaluation": results,
        "search_time": search_time if verbose else end_time - start_time,
    }

    return best_model, search_results


def save_model(
    model: BaseEstimator,
    model_path: str,
    model_info: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Save a trained model to disk.

    Args:
        model: Trained model to save
        model_path: Path where the model will be saved
        model_info: Additional model information to save

    Returns:
        Path to the saved model
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Create model package with both model and info
    model_package = {"model": model, "info": model_info or {}}

    # Save the model package
    joblib.dump(model_package, model_path)

    return model_path


def load_model(model_path: str) -> Tuple[BaseEstimator, Dict[str, Any]]:
    """
    Load a trained model from disk.

    Args:
        model_path: Path to the saved model

    Returns:
        Tuple of (model, model_info)
    """
    # Load the model package
    model_package = joblib.load(model_path)

    # Extract model and info
    model = model_package["model"]
    model_info = model_package["info"]

    return model, model_info


def plot_regression_results(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    title: str = "Predicted vs Actual",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot regression results.

    Args:
        y_true: True target values
        y_pred: Predicted target values
        title: Plot title
        figsize: Figure size (width, height)
        save_path: Path to save the plot (optional)

    Returns:
        Matplotlib figure
    """
    # Convert to numpy arrays if needed
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot scatter plot
    ax.scatter(y_true, y_pred, alpha=0.6)

    # Plot perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--")

    # Add metrics to plot
    ax.text(
        0.05,
        0.95,
        f"RMSE: {rmse:.4f}\nR²: {r2:.4f}",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Customize plot
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Tight layout
    plt.tight_layout()

    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def compare_models(
    models_results: Dict[str, Dict[str, Any]],
    metric: str = "rmse",
    dataset: str = "val",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Compare multiple regression models.

    Args:
        models_results: Dictionary with model results
        metric: Metric to compare ('rmse', 'mae', 'r2', 'explained_variance')
        dataset: Dataset to compare ('train', 'val', 'test')
        figsize: Figure size (width, height)
        save_path: Path to save the plot (optional)

    Returns:
        Matplotlib figure
    """
    # Extract metric values for each model
    model_names = []
    metric_values = []

    for model_name, results in models_results.items():
        dataset_key = f"{dataset}_metrics"
        if dataset_key in results:
            model_names.append(model_name)
            metric_values.append(results[dataset_key][metric])

    # Sort by metric value (lower is better for rmse and mae, higher is better
    # for r2 and explained_variance)
    if metric in ["rmse", "mae"]:
        # Sort ascending for error metrics
        sorted_indices = np.argsort(metric_values)
    else:
        # Sort descending for R² and explained variance
        sorted_indices = np.argsort(metric_values)[::-1]

    sorted_model_names = [model_names[i] for i in sorted_indices]
    sorted_metric_values = [metric_values[i] for i in sorted_indices]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create bar chart
    bars = ax.bar(sorted_model_names, sorted_metric_values)

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.4f}",
            ha="center",
            va="bottom",
        )

    # Customize plot
    ax.set_xlabel("Models")
    ax.set_ylabel(metric.upper())
    ax.set_title(
        f"Model Comparison - {metric.upper()} on {dataset.capitalize()} Set"
    )
    ax.grid(True, alpha=0.3, axis="y")

    # Rotate x-labels for better readability
    plt.xticks(rotation=45, ha="right")

    # Tight layout
    plt.tight_layout()

    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig
