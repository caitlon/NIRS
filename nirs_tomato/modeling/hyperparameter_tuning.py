"""
Advanced hyperparameter tuning using Bayesian optimization with Optuna.

This module provides functions for optimizing model hyperparameters
using Bayesian optimization, which is much more efficient than
grid search or random search.
"""

import logging
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score

from .regression_models import (
    evaluate_regression_model,
    train_regression_model,
)


def bayesian_hyperparameter_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_type: str = "rf",
    n_trials: int = 50,
    cv: int = 3,
    scoring: str = "neg_root_mean_squared_error",
    timeout: Optional[int] = None,
    random_state: int = 42,
    verbose: bool = True,
    study_name: Optional[str] = None,
    direction: str = "maximize",
) -> Tuple[BaseEstimator, Dict[str, Any]]:
    """
    Perform hyperparameter optimization using Bayesian optimization with Optuna.

    Args:
        X_train: Training features
        y_train: Training target values
        X_val: Validation features
        y_val: Validation target values
        model_type: Type of model to train ('pls', 'svr', 'rf', 'xgb', 'lgbm')
        n_trials: Number of optimization trials
        cv: Number of cross-validation folds
        scoring: Scoring metric for model selection
        timeout: Time limit in seconds for the optimization
        random_state: Random seed for reproducibility
        verbose: Whether to print optimization information
        study_name: Name for the optimization study
        direction: Direction of optimization ('maximize' or 'minimize')

    Returns:
        Tuple of (best_model, search_results)
    """  
    # Configure logging
    logger = logging.getLogger(__name__)
    log_level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=log_level, format="%(message)s")

    # Create study name if not provided
    if study_name is None:
        study_name = (
            f"{model_type}_optimization_{time.strftime('%Y%m%d_%H%M%S')}"
        )

    # Set up cross-validation
    cv_folds = KFold(n_splits=cv, shuffle=True, random_state=random_state)

    # Define the objective function for optimization
    def objective(trial: optuna.Trial) -> float:
        # Get hyperparameters based on the model type
        params = _get_hyperparameters_for_trial(
            trial, model_type, random_state
        )

        try:
            # Train the model with current hyperparameters
            model = train_regression_model(
                X_train=X_train,
                y_train=y_train,
                model_type=model_type,
                model_params=params,
                random_state=random_state,
                verbose=False,
            )

            # Compute cross-validation score
            cv_scores = cross_val_score(
                model,
                X_train,
                y_train,
                cv=cv_folds,
                scoring=scoring,
                n_jobs=-1,
            )

            # Calculate mean CV score
            mean_score = np.mean(cv_scores)

            # For minimization metrics, negate the score
            if "neg_" in scoring and direction == "maximize":
                mean_score = -mean_score

            # Report intermediate values for pruning
            if X_val is not None and y_val is not None:
                y_val_pred = model.predict(X_val)
                val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
                val_r2 = r2_score(y_val, y_val_pred)

                trial.report(mean_score, step=0)
                trial.set_user_attr("val_rmse", val_rmse)
                trial.set_user_attr("val_r2", val_r2)

                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            return mean_score

        except Exception as e:
            logger.warning(f"Trial failed with error: {str(e)}")
            return float("-inf") if direction == "maximize" else float("inf")

    # Create a study object and optimize the objective function
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    study = optuna.create_study(
        study_name=study_name, direction=direction, pruner=pruner
    )

    # Start optimization
    start_time = time.time()
    if verbose:
        logger.info(
            f"Starting Bayesian hyperparameter optimization for {
                model_type.upper()
            } model"
        )
        logger.info(
            f"Running {n_trials} trials with {cv}-fold cross-validation"
        )

    # Run optimization
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=verbose,
    )

    # Calculate elapsed time
    elapsed_time = time.time() - start_time

    if verbose:
        best_params = study.best_params
        best_value = study.best_value
        logger.info(f"Optimization completed in {elapsed_time:.2f} seconds")
        logger.info(f"Best score: {best_value:.4f}")
        logger.info("Best hyperparameters:")
        for param, value in best_params.items():
            logger.info(f"  {param}: {value}")

    # Train final model with best parameters
    final_params = study.best_params.copy()

    # Remove control parameters that aren't actual model parameters
    if model_type == "rf" and "limit_depth" in final_params:
        # Remove the control parameter
        limit_depth = final_params.pop("limit_depth")
        # If limit_depth was False, set max_depth to None
        if not limit_depth and "max_depth" in final_params:
            final_params["max_depth"] = None

    # SVR specific parameter cleanup
    if model_type == "svr":
        if "gamma_auto" in final_params:
            final_params.pop("gamma_auto")
        if "gamma_value" in final_params and "gamma" not in final_params:
            final_params["gamma"] = final_params.pop("gamma_value")

    best_model = train_regression_model(
        X_train=X_train,
        y_train=y_train,
        model_type=model_type,
        model_params=final_params,
        random_state=random_state,
        verbose=verbose,
    )

    # Evaluate on validation set
    if X_val is not None and y_val is not None:
        results = evaluate_regression_model(
            model=best_model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
        )

        if verbose:
            logger.info("Validation metrics with best parameters:")
            for metric, value in results["val_metrics"].items():
                logger.info(f"  {metric}: {value:.4f}")

    # Prepare search results
    search_results = {
        "best_params": study.best_params,
        "best_score": study.best_value,
        "n_trials": len(study.trials),
        "elapsed_time": elapsed_time,
        "study": study,
        "val_metrics": results["val_metrics"] if X_val is not None else None,
    }

    return best_model, search_results


def _get_hyperparameters_for_trial(
    trial: optuna.Trial, model_type: str, random_state: int = 42
) -> Dict[str, Any]:
    """
    Define hyperparameter search space based on model type.

    Args:
        trial: Optuna trial object
        model_type: Type of model ('pls', 'svr', 'rf', 'xgb', 'lgbm')
        random_state: Random seed

    Returns:
        Dictionary of hyperparameters
    """
    if model_type == "pls":
        return {
            "n_components": trial.suggest_int("n_components", 1, 30),
            "scale": trial.suggest_categorical("scale", [True, False]),
        }

    elif model_type == "svr":
        return {
            "kernel": trial.suggest_categorical(
                "kernel", ["linear", "poly", "rbf", "sigmoid"]
            ),
            "C": trial.suggest_float("C", 0.01, 100.0, log=True),
            "epsilon": trial.suggest_float("epsilon", 0.01, 1.0, log=True),
            "gamma": (
                trial.suggest_categorical("gamma", ["scale", "auto"])
                if trial.suggest_categorical("gamma_auto", [True, False])
                else trial.suggest_float("gamma_value", 0.001, 1.0, log=True)
            ),
        }

    elif model_type == "rf":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": (
                trial.suggest_int("max_depth", 3, 50)
                if trial.suggest_categorical("limit_depth", [True, False])
                else None
            ),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", None]
            ),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "random_state": random_state,
        }

    elif model_type == "xgb":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.3, log=True
            ),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.5, 1.0
            ),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "random_state": random_state,
        }

    elif model_type == "lgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.3, log=True
            ),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "max_depth": trial.suggest_int("max_depth", -1, 50),
            "min_child_samples": trial.suggest_int(
                "min_child_samples", 5, 100
            ),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.5, 1.0
            ),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0, log=True),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", 0.0, 10.0, log=True
            ),
            "random_state": random_state,
        }

    else:
        raise ValueError(f"Unsupported model type: {model_type}")
