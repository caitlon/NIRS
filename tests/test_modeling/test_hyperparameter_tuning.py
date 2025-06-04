"""
Tests for hyperparameter tuning functionality.
"""

import numpy as np

from nirs_tomato.modeling.hyperparameter_tuning import (
    _get_hyperparameters_for_trial,
    bayesian_hyperparameter_search,
)


def test_bayesian_hyperparameter_search():
    """Test the core hyperparameter optimization function."""
    # Create synthetic data
    n_samples, n_features = 30, 10
    X = np.random.rand(n_samples, n_features)
    X_train = X[:20]
    X_val = X[20:]
    y = np.random.rand(n_samples)
    y_train = y[:20]
    y_val = y[20:]

    # Convert to pandas
    import pandas as pd

    X_train = pd.DataFrame(X_train)
    X_val = pd.DataFrame(X_val)
    y_train = pd.Series(y_train)
    y_val = pd.Series(y_val)

    # Run optimization with minimal trials to speed up test
    best_model, search_results = bayesian_hyperparameter_search(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        model_type="rf",  # Random Forest is typically fast
        n_trials=3,  # Very few trials for test
        cv=2,  # Small CV to speed up test
        random_state=42,
        verbose=False,
    )

    # Check that we got valid results
    assert best_model is not None
    assert "best_params" in search_results
    assert "best_score" in search_results
    assert isinstance(search_results["best_params"], dict)

    # Check that model can make predictions
    preds = best_model.predict(X_val)
    assert len(preds) == len(X_val)


def test_get_hyperparameters_for_trial():
    """Test hyperparameter space definition for different models."""
    # Instead of creating actual trial, use MagicMock
    from unittest.mock import MagicMock

    # Test with different model types
    model_types = ["pls", "svr", "rf", "xgb", "lgbm"]

    for model_type in model_types:
        # Create mock object
        trial = MagicMock()

        # Configure return values for methods
        trial.suggest_int.return_value = 5
        trial.suggest_float.return_value = 0.5
        trial.suggest_categorical.return_value = (
            "linear" if model_type == "svr" else True
        )

        # Test hyperparameter generation
        params = _get_hyperparameters_for_trial(
            trial, model_type, random_state=42
        )

        # Check that we got valid parameters
        assert isinstance(params, dict)
        assert len(params) > 0

        # Check for random_state in models that support it
        if model_type in ["rf", "xgb", "lgbm"]:
            assert "random_state" in params
