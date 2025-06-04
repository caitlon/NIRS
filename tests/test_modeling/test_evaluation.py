"""
Tests for model evaluation functions.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from nirs_tomato.modeling.evaluation import evaluate_regression_model


def test_evaluate_regression_model():
    """Test regression model evaluation functionality."""
    # Create simple data
    X_test = np.array([[1], [2], [3], [4]])
    y_test = np.array([1, 2, 3, 4])

    # Create and fit a simple model
    model = LinearRegression()
    model.fit(X_test, y_test)  # Perfect fit for this data

    # Evaluate model
    metrics, y_pred = evaluate_regression_model(model, X_test, y_test)

    # Check metrics
    assert "rmse" in metrics
    assert "mae" in metrics
    assert "r2" in metrics
    assert "explained_variance" in metrics

    # For perfect fit, metrics should be ideal
    assert metrics["rmse"] < 1e-10
    assert metrics["mae"] < 1e-10
    assert np.isclose(metrics["r2"], 1.0)
    assert np.isclose(metrics["explained_variance"], 1.0)

    # Check predictions
    assert np.allclose(y_pred, y_test)


def test_evaluate_regression_model_imperfect():
    """Test evaluation with imperfect predictions."""
    # Create data with noise
    X_test = np.array([[1], [2], [3], [4], [5]])
    y_test = np.array([1.1, 2.2, 2.9, 4.1, 5.2])  # Some noise

    # Create basic model
    model = LinearRegression()
    model.fit(X_test, y_test)

    # Evaluate model
    metrics, y_pred = evaluate_regression_model(model, X_test, y_test)

    # Check reasonable metrics for good but not perfect fit
    assert 0 < metrics["rmse"] < 0.5
    assert 0 < metrics["mae"] < 0.5
    assert 0.9 < metrics["r2"] <= 1.0
    assert 0.9 < metrics["explained_variance"] <= 1.0

    # Check predictions were made
    assert len(y_pred) == len(y_test)


def test_evaluate_regression_model_with_dataframe():
    """Test evaluation with pandas DataFrame and Series."""
    # Create dataframe data
    X_test = pd.DataFrame({"x1": [1, 2, 3, 4], "x2": [0.1, 0.2, 0.3, 0.4]})
    y_test = pd.Series([1, 2, 3, 4])

    # Create and fit model
    model = LinearRegression()
    model.fit(X_test, y_test)

    # Evaluate model
    metrics, y_pred = evaluate_regression_model(model, X_test, y_test)

    # Check metrics exist
    assert "rmse" in metrics
    assert "mae" in metrics
    assert "r2" in metrics
    assert metrics["r2"] > 0.9  # Should be good fit

    # Check predictions format matches input
    assert len(y_pred) == len(y_test)
