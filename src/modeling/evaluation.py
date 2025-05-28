"""
Functions for evaluating regression models.

This module provides functions for evaluating regression model performance
with standard metrics like RMSE, MAE, R², etc.
"""

import numpy as np
from typing import Dict, Tuple, Any
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score, 
    explained_variance_score
)

def evaluate_regression_model(
    model: Any,
    X_test: Any,
    y_test: Any
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Evaluate a regression model on test data.
    
    Args:
        model: Trained regression model
        X_test: Test features
        y_test: True test target values
        
    Returns:
        Tuple of (metrics dict, predicted values)
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
        'mae': float(mean_absolute_error(y_test, y_pred)),
        'r2': float(r2_score(y_test, y_pred)),
        'explained_variance': float(explained_variance_score(y_test, y_pred))
    }
    
    return metrics, y_pred

def print_regression_metrics(metrics: Dict[str, float]) -> None:
    """
    Print regression metrics in a formatted way.
    
    Args:
        metrics: Dictionary of regression metrics
    """
    print("\nRegression Model Performance:")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  R²:   {metrics['r2']:.4f}")
    print(f"  Explained Variance: {metrics['explained_variance']:.4f}") 