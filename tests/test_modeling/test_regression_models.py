"""
Tests for regression models in the modeling module.
"""

import os
import tempfile

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor

from nirs_tomato.modeling.regression_models import (
    evaluate_regression_model,
    load_model,
    save_model,
    train_regression_model,
)


def test_train_regression_model(
    sample_spectra_data, sample_target_data, train_test_split_indices
):
    """Test that models can be trained successfully."""
    # Prepare train data
    X_train = sample_spectra_data.iloc[train_test_split_indices["train"]]
    y_train = sample_target_data.iloc[train_test_split_indices["train"]]

    # Test different model types
    model_types = ["rf", "pls", "xgb"]

    for model_type in model_types:
        # Train model
        model = train_regression_model(
            X_train=X_train,
            y_train=y_train,
            model_type=model_type,
            random_state=42,
            verbose=False,
        )

        # Check that model is trained and has correct type
        if model_type == "rf":
            assert isinstance(model, RandomForestRegressor)
        elif model_type == "pls":
            assert isinstance(model, PLSRegression)

        # Check that model can predict
        predictions = model.predict(X_train)
        if (
            isinstance(predictions, np.ndarray)
            and predictions.ndim > 1
            and predictions.shape[1] == 1
        ):
            predictions = predictions.flatten()

        assert len(predictions) == len(y_train)


def test_evaluate_regression_model(
    sample_spectra_data, sample_target_data, train_test_split_indices
):
    """Test model evaluation functionality."""
    # Prepare train and validation data
    X_train = sample_spectra_data.iloc[train_test_split_indices["train"]]
    y_train = sample_target_data.iloc[train_test_split_indices["train"]]
    X_val = sample_spectra_data.iloc[train_test_split_indices["test"]]
    y_val = sample_target_data.iloc[train_test_split_indices["test"]]

    # Train a simple model
    model = train_regression_model(
        X_train=X_train,
        y_train=y_train,
        model_type="rf",
        random_state=42,
        verbose=False,
    )

    # Evaluate the model
    evaluation = evaluate_regression_model(
        model=model, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val
    )

    # Check that evaluation contains metrics
    assert "train_metrics" in evaluation
    assert "val_metrics" in evaluation

    # Check specific metrics in train_metrics
    train_metrics = evaluation["train_metrics"]
    assert "rmse" in train_metrics
    assert "mae" in train_metrics
    assert "r2" in train_metrics

    # Check specific metrics in val_metrics
    val_metrics = evaluation["val_metrics"]
    assert "rmse" in val_metrics
    assert "mae" in val_metrics
    assert "r2" in val_metrics

    # Check that all metrics are floats
    for _metric_name, metric_value in train_metrics.items():
        assert isinstance(metric_value, (float, np.float64))

    for _metric_name, metric_value in val_metrics.items():
        assert isinstance(metric_value, (float, np.float64))


def test_save_and_load_model(
    sample_spectra_data, sample_target_data, train_test_split_indices
):
    """Test that models can be saved and loaded correctly."""
    # Prepare train data
    X_train = sample_spectra_data.iloc[train_test_split_indices["train"]]
    y_train = sample_target_data.iloc[train_test_split_indices["train"]]

    # Train a simple model
    model = train_regression_model(
        X_train=X_train,
        y_train=y_train,
        model_type="rf",
        random_state=42,
        verbose=False,
    )

    # Create a temporary directory and file for the model
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = os.path.join(tmp_dir, "test_model.joblib")

        # Save the model
        model_info = {
            "model_type": "rf",
            "n_features": X_train.shape[1],
            "date_trained": "2023-01-01",
        }
        save_path = save_model(model, model_path, model_info)

        # Check that file exists
        assert os.path.exists(save_path)

        # Load the model
        loaded_model, loaded_info = load_model(save_path)

        # Check that loaded model info matches original
        assert loaded_info["model_type"] == model_info["model_type"]
        assert loaded_info["n_features"] == model_info["n_features"]

        # Check that loaded model makes the same predictions
        original_preds = model.predict(X_train)
        loaded_preds = loaded_model.predict(X_train)
        assert np.allclose(original_preds, loaded_preds)
