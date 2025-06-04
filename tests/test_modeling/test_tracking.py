"""
Tests for MLflow tracking utilities.

These tests ensure the MLflow tracking functions work correctly.
"""

from unittest.mock import MagicMock, patch

import pytest

from nirs_tomato.modeling.tracking import (
    end_run,
    log_artifact,
    log_figure,
    log_metrics,
    log_model,
    log_parameters,
    setup_mlflow,
    start_run,
)


@pytest.fixture
def mock_mlflow():
    """Mock MLflow functions."""
    with patch("nirs_tomato.modeling.tracking.mlflow") as mock_mlflow:
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.active_run.return_value = mock_run
        mock_mlflow.get_experiment_by_name.return_value = None
        mock_mlflow.create_experiment.return_value = "test_experiment_id"
        mock_mlflow.set_tracking_uri = MagicMock()
        yield mock_mlflow


def test_setup_mlflow(mock_mlflow):
    """Test MLflow setup with new experiment."""
    experiment_id = setup_mlflow(experiment_name="test_experiment")
    assert experiment_id == "test_experiment_id"
    mock_mlflow.create_experiment.assert_called_once_with(
        name="test_experiment"
    )


def test_setup_mlflow_existing(mock_mlflow):
    """Test MLflow setup with existing experiment."""
    # Mock an existing experiment
    mock_experiment = MagicMock()
    mock_experiment.experiment_id = "existing_id"
    mock_mlflow.get_experiment_by_name.return_value = mock_experiment

    experiment_id = setup_mlflow(experiment_name="existing_experiment")
    assert experiment_id == "existing_id"
    mock_mlflow.create_experiment.assert_not_called()


def test_start_run(mock_mlflow):
    """Test starting an MLflow run."""
    run_id = start_run(run_name="test_run", experiment_name="test_experiment")
    assert run_id == "test_run_id"
    mock_mlflow.start_run.assert_called_once()


def test_log_parameters(mock_mlflow):
    """Test logging parameters to MLflow."""
    params = {"param1": 1, "param2": "value2", "param3": [1, 2, 3]}
    log_parameters(params)
    assert mock_mlflow.log_param.call_count == 3


def test_log_metrics(mock_mlflow):
    """Test logging metrics to MLflow."""
    metrics = {"metric1": 0.9, "metric2": 0.85}
    log_metrics(metrics)
    assert mock_mlflow.log_metric.call_count == 2


def test_log_model(mock_mlflow):
    """Test logging a model to MLflow."""
    model = MagicMock()
    log_model(model)
    mock_mlflow.sklearn.log_model.assert_called_once()


def test_log_figure(mock_mlflow):
    """Test logging a figure to MLflow."""
    fig = MagicMock()
    log_figure(fig, "test_figure.png")
    mock_mlflow.log_figure.assert_called_once_with(fig, "test_figure.png")


def test_log_artifact(mock_mlflow):
    """Test logging an artifact to MLflow."""
    log_artifact("test_file.txt")
    mock_mlflow.log_artifact.assert_called_once_with("test_file.txt", None)


def test_end_run(mock_mlflow):
    """Test ending an MLflow run."""
    end_run()
    mock_mlflow.end_run.assert_called_once()
