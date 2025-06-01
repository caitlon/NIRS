"""
MLflow tracking utilities for NIRS tomato models.

This module provides helper functions for tracking experiments with MLflow.
"""

import logging
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)


def setup_mlflow(
    experiment_name: str = "nirs-tomato",
    tracking_uri: Optional[str] = None,
    artifacts_uri: Optional[str] = None,
) -> str:
    """
    Setup MLflow tracking.

    Args:
        experiment_name: Name of the MLflow experiment
        tracking_uri: URI of the MLflow tracking server (None for local)
        artifacts_uri: URI for storing artifacts (None for default)

    Returns:
        ID of the experiment
    """
    # Set MLflow tracking URI if provided
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        logger.info(f"Set MLflow tracking URI to {tracking_uri}")

    # Get or create the experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        if artifacts_uri:
            experiment_id = mlflow.create_experiment(
                name=experiment_name, artifact_location=artifacts_uri
            )
            logger.info(f"Created new experiment '{
                experiment_name}' with artifacts at {artifacts_uri}")
        else:
            experiment_id = mlflow.create_experiment(name=experiment_name)
            logger.info(f"Created new experiment '{experiment_name}'")
    else:
        experiment_id = experiment.experiment_id
        logger.info(f"Using existing experiment '{
            experiment_name}' (ID: {experiment_id})")

    return experiment_id


def start_run(
    run_name: Optional[str] = None,
    experiment_name: str = "nirs-tomato",
    tracking_uri: Optional[str] = None,
    artifacts_uri: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
) -> str:
    """
    Start a new MLflow run.

    Args:
        run_name: Name for the run
        experiment_name: Name of the MLflow experiment
        tracking_uri: URI of the MLflow tracking server
        artifacts_uri: URI for storing artifacts
        tags: Tags to set on the run

    Returns:
        ID of the run
    """
    # Setup MLflow experiment
    experiment_id = setup_mlflow(
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
        artifacts_uri=artifacts_uri,
    )

    # Start a new run
    mlflow.start_run(run_name=run_name, experiment_id=experiment_id, tags=tags)

    run_id = mlflow.active_run().info.run_id
    logger.info(f"Started MLflow run '{run_name}' with ID: {run_id}")

    return run_id


def log_parameters(params: Dict[str, Any]) -> None:
    """
    Log parameters to MLflow.

    Args:
        params: Dictionary of parameters to log
    """
    if not mlflow.active_run():
        logger.warning("No active MLflow run. Parameters will not be logged.")
        return

    # Log each parameter
    for key, value in params.items():
        # Convert non-string parameters to strings if needed
        if isinstance(value, (list, dict, tuple)):
            value = str(value)

        mlflow.log_param(key, value)

    logger.info(f"Logged {len(params)} parameters to MLflow")


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    """
    Log metrics to MLflow.

    Args:
        metrics: Dictionary of metrics to log
        step: Step number (optional)
    """
    if not mlflow.active_run():
        logger.warning("No active MLflow run. Metrics will not be logged.")
        return

    # Log each metric
    for key, value in metrics.items():
        # Make sure value is a float
        if isinstance(value, np.ndarray):
            value = float(value.item())
        else:
            value = float(value)

        mlflow.log_metric(key, value, step=step)

    logger.info(f"Logged {len(metrics)} metrics to MLflow")


def log_model(
    model: BaseEstimator,
    artifact_path: str = "model",
    registered_model_name: Optional[str] = None,
    conda_env: Optional[Dict[str, Any]] = None,
    signature: Optional[Any] = None,
    input_example: Optional[Any] = None,
) -> None:
    """
    Log model to MLflow.

    Args:
        model: Trained model to log
        artifact_path: Path within the run artifacts to store the model
        registered_model_name: Name to register the model under in the MLflow registry
        conda_env: Conda environment specification
        signature: Model signature (input/output schema)
        input_example: Example input for the model
    """
    if not mlflow.active_run():
        logger.warning("No active MLflow run. Model will not be logged.")
        return

    # Create input example if not provided
    if input_example is None and hasattr(model, "feature_importances_"):
        # For models that have feature importances, create a random example
        # with correct shape
        n_features = len(model.feature_importances_)
        input_example = pd.DataFrame(np.random.random((1, n_features)))
    elif input_example is None and hasattr(model, "n_features_in_"):
        # For scikit-learn models that store number of features
        n_features = model.n_features_in_
        input_example = pd.DataFrame(np.random.random((1, n_features)))

    # Infer signature if not provided and we have an input example
    if signature is None and input_example is not None:
        try:
            # Predict on the input example to get output shape
            example_output = model.predict(input_example)
            # Infer signature from input and output
            signature = infer_signature(input_example, example_output)
        except Exception as e:
            logger.warning(f"Could not infer model signature: {e}")

    # Log the model
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path=artifact_path,
        registered_model_name=registered_model_name,
        conda_env=conda_env,
        signature=signature,
        input_example=input_example,
    )

    logger.info(f"Logged model to MLflow (artifact_path: {artifact_path})")

    if registered_model_name:
        logger.info(f"Registered model as '{registered_model_name}'")


def log_figure(
        fig: plt.Figure,
        artifact_path: str,
        close_figure: bool = True) -> None:
    """
    Log a matplotlib figure to MLflow.

    Args:
        fig: Figure to log
        artifact_path: Path within the run artifacts to store the figure
        close_figure: Whether to close the figure after logging
    """
    if not mlflow.active_run():
        logger.warning("No active MLflow run. Figure will not be logged.")
        return

    # Log the figure
    mlflow.log_figure(fig, artifact_path)
    logger.info(f"Logged figure to MLflow (artifact_path: {artifact_path})")

    if close_figure:
        plt.close(fig)


def log_artifact(local_path: str, artifact_path: Optional[str] = None) -> None:
    """
    Log a local file as an artifact to MLflow.

    Args:
        local_path: Path to the local file
        artifact_path: Path within the run artifacts to store the file
    """
    if not mlflow.active_run():
        logger.warning("No active MLflow run. Artifact will not be logged.")
        return

    # Log the artifact
    mlflow.log_artifact(local_path, artifact_path)
    logger.info(f"Logged artifact from {local_path} to MLflow")


def end_run() -> None:
    """End the current MLflow run."""
    if mlflow.active_run():
        mlflow.end_run()
        logger.info("Ended MLflow run")
    else:
        logger.warning("No active MLflow run to end.")


def get_tracking_uri() -> str:
    """Get the current MLflow tracking URI."""
    return mlflow.get_tracking_uri()


def create_remote_tracking_uri(
    storage_type: str = "s3",
    bucket_name: str = "mlflow",
    endpoint_url: Optional[str] = None,
) -> str:
    """
    Create a remote tracking URI for MLflow.

    Args:
        storage_type: Type of storage ('s3', 'azure', 'gcs')
        bucket_name: Name of the bucket
        endpoint_url: Custom endpoint URL (for MinIO or other S3-compatible services)

    Returns:
        Tracking URI string
    """
    if storage_type == "s3":
        if endpoint_url:
            tracking_uri = f"{
                storage_type}://{bucket_name}?endpoint_url={endpoint_url}"
        else:
            tracking_uri = f"{storage_type}://{bucket_name}"
    else:
        tracking_uri = f"{storage_type}://{bucket_name}"

    return tracking_uri
