"""
Pytest configuration file.

This file configures pytest for the NIRS Tomato project.
"""

import logging
import sys
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from nirs_tomato.config import (
    DataConfig,
    ExperimentConfig,
    FeatureSelectionConfig,
    FeatureSelectionMethod,
    ModelConfig,
    ModelType,
    TransformType,
)

# Import fixtures so they're available to all tests


# Configure logging for tests
@pytest.fixture(autouse=True)
def configure_logging():
    """Set up logging for tests."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )

    # Suppress excessive logging from libraries
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    # Reset handlers to avoid duplicate logs
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        handler.close()
        root_logger.removeHandler(handler)

    yield


@pytest.fixture
def sample_spectra_data():
    """
    Create sample NIR spectral data for testing.

    Returns:
        pd.DataFrame: DataFrame with spectral data (X), only spectral columns without metadata
    """
    # Create synthetic wavelengths (e.g., 900-1700 nm with 5nm intervals)
    wavelengths = np.arange(900, 1705, 5)
    n_wavelengths = len(wavelengths)

    # Create sample column names for the spectra
    column_names = [f"wl_{w}" for w in wavelengths]

    # Create 30 synthetic spectra with random variations
    n_samples = 30
    np.random.seed(42)  # For reproducibility

    # Base spectrum shape (similar to NIR spectra of organic materials)
    base_spectrum = np.sin(np.linspace(0, 3 * np.pi, n_wavelengths)) + 2

    # Add random variations to create multiple samples
    spectra = np.array(
        [
            base_spectrum + 0.2 * np.random.randn(n_wavelengths)
            for _ in range(n_samples)
        ]
    )

    # Create DataFrame with spectra only - no metadata columns
    X = pd.DataFrame(spectra, columns=column_names)

    return X


@pytest.fixture
def sample_spectra_data_with_metadata():
    """
    Create sample NIR spectral data with metadata for testing.

    Returns:
        pd.DataFrame: DataFrame with spectral data and metadata columns
    """
    # Get the base spectral data
    X = sample_spectra_data()

    # Add some non-spectral columns
    X["sample_id"] = [f"S{i:03d}" for i in range(len(X))]
    X["batch"] = np.random.choice(["A", "B", "C"], size=len(X))

    return X


@pytest.fixture
def sample_target_data():
    """
    Create sample target values for testing.

    Returns:
        pd.Series: Series with target values (y)
    """
    # Create 30 synthetic spectra with random variations
    n_samples = 30
    np.random.seed(42)  # For reproducibility

    # Create synthetic target values (e.g., Brix values for tomatoes)
    y = pd.Series(3.5 + 0.5 * np.random.randn(n_samples), name="brix")

    return y


@pytest.fixture
def train_test_split_indices():
    """
    Create indices for train/test/validation split.

    Returns:
        dict: Dictionary with train, test, and validation indices
    """
    np.random.seed(42)

    # Total number of samples
    n_samples = 30

    # Generate indices and shuffle them
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    # Split indices into train (70%), validation (15%), and test (15%)
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)

    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]

    return {"train": train_indices, "val": val_indices, "test": test_indices}


@pytest.fixture
def mock_mlflow():
    """
    Create a mock MLflow object for testing tracking functionality.

    Returns:
        MagicMock: Mock MLflow object with commonly used methods
    """
    mlflow_mock = MagicMock()
    mlflow_mock.start_run.return_value.__enter__ = lambda x: None
    mlflow_mock.start_run.return_value.__exit__ = lambda x, y, z, w: None
    mlflow_mock.set_experiment.return_value = None
    mlflow_mock.log_params.return_value = None
    mlflow_mock.log_metrics.return_value = None
    mlflow_mock.log_artifact.return_value = None
    mlflow_mock.sklearn.log_model.return_value = None
    return mlflow_mock


@pytest.fixture
def sample_config():
    """
    Create a sample experiment configuration for testing.

    Returns:
        ExperimentConfig: Sample configuration object
    """
    return ExperimentConfig(
        name="test_experiment",
        description="Test experiment configuration",
        data=DataConfig(
            data_path="data/test_data.csv",
            target_column="Brix",
            transform=TransformType.SNV,
            savgol={
                "enabled": True,
                "window_length": 15,
                "polyorder": 2,
                "deriv": 0,
            },
            remove_outliers=False,
        ),
        feature_selection=FeatureSelectionConfig(
            method=FeatureSelectionMethod.VIP, n_features=10
        ),
        model=ModelConfig(
            model_type=ModelType.PLS,
            pls_n_components=5,
            test_size=0.2,
            random_state=42,
        ),
        mlflow={"enabled": True, "experiment_name": "test_run"},
    )
