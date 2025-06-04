"""
Tests for CARS (Competitive Adaptive Reweighted Sampling) feature selector.
"""

import numpy as np
import pandas as pd

from nirs_tomato.data_processing.feature_selection.cars_selector import (
    CARSSelector,
)


def test_cars_selector_initialization():
    """Test CARS selector initialization with various parameters."""
    # Default initialization
    selector = CARSSelector()
    assert selector.n_pls_components == 10
    assert selector.n_sampling_runs == 50
    assert selector.exponential_decay == 0.95

    # Custom initialization
    selector = CARSSelector(
        n_pls_components=5,
        n_sampling_runs=30,
        exponential_decay=0.9,
        n_features_to_select=10,
        cv=3,
        random_state=42,
    )
    assert selector.n_pls_components == 5
    assert selector.n_sampling_runs == 30
    assert selector.exponential_decay == 0.9
    assert selector.n_features_to_select == 10
    assert selector.cv == 3
    assert selector.random_state == 42


def test_cars_selector_fit():
    """Test CARS selector fitting process."""
    # Create synthetic data
    n_samples, n_features = 20, 30
    X = np.random.rand(n_samples, n_features)
    y = np.random.rand(n_samples)

    # Create feature column names
    feature_names = [f"feat_{i}" for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)

    # Fit selector
    selector = CARSSelector(
        n_pls_components=5,
        n_sampling_runs=3,  # Small number for test speed
        n_features_to_select=5,
        random_state=42,
    )
    selector.fit(X_df, y)

    # Check if attributes are set
    assert hasattr(selector, "selected_features_indices_")
    assert hasattr(selector, "selected_features_mask_")
    assert hasattr(selector, "weights_")
    assert hasattr(selector, "rmse_history_")
    assert (
        selector.selected_features_mask_.sum() >= 1
    )  # At least one feature selected
    assert len(selector.selected_features_indices_) >= 1


def test_cars_selector_transform_array():
    """Test CARS selector transform with numpy array."""
    # Create synthetic data
    n_samples, n_features = 20, 30
    X = np.random.rand(n_samples, n_features)
    y = np.random.rand(n_samples)

    # Fit selector
    selector = CARSSelector(
        n_pls_components=5,
        n_sampling_runs=3,  # Small number for test speed
        n_features_to_select=5,
        random_state=42,
    )
    selector.fit(X, y)

    # Transform
    X_selected = selector.transform(X)

    # Check output
    assert X_selected.shape[0] == n_samples
    assert X_selected.shape[1] == selector.selected_features_mask_.sum()


def test_cars_selector_transform_dataframe():
    """Test CARS selector transform with pandas DataFrame."""
    # Create synthetic data
    n_samples, n_features = 20, 30
    X = np.random.rand(n_samples, n_features)
    y = np.random.rand(n_samples)

    # Create feature column names
    feature_names = [f"feat_{i}" for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)

    # Fit selector
    selector = CARSSelector(
        n_pls_components=5,
        n_sampling_runs=3,  # Small number for test speed
        n_features_to_select=5,
        random_state=42,
    )
    selector.fit(X_df, y)

    # Transform DataFrame
    X_selected = selector.transform(X_df)

    # Check output
    assert isinstance(X_selected, pd.DataFrame)
    assert X_selected.shape[0] == n_samples
    assert X_selected.shape[1] == selector.selected_features_mask_.sum()


def test_cars_selector_fit_transform():
    """Test CARS selector fit_transform method."""
    # Create synthetic data
    n_samples, n_features = 20, 30
    X = np.random.rand(n_samples, n_features)
    y = np.random.rand(n_samples)

    # Fit and transform
    selector = CARSSelector(
        n_pls_components=5,
        n_sampling_runs=3,  # Small number for test speed
        n_features_to_select=5,
        random_state=42,
    )
    X_selected = selector.fit_transform(X, y)

    # Check output
    assert X_selected.shape[0] == n_samples
    assert X_selected.shape[1] <= n_features  # Should have fewer features
    assert len(selector.selected_features_indices_) <= n_features
