"""
Tests for feature selection methods.
"""

import numpy as np
import pandas as pd

from nirs_tomato.data_processing.feature_selection import PLSVIPSelector


def test_pls_vip_selector():
    """Test that PLS VIP feature selection works correctly."""
    # Generate simple test data where some features are more important
    np.random.seed(42)
    n_samples = 20
    n_features = 50

    # Create features where only the first 10 are meaningful
    X = np.random.rand(n_samples, n_features)
    y = 2 * X[:, :10].mean(axis=1) + 0.1 * np.random.randn(n_samples)

    # Convert to DataFrame
    feature_names = [f"feature_{i}" for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name="target")

    # Create and fit the selector
    n_features_to_select = 15
    selector = PLSVIPSelector(
        n_components=5, n_features_to_select=n_features_to_select
    )
    selector.fit(X_df, y_series)

    # Check that attributes are set correctly
    assert hasattr(selector, "vip_scores_")
    assert len(selector.vip_scores_) == n_features
    assert hasattr(selector, "selected_features_mask_")
    assert np.sum(selector.selected_features_mask_) == n_features_to_select

    # Transform the data
    X_selected = selector.transform(X_df)

    # Check that output has correct shape
    assert X_selected.shape == (n_samples, n_features_to_select)

    # Check that the important features are selected
    # We should capture some of the first 10 features
    important_features_selected = sum(
        1 for i in range(10) if f"feature_{i}" in X_selected.columns
    )
    assert (
        important_features_selected >= 3
    )  # At least 3 of the important features should be selected


def test_pls_vip_selector_with_real_data(
    sample_spectra_data, sample_target_data
):
    """Test PLS VIP selector with our sample spectral data."""
    # Setup
    n_features_to_select = 20
    selector = PLSVIPSelector(
        n_components=5, n_features_to_select=n_features_to_select
    )

    # Fit and transform
    selector.fit(sample_spectra_data, sample_target_data)
    X_selected = selector.transform(sample_spectra_data)

    # Basic checks
    assert X_selected.shape == (
        sample_spectra_data.shape[0],
        n_features_to_select,
    )
    assert hasattr(selector, "vip_scores_")
    assert len(selector.vip_scores_) == sample_spectra_data.shape[1]

    # Check that scores and mask are consistent
    top_indices = np.argsort(selector.vip_scores_)[-n_features_to_select:]
    selected_indices = np.where(selector.selected_features_mask_)[0]
    assert set(top_indices) == set(selected_indices)
