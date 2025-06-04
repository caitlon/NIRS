"""
Tests for preprocessing transformers in data_processing module.
"""

import numpy as np

from nirs_tomato.data_processing.transformers import (
    MSCTransformer,
    PCATransformer,
    SavGolTransformer,
    SNVTransformer,
)


def test_snv_transformer(sample_spectra_data):
    """Test SNV transformation on sample data."""
    # Setup
    transformer = SNVTransformer()

    # Transform data
    transformed_data = transformer.fit_transform(sample_spectra_data)

    # Check that output has same shape as input
    assert transformed_data.shape == sample_spectra_data.shape

    # Check that each row has mean close to 0 and std close to 1
    for i in range(transformed_data.shape[0]):
        row = transformed_data.iloc[i].values
        assert np.isclose(np.mean(row), 0, atol=1e-10)
        assert np.isclose(np.std(row), 1, atol=1e-10)


def test_savgol_transformer(sample_spectra_data):
    """Test Savitzky-Golay filter transformation."""
    # Setup - use 1st derivative
    transformer = SavGolTransformer(window_length=11, polyorder=2, deriv=1)

    # Transform data
    transformed_data = transformer.fit_transform(sample_spectra_data)

    # Check that output has same shape as input
    assert transformed_data.shape == sample_spectra_data.shape

    # For derivative, sum should be close to zero
    for i in range(transformed_data.shape[0]):
        row = transformed_data.iloc[i].values
        # The sum of derivatives should be smaller than original data
        assert abs(np.sum(row)) < abs(
            np.sum(sample_spectra_data.iloc[i].values)
        )


def test_msc_transformer(sample_spectra_data):
    """Test Multiplicative Scatter Correction transformation."""
    # Setup
    transformer = MSCTransformer()

    # Fit and transform data
    transformer.fit(sample_spectra_data)
    transformed_data = transformer.transform(sample_spectra_data)

    # Check that output has same shape as input
    assert transformed_data.shape == sample_spectra_data.shape

    # Check that mean spectrum exists after fitting
    assert transformer.mean_spectrum is not None
    assert len(transformer.mean_spectrum) == sample_spectra_data.shape[1]


def test_pca_transformer(sample_spectra_data):
    """Test PCA transformation."""
    # Setup - reduce to 5 components
    n_components = 5
    transformer = PCATransformer(n_components=n_components)

    # Fit and transform data
    transformer.fit(sample_spectra_data)
    transformed_data = transformer.transform(sample_spectra_data)

    # Check output shape - should have n_components columns
    assert transformed_data.shape[0] == sample_spectra_data.shape[0]
    assert transformed_data.shape[1] == n_components

    # Check column names
    for i in range(n_components):
        assert f"PC{i + 1}" in transformed_data.columns


def test_transformers_with_subset_columns(sample_spectra_data):
    """Test transformers when only a subset of columns should be transformed."""  
    # Add a non-spectral column to the data
    data_with_extra = sample_spectra_data.copy()
    data_with_extra["metadata"] = range(len(data_with_extra))

    # Get spectral columns
    spectral_cols = [
        col for col in data_with_extra.columns if col.startswith("wl_")
    ]

    # Test SNV with subset of columns
    snv = SNVTransformer(spectral_cols=spectral_cols)
    snv_result = snv.fit_transform(data_with_extra)

    # Check metadata column is unchanged
    assert np.array_equal(
        snv_result["metadata"].values, data_with_extra["metadata"].values
    )

    # Check spectral columns are transformed
    for i in range(len(data_with_extra)):
        row = snv_result.loc[i, spectral_cols].values
        assert np.isclose(np.mean(row), 0, atol=1e-10)
        assert np.isclose(np.std(row), 1, atol=1e-10)
