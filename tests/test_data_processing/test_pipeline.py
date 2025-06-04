"""
Tests for data processing pipeline module.
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nirs_tomato.data_processing.feature_selection import PLSVIPSelector
from nirs_tomato.data_processing.transformers import (
    SavGolTransformer,
    SNVTransformer,
)


def test_create_basic_pipeline():
    """Test creating a simple preprocessing pipeline."""
    # Create a pipeline with SNV and SavGol preprocessing
    pipeline = Pipeline(
        [
            ("snv", SNVTransformer()),
            (
                "savgol",
                SavGolTransformer(window_length=11, polyorder=2, deriv=1),
            ),
            ("scaler", StandardScaler()),
        ]
    )

    # Check that pipeline has the correct steps
    assert len(pipeline.steps) == 3
    assert pipeline.steps[0][0] == "snv"
    assert pipeline.steps[1][0] == "savgol"
    assert pipeline.steps[2][0] == "scaler"

    # Check that the pipeline can be fit and transform data
    X = np.random.rand(10, 20)
    X_df = pd.DataFrame(X, columns=[f"wl_{i}" for i in range(20)])

    pipeline.fit(X_df)
    result = pipeline.transform(X_df)

    # Check output shape
    assert result.shape == X_df.shape


def test_pipeline_with_feature_selection(
    sample_spectra_data, sample_target_data
):
    """Test pipeline with feature selection component."""
    # Create pipeline with preprocessing and feature selection
    pipeline = Pipeline(
        [
            ("snv", SNVTransformer()),
            (
                "savgol",
                SavGolTransformer(window_length=11, polyorder=2, deriv=1),
            ),
            (
                "feature_selector",
                PLSVIPSelector(n_components=3, n_features_to_select=10),
            ),
        ]
    )

    # Fit pipeline with X and y
    pipeline.fit(sample_spectra_data, sample_target_data)

    # Transform data
    result = pipeline.transform(sample_spectra_data)

    # Check output shape - should have 10 selected features
    assert result.shape == (sample_spectra_data.shape[0], 10)

    # Check that feature selector step has vip_scores_ attribute
    feature_selector = pipeline.named_steps["feature_selector"]
    assert hasattr(feature_selector, "vip_scores_")
    assert len(feature_selector.vip_scores_) == sample_spectra_data.shape[1]
