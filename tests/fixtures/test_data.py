"""
Test data fixtures for NIRS Tomato project.
"""

import numpy as np
import pandas as pd
import pytest
from typing import Tuple, Dict, Any

@pytest.fixture
def sample_spectra_data() -> pd.DataFrame:
    """
    Generate a small sample of synthetic NIR spectra data.
    
    Returns:
        DataFrame with wavelengths as columns and samples as rows
    """
    # Generate 20 synthetic spectra with 100 wavelength points
    np.random.seed(42)  # For reproducibility
    n_samples = 20
    n_wavelengths = 100
    
    # Generate wavelengths from 900 to 1700 nm
    wavelengths = np.linspace(900, 1700, n_wavelengths)
    
    # Create synthetic spectra with random noise
    # Base spectra shape (gaussian-like)
    base_spectrum = np.exp(-0.5 * ((wavelengths - 1300) / 200) ** 2)
    
    # Add some random variations to each spectrum
    spectra = np.array([
        base_spectrum + 0.1 * np.random.randn(n_wavelengths) 
        for _ in range(n_samples)
    ])
    
    # Convert to DataFrame with wavelengths as column names
    column_names = [f"wl_{int(wl)}" for wl in wavelengths]
    df = pd.DataFrame(spectra, columns=column_names)
    
    return df

@pytest.fixture
def sample_target_data() -> pd.Series:
    """
    Generate synthetic target values (Brix) for the sample spectra.
    
    Returns:
        Series with Brix values
    """
    np.random.seed(42)  # For reproducibility
    n_samples = 20
    
    # Generate Brix values between 4 and 8 with some correlation to spectra
    brix_values = 6 + 2 * np.random.randn(n_samples)
    
    # Ensure values are within reasonable range
    brix_values = np.clip(brix_values, 4, 8)
    
    return pd.Series(brix_values, name="brix")

@pytest.fixture
def train_test_split_indices() -> Dict[str, np.ndarray]:
    """
    Generate sample indices for train/test splitting.
    
    Returns:
        Dictionary with train and test indices
    """
    np.random.seed(42)
    n_samples = 20
    
    # Create random permutation of indices
    indices = np.random.permutation(n_samples)
    
    # Split into 70% train, 30% test
    train_size = int(0.7 * n_samples)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    return {
        "train": train_indices,
        "test": test_indices
    }

@pytest.fixture
def sample_model_params() -> Dict[str, Dict[str, Any]]:
    """
    Sample model parameters for testing.
    
    Returns:
        Dictionary with model parameters
    """
    return {
        "rf": {
            "n_estimators": 10,
            "max_depth": 3,
            "random_state": 42
        },
        "pls": {
            "n_components": 3,
            "scale": False
        },
        "xgb": {
            "n_estimators": 10,
            "max_depth": 2,
            "learning_rate": 0.1,
            "random_state": 42
        }
    } 