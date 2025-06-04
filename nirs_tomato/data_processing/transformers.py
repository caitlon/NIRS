"""
NIR Preprocessing Transformers Module

This module contains preprocessing transformers for NIR spectral data analysis.
All transformers follow scikit-learn's BaseEstimator and TransformerMixin interfaces.
"""  

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import signal
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .constants import (
    DEFAULT_PCA_COMPONENTS,
    DEFAULT_SAVGOL_DERIV,
    DEFAULT_SAVGOL_POLYORDER,
    DEFAULT_SAVGOL_WINDOW,
)


class SNVTransformer(BaseEstimator, TransformerMixin):
    """
    Standard Normal Variate transformation for spectral data.

    For each spectrum (row), this subtracts the mean and divides by the standard deviation.
    This helps reduce scatter effects and baseline variations.
    """  

    def __init__(self, spectral_cols: Optional[List[str]] = None):
        self.spectral_cols = spectral_cols

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        # Handle both DataFrame and array inputs
        if isinstance(X, pd.DataFrame):
            if self.spectral_cols:
                # Process only specified columns
                X_copy = X.copy()
                spectral_data = X[self.spectral_cols].values
                spectral_data_snv = self._apply_snv(spectral_data)
                X_copy[self.spectral_cols] = spectral_data_snv
                return X_copy
            else:
                # Process entire DataFrame
                X_array = X.values
                X_snv = self._apply_snv(X_array)
                return pd.DataFrame(X_snv, index=X.index, columns=X.columns)
        else:
            # Process NumPy array
            return self._apply_snv(X)

    def _apply_snv(self, X_array: np.ndarray) -> np.ndarray:
        """Apply SNV to a NumPy array."""
        X_snv = np.zeros_like(X_array)
        for i in range(X_array.shape[0]):
            X_snv[i, :] = (X_array[i, :] - np.mean(X_array[i, :])) / np.std(
                X_array[i, :]
            )
        return X_snv


class SavGolTransformer(BaseEstimator, TransformerMixin):
    """
    Savitzky-Golay filter for spectral derivatives.

    Applies a polynomial smoothing filter and calculates derivatives.
    Derivatives help enhance spectral features and reduce baseline effects.
    """

    def __init__(
        self,
        window_length: int = DEFAULT_SAVGOL_WINDOW,
        polyorder: int = DEFAULT_SAVGOL_POLYORDER,
        deriv: int = DEFAULT_SAVGOL_DERIV,
        spectral_cols: Optional[List[str]] = None,
    ):
        self.window_length = window_length
        self.polyorder = polyorder
        self.deriv = deriv
        self.spectral_cols = spectral_cols

    def fit(self, X, y=None):
        return self

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        # Handle both DataFrame and array inputs
        if isinstance(X, pd.DataFrame):
            if self.spectral_cols:
                # Process only specified columns
                X_copy = X.copy()
                spectral_data = X[self.spectral_cols].values
                spectral_data_sg = self._apply_savgol(spectral_data)
                X_copy[self.spectral_cols] = spectral_data_sg
                return X_copy
            else:
                # Process entire DataFrame
                X_array = X.values
                X_sg = self._apply_savgol(X_array)
                return pd.DataFrame(X_sg, index=X.index, columns=X.columns)
        else:
            # Process NumPy array
            return self._apply_savgol(X)

    def _apply_savgol(self, X_array: np.ndarray) -> np.ndarray:
        """Apply Savitzky-Golay filter to a NumPy array."""
        X_sg = np.zeros_like(X_array)
        for i in range(X_array.shape[0]):
            X_sg[i, :] = signal.savgol_filter(
                X_array[i, :],
                window_length=self.window_length,
                polyorder=self.polyorder,
                deriv=self.deriv,
            )
        return X_sg


class MSCTransformer(BaseEstimator, TransformerMixin):
    """
    Multiplicative Scatter Correction for spectral data.

    Corrects for scatter effects by regressing each spectrum against a reference
    spectrum (usually the mean) and then correcting using the slope and intercept.
    """  

    def __init__(self, spectral_cols: Optional[List[str]] = None):
        self.mean_spectrum = None
        self.spectral_cols = spectral_cols

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        # Calculate mean spectrum as reference
        if isinstance(X, pd.DataFrame):
            if self.spectral_cols:
                spectral_data = X[self.spectral_cols].values
                self.mean_spectrum = np.mean(spectral_data, axis=0)
            else:
                self.mean_spectrum = np.mean(X.values, axis=0)
        else:
            self.mean_spectrum = np.mean(X, axis=0)
        return self

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        if self.mean_spectrum is None:
            raise ValueError("MSCTransformer must be fitted before transform")

        # Handle both DataFrame and array inputs
        if isinstance(X, pd.DataFrame):
            if self.spectral_cols:
                # Process only specified columns
                X_copy = X.copy()
                spectral_data = X[self.spectral_cols].values
                spectral_data_msc = self._apply_msc(spectral_data)
                X_copy[self.spectral_cols] = spectral_data_msc
                return X_copy
            else:
                # Process entire DataFrame
                X_array = X.values
                X_msc = self._apply_msc(X_array)
                return pd.DataFrame(X_msc, index=X.index, columns=X.columns)
        else:
            # Process NumPy array
            return self._apply_msc(X)

    def _apply_msc(self, X_array: np.ndarray) -> np.ndarray:
        """Apply MSC to a NumPy array."""
        X_msc = np.zeros_like(X_array)
        for i in range(X_array.shape[0]):
            # Linear regression of spectrum against reference
            slope, intercept = np.polyfit(self.mean_spectrum, X_array[i, :], 1)
            # Apply correction
            X_msc[i, :] = (X_array[i, :] - intercept) / slope
        return X_msc


class OutlierDetector(BaseEstimator, TransformerMixin):
    """
    Detects outliers in spectral data using different methods.
    """

    def __init__(
        self,
        method: str = "zscore",
        threshold: float = 3.0,
        remove_outliers: bool = False,
        spectral_cols: Optional[List[str]] = None,
    ):
        self.method = method
        self.threshold = threshold
        self.remove_outliers = remove_outliers
        self.spectral_cols = spectral_cols
        self.outlier_mask_ = None

    def fit(self, X: pd.DataFrame, y=None):
        from .utils import detect_outliers_pca, detect_outliers_zscore

        # Use spectral columns if specified, otherwise use all columns
        spectral_data = (
            X[self.spectral_cols].values if self.spectral_cols else X.values
        )

        if self.method == "zscore":
            self.outlier_mask_ = detect_outliers_zscore(
                spectral_data, threshold=self.threshold
            )
        elif self.method == "pca":
            self.outlier_mask_ = detect_outliers_pca(
                spectral_data, threshold=self.threshold
            )
        elif self.method == "both":
            z_outliers = detect_outliers_zscore(
                spectral_data, threshold=self.threshold
            )
            pca_outliers = detect_outliers_pca(
                spectral_data, threshold=self.threshold
            )
            self.outlier_mask_ = z_outliers | pca_outliers
        else:
            raise ValueError(
                f"Unknown outlier detection method: {self.method}"
            )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        result_df = X.copy()

        if self.outlier_mask_ is not None:
            # Add outlier flag to dataframe
            result_df["is_outlier"] = self.outlier_mask_

            # Remove outliers if requested
            if self.remove_outliers:
                result_df = result_df[~self.outlier_mask_].copy()

        return result_df


class PCATransformer(BaseEstimator, TransformerMixin):
    """
    Applies PCA dimensionality reduction to spectral data.
    """

    def __init__(
        self,
        n_components: int = DEFAULT_PCA_COMPONENTS,
        standardize: bool = True,
        spectral_cols: Optional[List[str]] = None,
    ):
        self.n_components = n_components
        self.standardize = standardize
        self.spectral_cols = spectral_cols
        self.pca = None
        self.scaler = StandardScaler() if standardize else None

    def fit(self, X: pd.DataFrame, y=None):
        # Use spectral columns if specified, otherwise use all columns
        if self.spectral_cols:
            data = X[self.spectral_cols].values
        else:
            data = X.values

        # Apply standardization if requested
        if self.standardize:
            data = self.scaler.fit_transform(data)

        # Create and fit PCA
        n_components = min(self.n_components, data.shape[0], data.shape[1])
        self.pca = PCA(n_components=n_components)
        self.pca.fit(data)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.pca is None:
            raise ValueError("PCATransformer must be fitted before transform")

        # Use spectral columns if specified, otherwise use all columns
        if self.spectral_cols:
            data = X[self.spectral_cols].values
        else:
            data = X.values

        # Apply standardization if requested
        if self.standardize:
            data = self.scaler.transform(data)

        # Apply PCA transformation
        pca_result = self.pca.transform(data)

        # Create DataFrame with PCA components
        component_names = [f"PC{i + 1}" for i in range(pca_result.shape[1])]
        df_pca = pd.DataFrame(
            pca_result, index=X.index, columns=component_names
        )

        # Combine with non-spectral columns if specified
        if self.spectral_cols:
            non_spectral_cols = [
                col for col in X.columns if col not in self.spectral_cols
            ]
            if non_spectral_cols:
                return pd.concat([df_pca, X[non_spectral_cols]], axis=1)

        return df_pca


def create_preprocessing_pipelines() -> Dict[str, Any]:
    """
    Create preprocessing pipelines for NIR spectral data.

    Returns:
        Dictionary of preprocessing pipelines for different methods.
    """
    from sklearn.pipeline import Pipeline

    # Raw data with scaling
    raw_pipeline = Pipeline([("scaler", StandardScaler())])

    # SNV preprocessing
    snv_pipeline = Pipeline([("snv", SNVTransformer())])

    # First derivative preprocessing
    sg1_pipeline = Pipeline(
        [("sg1", SavGolTransformer(window_length=15, polyorder=2, deriv=1))]
    )

    # Second derivative preprocessing
    sg2_pipeline = Pipeline(
        [("sg2", SavGolTransformer(window_length=15, polyorder=2, deriv=2))]
    )

    # MSC preprocessing
    msc_pipeline = Pipeline([("msc", MSCTransformer())])

    # SNV followed by PCA
    snv_pca_pipeline = Pipeline(
        [("snv", SNVTransformer()), ("pca", PCATransformer(n_components=10))]
    )

    # First derivative followed by PCA
    sg1_pca_pipeline = Pipeline(
        [
            ("sg1", SavGolTransformer(window_length=15, polyorder=2, deriv=1)),
            ("pca", PCATransformer(n_components=10)),
        ]
    )

    # Store all pipelines in a dictionary
    preprocessing_pipelines = {
        "raw": raw_pipeline,
        "snv": snv_pipeline,
        "sg1": sg1_pipeline,
        "sg2": sg2_pipeline,
        "msc": msc_pipeline,
        "snv_pca": snv_pca_pipeline,
        "sg1_pca": sg1_pca_pipeline,
    }

    return preprocessing_pipelines
