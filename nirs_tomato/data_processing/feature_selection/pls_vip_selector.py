"""
Variable Importance in Projection (VIP) for PLS Feature Selection in NIR Spectroscopy

This module implements the VIP method for selecting the most informative
wavelengths from NIR spectral data based on their importance in PLS models.
"""

from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSRegression


class PLSVIPSelector(BaseEstimator, TransformerMixin):
    """
    Variable Importance in Projection (VIP) for PLS model feature selection.

    VIP scores estimate the importance of each variable in the PLS projection.
    This selector keeps the variables with VIP scores above a threshold or
    the top k variables by VIP score.

    Parameters:
    -----------
    n_components : int, optional (default=10)
        Number of PLS components to use.
    n_features_to_select : int, optional (default=None)
        Number of features to select. If None, uses threshold.
    threshold : float, optional (default=1.0)
        VIP score threshold for feature selection.
    use_n_features : bool, optional (default=True)
        If True, selects n_features_to_select. If False, uses threshold.
    wavelengths : list or array, optional (default=None)
        Actual wavelength values (for visualization and interpretation).
    """

    def __init__(
        self,
        n_components: int = 10,
        n_features_to_select: Optional[int] = None,
        threshold: float = 1.0,
        use_n_features: bool = True,
        wavelengths: Optional[List[float]] = None,
    ):
        self.n_components = n_components
        self.n_features_to_select = n_features_to_select
        self.threshold = threshold
        self.use_n_features = use_n_features
        self.wavelengths = wavelengths
        self.selected_features_mask_ = None
        self.selected_features_indices_ = None
        self.vip_scores_ = None
        self.pls_model_ = None

    def fit(self, X: Union[pd.DataFrame, np.ndarray],
            y: Union[pd.Series, np.ndarray]) -> "PLSVIPSelector":
        """
        Compute VIP scores and select features.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.

        Returns:
        --------
        self : object
            Returns self.
        """
        # Convert to numpy arrays if pandas objects
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else y

        n_samples, n_features = X_array.shape

        # Initialize PLS model
        n_comp = min(self.n_components, n_samples, n_features)
        self.pls_model_ = PLSRegression(n_components=n_comp)

        # Fit PLS model
        self.pls_model_.fit(X_array, y_array)

        # Calculate VIP scores
        self.vip_scores_ = self._calculate_vip_scores(
            self.pls_model_, X_array, n_features
        )

        # Select features based on VIP scores
        if self.use_n_features and self.n_features_to_select is not None:
            n_to_select = min(self.n_features_to_select, n_features)
            vip_threshold = np.sort(self.vip_scores_)[-n_to_select]
        else:
            vip_threshold = self.threshold

        self.selected_features_mask_ = self.vip_scores_ >= vip_threshold
        self.selected_features_indices_ = np.where(
            self.selected_features_mask_)[0]

        return self

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Reduce X to the selected features.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns:
        --------
        X_r : array-like of shape (n_samples, n_selected_features)
            The input samples with only the selected features.
        """
        if self.selected_features_mask_ is None:
            raise ValueError("PLSVIPSelector has not been fitted yet.")

        if isinstance(X, pd.DataFrame):
            return X.iloc[:, self.selected_features_mask_]
        else:
            return X[:, self.selected_features_mask_]

    def _calculate_vip_scores(
        self, pls_model: PLSRegression, X: np.ndarray, n_features: int
    ) -> np.ndarray:
        """
        Calculate VIP scores for a fitted PLS model.

        Parameters:
        -----------
        pls_model : PLSRegression
            Fitted PLS model.
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        n_features : int
            Number of features.

        Returns:
        --------
        vip_scores : array-like of shape (n_features,)
            VIP scores for each feature.
        """
        # Get PLS components (T scores)
        t = pls_model.x_scores_

        # Get PLS weights
        w = pls_model.x_weights_

        # Get PLS loadings
        p = pls_model.x_loadings_

        # Number of components
        n_comp = pls_model.n_components

        # Explained variance by each component
        q = np.var(t, axis=0)

        # Sum of squares of all T scores
        t_ss = np.sum(t**2, axis=0)

        # Initialize VIP scores
        vip_scores = np.zeros(n_features)

        # Calculate VIP scores
        for j in range(n_features):
            weighted_sum = 0
            for i in range(n_comp):
                weighted_sum += q[i] * \
                    (w[j, i] / np.sqrt(np.sum(w[:, i] ** 2))) ** 2

            vip_scores[j] = np.sqrt(n_features * weighted_sum / np.sum(q))

        return vip_scores

    def plot_vip_scores(self, figsize: Tuple[int, int] = (
            12, 6), save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot VIP scores for all features.

        Parameters:
        -----------
        figsize : tuple, optional (default=(12, 6))
            Figure size.
        save_path : str, optional (default=None)
            Path to save the figure.

        Returns:
        --------
        fig : matplotlib.figure.Figure
            The matplotlib figure object.
        """
        if self.vip_scores_ is None:
            raise ValueError("PLSVIPSelector has not been fitted yet.")

        fig, ax = plt.subplots(figsize=figsize)

        if self.wavelengths is not None:
            x_values = self.wavelengths
        else:
            x_values = np.arange(len(self.vip_scores_))

        # Plot VIP scores
        ax.plot(x_values, self.vip_scores_, "o-", markersize=4)

        # Add horizontal line at threshold
        if not self.use_n_features:
            ax.axhline(
                y=self.threshold,
                color="r",
                linestyle="--",
                label=f"Threshold = {self.threshold}",
            )
        else:
            vip_threshold = np.sort(
                self.vip_scores_)[-self.n_features_to_select]
            ax.axhline(
                y=vip_threshold,
                color="r",
                linestyle="--",
                label=f"Threshold (top {self.n_features_to_select})",
            )

        # Highlight selected wavelengths
        if self.wavelengths is not None:
            selected_wavelengths = self.wavelengths[self.selected_features_mask_]
        else:
            selected_wavelengths = x_values[self.selected_features_mask_]

        selected_vips = self.vip_scores_[self.selected_features_mask_]
        ax.plot(
            selected_wavelengths,
            selected_vips,
            "ro",
            markersize=6,
            label="Selected")

        ax.set_xlabel(
            "Wavelength (nm)" if self.wavelengths is not None else "Feature Index")
        ax.set_ylabel("VIP Score")
        ax.set_title(
            f"PLS VIP Scores (Selected: {np.sum(self.selected_features_mask_)} features)")
        ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig
