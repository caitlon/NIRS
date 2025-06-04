"""
Competitive Adaptive Reweighted Sampling (CARS) for Feature Selection

This module implements the CARS algorithm for selecting the most informative
wavelengths from NIR spectral data by iteratively sampling subsets of features
based on their weights.
"""

from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold


class CARSSelector(BaseEstimator, TransformerMixin):
    """
    Competitive Adaptive Reweighted Sampling (CARS) for wavelength selection.

    CARS is an algorithm that selects variables by adaptively sampling subsets
    of variables according to their weights, which are derived from
    PLS regression coefficients.

    Parameters:
    -----------
    n_pls_components : int, optional (default=10)
        Number of PLS components to use.
    n_sampling_runs : int, optional (default=50)
        Number of sampling runs.
    n_features_to_select : int, optional (default=None)
        Number of features to select. If None, selects the subset with minimum RMSE.
    exponential_decay : float, optional (default=0.95)
        Controls how fast the number of sampled variables decreases.
    cv : int, optional (default=5)
        Number of cross-validation folds.
    wavelengths : list or array, optional (default=None)
        Actual wavelength values (for visualization and interpretation).
    random_state : int, optional (default=None)
        Random seed for reproducibility.
    """  

    def __init__(
        self,
        n_pls_components: int = 10,
        n_sampling_runs: int = 50,
        n_features_to_select: Optional[int] = None,
        exponential_decay: float = 0.95,
        cv: int = 5,
        wavelengths: Optional[List[float]] = None,
        random_state: Optional[int] = None,
    ):
        self.n_pls_components = n_pls_components
        self.n_sampling_runs = n_sampling_runs
        self.n_features_to_select = n_features_to_select
        self.exponential_decay = exponential_decay
        self.cv = cv
        self.wavelengths = wavelengths
        self.random_state = random_state
        self.selected_features_mask_ = None
        self.selected_features_indices_ = None
        self.weights_ = None
        self.rmse_history_ = None
        self.feature_subset_history_ = None

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
    ) -> "CARSSelector":
        """
        Run CARS to select features.

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
        # Set random seed if specified
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Convert to numpy arrays if pandas objects
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else y

        n_samples, n_features = X_array.shape

        # Initialize PLS model
        pls = PLSRegression(
            n_components=min(self.n_pls_components, n_samples, n_features)
        )

        # Initial fit to get regression coefficients
        pls.fit(X_array, y_array)

        # Get absolute regression coefficients as initial weights
        weights = np.abs(pls.coef_.ravel())

        # Normalize weights to [0, 1]
        weights = (weights - weights.min()) / (
            weights.max() - weights.min() + 1e-10
        )

        # Storage for RMSE values and selected feature subsets
        rmse_history = []
        feature_subset_history = []

        # Run the CARS algorithm
        for i in range(self.n_sampling_runs):
            # Compute how many features to keep in this iteration
            keep_fraction = self.exponential_decay**i
            # At least keep 1 feature
            k = max(int(n_features * keep_fraction), 1)

            # Phase 1: Enforced selection - keep top k features by weight
            top_k_indices = np.argsort(weights)[-k:]
            mask = np.zeros(n_features, dtype=bool)
            mask[top_k_indices] = True

            # Phase 2: Adaptive reweighted sampling - probabilistic selection
            # based on weights
            if i > 0:  # Skip in first iteration
                # Normalize remaining weights for sampling
                remaining_weights = weights[mask]
                sampling_probs = remaining_weights / remaining_weights.sum()

                # Sample features based on their weights
                selected_indices = np.random.choice(
                    top_k_indices,
                    size=min(k, len(top_k_indices)),
                    replace=False,
                    p=sampling_probs,
                )

                # Update mask to only include sampled features
                mask = np.zeros(n_features, dtype=bool)
                mask[selected_indices] = True

            # Skip if no features selected
            if not np.any(mask):
                continue

            X_selected = X_array[:, mask]

            # Cross-validation to evaluate this subset
            cv_rmse = 0
            kf = KFold(
                n_splits=self.cv, shuffle=True, random_state=self.random_state
            )
            fold_count = 0

            for train_idx, test_idx in kf.split(X_selected):
                # Skip if train or test set too small
                if (
                    len(train_idx) <= self.n_pls_components
                    or len(test_idx) == 0
                ):
                    continue

                X_train, X_test = X_selected[train_idx], X_selected[test_idx]
                y_train, y_test = y_array[train_idx], y_array[test_idx]

                # Fit PLS on training data
                n_comp = min(
                    self.n_pls_components, X_train.shape[1], X_train.shape[0]
                )
                pls_cv = PLSRegression(n_components=max(1, n_comp))
                pls_cv.fit(X_train, y_train)

                # Predict and compute RMSE on test data
                y_pred = pls_cv.predict(X_test).ravel()
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                cv_rmse += rmse
                fold_count += 1

            cv_rmse /= max(1, fold_count)  # Average RMSE across folds

            # Store results
            rmse_history.append(cv_rmse)
            feature_subset_history.append(mask.copy())

            # Update weights based on the PLS regression on this subset
            pls.fit(X_selected, y_array)
            subset_indices = np.where(mask)[0]
            for j, idx in enumerate(subset_indices):
                weights[idx] = np.abs(pls.coef_.ravel()[j])

            # Normalize weights again
            weights = (weights - weights.min()) / (
                weights.max() - weights.min() + 1e-10
            )

        # Select the best subset based on minimum RMSE
        if len(rmse_history) == 0:
            # If no valid subsets were created, use all features
            self.selected_features_mask_ = np.ones(n_features, dtype=bool)
            self.selected_features_indices_ = np.arange(n_features)
        elif self.n_features_to_select is None:
            best_idx = np.argmin(rmse_history)
            self.selected_features_mask_ = feature_subset_history[best_idx]
            self.selected_features_indices_ = np.where(
                self.selected_features_mask_
            )[0]
        else:
            # Find the subset with closest number of features to
            # n_features_to_select
            n_features_per_subset = [
                np.sum(mask) for mask in feature_subset_history
            ]
            best_idx = np.argmin(
                np.abs(
                    np.array(n_features_per_subset) - self.n_features_to_select
                )
            )
            self.selected_features_mask_ = feature_subset_history[best_idx]
            self.selected_features_indices_ = np.where(
                self.selected_features_mask_
            )[0]

        self.weights_ = weights
        self.rmse_history_ = rmse_history
        self.feature_subset_history_ = feature_subset_history

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
            raise ValueError("CARSSelector has not been fitted yet.")

        if isinstance(X, pd.DataFrame):
            return X.iloc[:, self.selected_features_mask_]
        else:
            return X[:, self.selected_features_mask_]

    def plot_selection_history(
        self,
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot the RMSE history during feature selection.

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
        if self.rmse_history_ is None:
            raise ValueError("CARSSelector has not been fitted yet.")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Plot RMSE history
        ax1.plot(
            range(1, len(self.rmse_history_) + 1), self.rmse_history_, "o-"
        )
        ax1.set_xlabel("Sampling Run")
        ax1.set_ylabel("RMSE")
        ax1.set_title("RMSE vs. Sampling Run")

        # Plot number of selected features
        n_features = [np.sum(mask) for mask in self.feature_subset_history_]
        ax2.plot(range(1, len(n_features) + 1), n_features, "o-")
        ax2.set_xlabel("Sampling Run")
        ax2.set_ylabel("Number of Selected Features")
        ax2.set_title("Feature Subset Size vs. Sampling Run")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_selected_wavelengths(
        self,
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot the selected wavelengths.

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
        if self.selected_features_mask_ is None:
            raise ValueError("CARSSelector has not been fitted yet.")

        fig, ax = plt.subplots(figsize=figsize)

        if self.wavelengths is not None:
            x_values = self.wavelengths
            selected_wavelengths = self.wavelengths[
                self.selected_features_mask_
            ]
        else:
            x_values = np.arange(len(self.selected_features_mask_))
            selected_wavelengths = x_values[self.selected_features_mask_]

        # Plot all wavelengths as background
        ax.plot(
            x_values,
            np.zeros_like(x_values),
            "o",
            color="lightgray",
            alpha=0.5,
        )

        # Highlight selected wavelengths
        ax.plot(
            selected_wavelengths,
            np.zeros_like(selected_wavelengths),
            "o",
            color="red",
            markersize=10,
        )

        ax.set_xlabel("Wavelength (nm)")
        ax.set_title(
            f"Selected Wavelengths (CARS): {
                len(selected_wavelengths)
            } features"
        )
        ax.set_yticks([])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig
