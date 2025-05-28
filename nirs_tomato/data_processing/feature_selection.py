"""
Feature Selection Methods for NIR Spectroscopy

This module provides implementations of advanced feature selection methods 
specifically designed for NIR spectral data:

1. Genetic Algorithm (GA) for wavelength selection
2. Competitive Adaptive Reweighted Sampling (CARS)
3. Variable Importance in Projection (VIP) for PLS models

All selectors follow scikit-learn's BaseEstimator and TransformerMixin interfaces
for seamless integration with preprocessing pipelines.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union, Any, Callable
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_score, KFold
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import deap
from deap import base, creator, tools, algorithms


class GeneticAlgorithmSelector(BaseEstimator, TransformerMixin):
    """
    Genetic Algorithm for wavelength selection in NIR spectroscopy.
    
    This method uses a genetic algorithm to find an optimal subset of wavelengths
    that maximize predictive performance while minimizing the number of features.
    
    Parameters:
    -----------
    estimator : object
        The regression model to use for fitness evaluation.
    n_features_to_select : int, optional (default=10)
        Number of features to select.
    population_size : int, optional (default=50)
        Size of the genetic algorithm population.
    n_generations : int, optional (default=20)
        Number of generations to evolve.
    crossover_prob : float, optional (default=0.5)
        Probability of crossover.
    mutation_prob : float, optional (default=0.2)
        Probability of mutation.
    scoring : callable or str, optional (default='neg_root_mean_squared_error')
        Scoring metric for evaluation.
    cv : int, optional (default=5)
        Number of cross-validation folds.
    wavelengths : list or array, optional (default=None)
        Actual wavelength values (for visualization and interpretation).
    random_state : int, optional (default=None)
        Random seed for reproducibility.
    """
    
    def __init__(
        self,
        estimator: Any,
        n_features_to_select: int = 10,
        population_size: int = 50,
        n_generations: int = 20,
        crossover_prob: float = 0.5,
        mutation_prob: float = 0.2,
        scoring: Union[str, Callable] = 'neg_root_mean_squared_error',
        cv: int = 5,
        wavelengths: Optional[List[float]] = None,
        random_state: Optional[int] = None
    ):
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.scoring = scoring
        self.cv = cv
        self.wavelengths = wavelengths
        self.random_state = random_state
        self.selected_features_mask_ = None
        self.selected_features_indices_ = None
        self.evolution_history_ = None
        
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> 'GeneticAlgorithmSelector':
        """
        Run the genetic algorithm to select features.
        
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
        
        n_features = X_array.shape[1]
        
        # Set up the DEAP environment
        creator.create("FitnessMax", base.Fitness, weights=(1.0, -0.1))  # Maximize score, minimize # of features
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        toolbox = base.Toolbox()
        toolbox.register("attr_bool", np.random.choice, [0, 1])
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=n_features)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=self.population_size)
        
        def evaluate(individual):
            # Count number of selected features
            n_selected = sum(individual)
            if n_selected == 0:  # No features selected, worst fitness
                return -float('inf'), n_features
            
            # Select features according to the individual
            X_selected = X_array[:, np.array(individual, dtype=bool)]
            
            # Evaluate model using cross-validation
            try:
                cv_scores = cross_val_score(
                    self.estimator, X_selected, y_array, 
                    cv=self.cv, scoring=self.scoring
                )
                mean_score = np.mean(cv_scores)
            except Exception as e:
                print(f"Error in evaluation: {e}")
                return -float('inf'), n_features
            
            return mean_score, n_selected
        
        # Register the genetic operators
        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", tools.cxUniform, indpb=0.5)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        # Create initial population
        pop = toolbox.population()
        
        # Set up statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        # Track the best solutions over generations
        hall_of_fame = tools.HallOfFame(1)
        
        # Run the genetic algorithm
        pop, logbook = algorithms.eaSimple(
            pop, toolbox, 
            cxpb=self.crossover_prob, 
            mutpb=self.mutation_prob, 
            ngen=self.n_generations,
            stats=stats, 
            halloffame=hall_of_fame,
            verbose=False
        )
        
        # Extract the best individual
        best_individual = hall_of_fame[0]
        self.selected_features_mask_ = np.array(best_individual, dtype=bool)
        self.selected_features_indices_ = np.where(self.selected_features_mask_)[0]
        
        # Limit to n_features_to_select if necessary
        if sum(self.selected_features_mask_) > self.n_features_to_select:
            # If too many features selected, keep only the top n_features_to_select
            # based on individual feature importance (correlation with target)
            correlations = np.array([
                abs(pearsonr(X_array[:, i], y_array)[0]) 
                for i in self.selected_features_indices_
            ])
            top_indices = self.selected_features_indices_[np.argsort(correlations)[-self.n_features_to_select:]]
            new_mask = np.zeros(n_features, dtype=bool)
            new_mask[top_indices] = True
            self.selected_features_mask_ = new_mask
            self.selected_features_indices_ = top_indices
            
        # Store evolution history
        self.evolution_history_ = logbook
            
        return self
        
    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
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
            raise ValueError("GeneticAlgorithmSelector has not been fitted yet.")
            
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, self.selected_features_mask_]
        else:
            return X[:, self.selected_features_mask_]
            
    def plot_selected_wavelengths(self, figsize: Tuple[int, int] = (12, 6), save_path: Optional[str] = None) -> plt.Figure:
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
            raise ValueError("GeneticAlgorithmSelector has not been fitted yet.")
            
        fig, ax = plt.subplots(figsize=figsize)
        
        if self.wavelengths is not None:
            x_values = self.wavelengths
            selected_wavelengths = self.wavelengths[self.selected_features_mask_]
        else:
            x_values = np.arange(len(self.selected_features_mask_))
            selected_wavelengths = x_values[self.selected_features_mask_]
            
        # Plot all wavelengths as background
        ax.plot(x_values, np.zeros_like(x_values), 'o', color='lightgray', alpha=0.5)
        
        # Highlight selected wavelengths
        ax.plot(selected_wavelengths, np.zeros_like(selected_wavelengths), 'o', color='red', markersize=10)
        
        ax.set_xlabel('Wavelength (nm)')
        ax.set_title(f'Selected Wavelengths (GA): {len(selected_wavelengths)} features')
        ax.set_yticks([])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig


class CARSSelector(BaseEstimator, TransformerMixin):
    """
    Competitive Adaptive Reweighted Sampling (CARS) for wavelength selection.
    
    CARS is an algorithm that selects variables by adaptively sampling subsets of variables
    according to their weights, which are derived from PLS regression coefficients.
    
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
        random_state: Optional[int] = None
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
        
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> 'CARSSelector':
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
        pls = PLSRegression(n_components=min(self.n_pls_components, n_samples, n_features))
        
        # Initial fit to get regression coefficients
        pls.fit(X_array, y_array)
        
        # Get absolute regression coefficients as initial weights
        weights = np.abs(pls.coef_.ravel())
        
        # Normalize weights to [0, 1]
        weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-10)
        
        # Storage for RMSE values and selected feature subsets
        rmse_history = []
        feature_subset_history = []
        
        # Run the CARS algorithm
        for i in range(self.n_sampling_runs):
            # Compute how many features to keep in this iteration
            keep_fraction = self.exponential_decay ** i
            k = max(int(n_features * keep_fraction), 1)  # At least keep 1 feature
            
            # Phase 1: Enforced selection - keep top k features by weight
            top_k_indices = np.argsort(weights)[-k:]
            mask = np.zeros(n_features, dtype=bool)
            mask[top_k_indices] = True
            
            # Phase 2: Adaptive reweighted sampling - probabilistic selection based on weights
            if i > 0:  # Skip in first iteration
                # Normalize remaining weights for sampling
                remaining_weights = weights[mask]
                sampling_probs = remaining_weights / remaining_weights.sum()
                
                # Sample features based on their weights
                selected_indices = np.random.choice(
                    top_k_indices, 
                    size=min(k, len(top_k_indices)),
                    replace=False,
                    p=sampling_probs
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
            kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
            fold_count = 0
            
            for train_idx, test_idx in kf.split(X_selected):
                # Skip if train or test set too small
                if len(train_idx) <= self.n_pls_components or len(test_idx) == 0:
                    continue
                    
                X_train, X_test = X_selected[train_idx], X_selected[test_idx]
                y_train, y_test = y_array[train_idx], y_array[test_idx]
                
                # Fit PLS on training data
                n_comp = min(self.n_pls_components, X_train.shape[1], X_train.shape[0])
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
            weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-10)
            
        # Select the best subset based on minimum RMSE
        if len(rmse_history) == 0:
            # If no valid subsets were created, use all features
            self.selected_features_mask_ = np.ones(n_features, dtype=bool)
            self.selected_features_indices_ = np.arange(n_features)
        elif self.n_features_to_select is None:
            best_idx = np.argmin(rmse_history)
            self.selected_features_mask_ = feature_subset_history[best_idx]
            self.selected_features_indices_ = np.where(self.selected_features_mask_)[0]
        else:
            # Find the subset with closest number of features to n_features_to_select
            n_features_per_subset = [np.sum(mask) for mask in feature_subset_history]
            best_idx = np.argmin(np.abs(np.array(n_features_per_subset) - self.n_features_to_select))
            self.selected_features_mask_ = feature_subset_history[best_idx]
            self.selected_features_indices_ = np.where(self.selected_features_mask_)[0]
            
        self.weights_ = weights
        self.rmse_history_ = rmse_history
        self.feature_subset_history_ = feature_subset_history
        
        return self
        
    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
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
            
    def plot_selection_history(self, figsize: Tuple[int, int] = (12, 6), save_path: Optional[str] = None) -> plt.Figure:
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
        ax1.plot(range(1, len(self.rmse_history_) + 1), self.rmse_history_, 'o-')
        ax1.set_xlabel('Sampling Run')
        ax1.set_ylabel('RMSE')
        ax1.set_title('RMSE vs. Sampling Run')
        
        # Plot number of selected features
        n_features = [np.sum(mask) for mask in self.feature_subset_history_]
        ax2.plot(range(1, len(n_features) + 1), n_features, 'o-')
        ax2.set_xlabel('Sampling Run')
        ax2.set_ylabel('Number of Selected Features')
        ax2.set_title('Feature Subset Size vs. Sampling Run')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def plot_selected_wavelengths(self, figsize: Tuple[int, int] = (12, 6), save_path: Optional[str] = None) -> plt.Figure:
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
            selected_wavelengths = self.wavelengths[self.selected_features_mask_]
        else:
            x_values = np.arange(len(self.selected_features_mask_))
            selected_wavelengths = x_values[self.selected_features_mask_]
            
        # Plot all wavelengths as background
        ax.plot(x_values, np.zeros_like(x_values), 'o', color='lightgray', alpha=0.5)
        
        # Highlight selected wavelengths
        ax.plot(selected_wavelengths, np.zeros_like(selected_wavelengths), 'o', color='red', markersize=10)
        
        ax.set_xlabel('Wavelength (nm)')
        ax.set_title(f'Selected Wavelengths (CARS): {len(selected_wavelengths)} features')
        ax.set_yticks([])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig


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
        wavelengths: Optional[List[float]] = None
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
        
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> 'PLSVIPSelector':
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
        self.vip_scores_ = self._calculate_vip_scores(self.pls_model_, X_array, n_features)
        
        # Select features based on VIP scores
        if self.use_n_features and self.n_features_to_select is not None:
            n_to_select = min(self.n_features_to_select, n_features)
            vip_threshold = np.sort(self.vip_scores_)[-n_to_select]
        else:
            vip_threshold = self.threshold
            
        self.selected_features_mask_ = self.vip_scores_ >= vip_threshold
        self.selected_features_indices_ = np.where(self.selected_features_mask_)[0]
        
        return self
        
    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
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
            
    def _calculate_vip_scores(self, pls_model: PLSRegression, X: np.ndarray, n_features: int) -> np.ndarray:
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
                weighted_sum += q[i] * (w[j, i] / np.sqrt(np.sum(w[:, i]**2)))**2
            
            vip_scores[j] = np.sqrt(n_features * weighted_sum / np.sum(q))
        
        return vip_scores
        
    def plot_vip_scores(self, figsize: Tuple[int, int] = (12, 6), save_path: Optional[str] = None) -> plt.Figure:
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
        ax.plot(x_values, self.vip_scores_, 'o-', markersize=4)
        
        # Add horizontal line at threshold
        if not self.use_n_features:
            ax.axhline(y=self.threshold, color='r', linestyle='--', label=f'Threshold = {self.threshold}')
        else:
            vip_threshold = np.sort(self.vip_scores_)[-self.n_features_to_select]
            ax.axhline(y=vip_threshold, color='r', linestyle='--', label=f'Threshold (top {self.n_features_to_select})')
            
        # Highlight selected wavelengths
        if self.wavelengths is not None:
            selected_wavelengths = self.wavelengths[self.selected_features_mask_]
        else:
            selected_wavelengths = x_values[self.selected_features_mask_]
            
        selected_vips = self.vip_scores_[self.selected_features_mask_]
        ax.plot(selected_wavelengths, selected_vips, 'ro', markersize=6, label='Selected')
        
        ax.set_xlabel('Wavelength (nm)' if self.wavelengths is not None else 'Feature Index')
        ax.set_ylabel('VIP Score')
        ax.set_title(f'PLS VIP Scores (Selected: {np.sum(self.selected_features_mask_)} features)')
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig 