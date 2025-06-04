"""
Genetic Algorithm for Feature Selection in NIR Spectroscopy

This module implements a genetic algorithm for selecting the most informative
wavelengths from NIR spectral data.
"""

from typing import Any, Callable, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from deap import algorithms, base, creator, tools
from scipy.stats import pearsonr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_score


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
        scoring: Union[str, Callable] = "neg_root_mean_squared_error",
        cv: int = 5,
        wavelengths: Optional[List[float]] = None,
        random_state: Optional[int] = None,
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

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
    ) -> "GeneticAlgorithmSelector":
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
        creator.create(
            "FitnessMax", base.Fitness, weights=(1.0, -0.1)
        )  # Maximize score, minimize # of features
        creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()
        toolbox.register("attr_bool", np.random.choice, [0, 1])
        toolbox.register(
            "individual",
            tools.initRepeat,
            creator.Individual,
            toolbox.attr_bool,
            n=n_features,
        )
        toolbox.register(
            "population",
            tools.initRepeat,
            list,
            toolbox.individual,
            n=self.population_size,
        )

        def evaluate(individual):
            # Count number of selected features
            n_selected = sum(individual)
            if n_selected == 0:  # No features selected, worst fitness
                return -float("inf"), n_features

            # Select features according to the individual
            X_selected = X_array[:, np.array(individual, dtype=bool)]

            # Evaluate model using cross-validation
            try:
                cv_scores = cross_val_score(
                    self.estimator,
                    X_selected,
                    y_array,
                    cv=self.cv,
                    scoring=self.scoring,
                )
                mean_score = np.mean(cv_scores)
            except Exception as e:
                print(f"Error in evaluation: {e}")
                return -float("inf"), n_features

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
            pop,
            toolbox,
            cxpb=self.crossover_prob,
            mutpb=self.mutation_prob,
            ngen=self.n_generations,
            stats=stats,
            halloffame=hall_of_fame,
            verbose=False,
        )

        # Extract the best individual
        best_individual = hall_of_fame[0]
        self.selected_features_mask_ = np.array(best_individual, dtype=bool)
        self.selected_features_indices_ = np.where(
            self.selected_features_mask_
        )[0]

        # Limit to n_features_to_select if necessary
        if sum(self.selected_features_mask_) > self.n_features_to_select:
            # If too many features selected, keep only the top n_features_to_select  
            # based on individual feature importance (correlation with target)
            correlations = np.array(
                [
                    abs(pearsonr(X_array[:, i], y_array)[0])
                    for i in self.selected_features_indices_
                ]
            )
            top_indices = self.selected_features_indices_[
                np.argsort(correlations)[-self.n_features_to_select :]
            ]
            new_mask = np.zeros(n_features, dtype=bool)
            new_mask[top_indices] = True
            self.selected_features_mask_ = new_mask
            self.selected_features_indices_ = top_indices

        # Store evolution history
        self.evolution_history_ = logbook

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
            raise ValueError(
                "GeneticAlgorithmSelector has not been fitted yet."
            )

        if isinstance(X, pd.DataFrame):
            return X.iloc[:, self.selected_features_mask_]
        else:
            return X[:, self.selected_features_mask_]

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
            raise ValueError(
                "GeneticAlgorithmSelector has not been fitted yet."
            )

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
            f"Selected Wavelengths (GA): {len(selected_wavelengths)} features"
        )
        ax.set_yticks([])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig
