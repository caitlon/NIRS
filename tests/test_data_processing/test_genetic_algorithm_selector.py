"""
Tests for Genetic Algorithm feature selector.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from nirs_tomato.data_processing.feature_selection.genetic_algorithm_selector import (
    GeneticAlgorithmSelector,
)


def test_genetic_algorithm_selector_initialization():
    """Test Genetic Algorithm selector initialization with various parameters."""  
    # Create a simple estimator
    estimator = LinearRegression()

    # Default initialization with estimator
    selector = GeneticAlgorithmSelector(estimator=estimator)
    assert selector.n_features_to_select == 10
    assert selector.population_size == 50
    assert selector.n_generations == 20
    assert selector.crossover_prob == 0.5
    assert selector.mutation_prob == 0.2

    # Custom initialization
    selector = GeneticAlgorithmSelector(
        estimator=estimator,
        n_features_to_select=5,
        population_size=30,
        n_generations=15,
        crossover_prob=0.6,
        mutation_prob=0.1,
        scoring="neg_mean_absolute_error",
        cv=3,
        random_state=42,
    )
    assert selector.n_features_to_select == 5
    assert selector.population_size == 30
    assert selector.n_generations == 15
    assert selector.crossover_prob == 0.6
    assert selector.mutation_prob == 0.1
    assert selector.scoring == "neg_mean_absolute_error"
    assert selector.cv == 3
    assert selector.random_state == 42


def test_genetic_algorithm_selector_fit():
    """Test Genetic Algorithm selector fitting process."""
    # Create synthetic data
    n_samples, n_features = 20, 10
    X = np.random.rand(n_samples, n_features)
    y = np.random.rand(n_samples)

    # Create feature column names
    feature_names = [f"feat_{i}" for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)

    # Create estimator
    estimator = LinearRegression()

    # Fit selector with small parameters for test speed
    selector = GeneticAlgorithmSelector(
        estimator=estimator,
        n_features_to_select=3,
        population_size=5,
        n_generations=2,
        random_state=42,
    )
    selector.fit(X_df, y)

    # Check if attributes are set
    assert hasattr(selector, "selected_features_indices_")
    assert hasattr(selector, "selected_features_mask_")
    assert hasattr(selector, "evolution_history_")
    assert (
        selector.selected_features_mask_.sum() <= 3
    )  # Should select at most n_features_to_select
    assert len(selector.selected_features_indices_) <= 3


def test_genetic_algorithm_selector_transform_array():
    """Test Genetic Algorithm selector transform with numpy array."""
    # Create synthetic data
    n_samples, n_features = 20, 10
    X = np.random.rand(n_samples, n_features)
    y = np.random.rand(n_samples)

    # Create estimator
    estimator = LinearRegression()

    # Fit selector with small parameters for test speed
    selector = GeneticAlgorithmSelector(
        estimator=estimator,
        n_features_to_select=3,
        population_size=5,
        n_generations=2,
        random_state=42,
    )
    selector.fit(X, y)

    # Transform
    X_selected = selector.transform(X)

    # Check output
    assert X_selected.shape[0] == n_samples
    assert X_selected.shape[1] <= 3  # Should have at most 3 features


def test_genetic_algorithm_selector_transform_dataframe():
    """Test Genetic Algorithm selector transform with pandas DataFrame."""
    # Create synthetic data
    n_samples, n_features = 20, 10
    X = np.random.rand(n_samples, n_features)
    y = np.random.rand(n_samples)

    # Create feature column names
    feature_names = [f"feat_{i}" for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)

    # Create estimator
    estimator = LinearRegression()

    # Fit selector with small parameters for test speed
    selector = GeneticAlgorithmSelector(
        estimator=estimator,
        n_features_to_select=3,
        population_size=5,
        n_generations=2,
        random_state=42,
    )
    selector.fit(X_df, y)

    # Transform DataFrame
    X_selected = selector.transform(X_df)

    # Check output
    assert isinstance(X_selected, pd.DataFrame)
    assert X_selected.shape[0] == n_samples
    assert X_selected.shape[1] <= 3  # Should have at most 3 features


def test_genetic_algorithm_selector_fit_transform():
    """Test Genetic Algorithm selector fit_transform method."""
    # Create synthetic data
    n_samples, n_features = 20, 10
    X = np.random.rand(n_samples, n_features)
    y = np.random.rand(n_samples)

    # Create estimator
    estimator = LinearRegression()

    # Fit and transform with small parameters for test speed
    selector = GeneticAlgorithmSelector(
        estimator=estimator,
        n_features_to_select=3,
        population_size=5,
        n_generations=2,
        random_state=42,
    )
    X_selected = selector.fit_transform(X, y)

    # Check output
    assert X_selected.shape[0] == n_samples
    assert X_selected.shape[1] <= 3  # Should have at most 3 features
    assert len(selector.selected_features_indices_) <= 3
