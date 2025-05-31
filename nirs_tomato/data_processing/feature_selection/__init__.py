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

from .genetic_algorithm_selector import GeneticAlgorithmSelector
from .cars_selector import CARSSelector
from .pls_vip_selector import PLSVIPSelector

__all__ = [
    "GeneticAlgorithmSelector",
    "CARSSelector",
    "PLSVIPSelector"
] 