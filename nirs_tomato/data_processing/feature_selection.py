"""
Feature Selection Methods for NIR Spectroscopy

This module has been refactored for better modularity.
The feature selection classes are now located in the feature_selection/ directory.

This file is kept for backward compatibility.
"""

from .feature_selection.cars_selector import CARSSelector
from .feature_selection.genetic_algorithm_selector import GeneticAlgorithmSelector
from .feature_selection.pls_vip_selector import PLSVIPSelector

__all__ = ["GeneticAlgorithmSelector", "CARSSelector", "PLSVIPSelector"]
