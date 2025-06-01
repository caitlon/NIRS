"""
Constants used in NIR tomato spectroscopy data processing.
"""

from typing import Dict, List

# Paths
DEFAULT_DATASET_PATH = "data/raw/Tomato_Viavi_Brix_model_pulp.csv"

# Target column
TARGET_COLUMN = "Brix"

# Default preprocessing parameters
DEFAULT_SAVGOL_WINDOW = 15
DEFAULT_SAVGOL_POLYORDER = 2
DEFAULT_SAVGOL_DERIV = 1
DEFAULT_PCA_COMPONENTS = 10
DEFAULT_WAVELENGTH_PERCENT = 0.05
DEFAULT_OUTLIER_THRESHOLD = 3.0

# Default ID columns for sample identification
DEFAULT_ID_COLS = ["SAMPLE NO", "wetlab ID", "plant"]

# Available preprocessing methods
AVAILABLE_PREPROCESSING_METHODS: List[str] = [
    "raw",  # Raw data without preprocessing
    "snv",  # Standard Normal Variate
    "sg1",  # Savitzky-Golay first derivative
    "sg2",  # Savitzky-Golay second derivative
    "msc",  # Multiplicative Scatter Correction
    "snv_pca",  # SNV followed by PCA
    "sg1_pca",  # First derivative followed by PCA
]

# Available aggregation methods
AVAILABLE_AGGREGATION_METHODS: List[str] = ["mean", "median", "max", "min"]

# Regression metrics
REGRESSION_METRICS: List[str] = ["rmse", "mae", "r2", "explained_variance"]

# Model types
REGRESSION_MODELS: Dict[str, str] = {
    "rf": "Random Forest",
    "svr": "Support Vector Regression",
    "elasticnet": "Elastic Net",
    "xgb": "XGBoost",
    "lgbm": "LightGBM",
    "mlp": "Multi-layer Perceptron",
}
