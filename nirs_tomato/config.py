"""
Configuration module for NIR tomato spectroscopy experiments.

This module defines Pydantic models for configuration of experiments,
data processing pipelines, and model training.
"""

import os
from enum import Enum
from typing import List, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class TransformType(str, Enum):
    """Types of spectral transformations."""

    SNV = "snv"
    MSC = "msc"
    NONE = "none"


class FeatureSelectionMethod(str, Enum):
    """Types of feature selection methods."""

    NONE = "none"
    GA = "ga"  # Genetic Algorithm
    CARS = "cars"  # Competitive Adaptive Reweighted Sampling
    VIP = "vip"  # Variable Importance in Projection


class ModelType(str, Enum):
    """Types of regression models."""

    PLS = "pls"
    SVR = "svr"
    RF = "rf"
    XGB = "xgb"
    LGBM = "lgbm"


class SavGolConfig(BaseModel):
    """Configuration for Savitzky-Golay filtering."""

    enabled: bool = Field(
        default=True,
        description="Whether to apply SG filtering")
    window_length: int = Field(default=15,
                               description="Window length for SG filter")
    polyorder: int = Field(
        default=2,
        description="Polynomial order for SG filter")
    deriv: int = Field(default=0, description="Derivative order")

    @field_validator("window_length")
    def window_length_must_be_odd(cls, v):
        """Validate that window_length is odd."""
        if v % 2 == 0:
            raise ValueError("window_length must be odd")
        return v


class FeatureSelectionConfig(BaseModel):
    """Configuration for feature selection."""

    method: FeatureSelectionMethod = Field(
        default=FeatureSelectionMethod.NONE,
        description="Feature selection method")
    n_features: int = Field(
        default=20,
        description="Number of features to select")
    plot_selection: bool = Field(
        default=False, description="Whether to plot selected features"
    )

    # Additional parameters for specific methods
    # Genetic Algorithm
    ga_population_size: int = Field(
        default=50, description="GA population size")
    ga_n_generations: int = Field(
        default=20, description="GA number of generations")
    ga_crossover_prob: float = Field(
        default=0.5, description="GA crossover probability"
    )
    ga_mutation_prob: float = Field(
        default=0.2, description="GA mutation probability")

    # CARS
    cars_n_sampling_runs: int = Field(
        default=50, description="CARS number of sampling runs"
    )
    cars_exponential_decay: float = Field(
        default=0.95, description="CARS exponential decay"
    )

    # VIP
    vip_n_components: int = Field(
        default=10, description="VIP number of PLS components"
    )


class ModelConfig(BaseModel):
    """Configuration for model training."""

    model_type: ModelType = Field(
        default=ModelType.PLS, description="Type of regression model"
    )
    tune_hyperparams: bool = Field(
        default=False, description="Whether to perform hyperparameter tuning"
    )
    test_size: float = Field(
        default=0.2,
        description="Proportion of data to use for testing",
        ge=0.1,
        le=0.5)
    random_state: int = Field(default=42,
                              description="Random seed for reproducibility")

    # Model-specific parameters
    # PLS
    pls_n_components: int = Field(
        default=10, description="PLS number of components")

    # SVR
    svr_kernel: str = Field(default="rbf", description="SVR kernel")
    svr_C: float = Field(
        default=1.0,
        description="SVR regularization parameter")
    svr_epsilon: float = Field(
        default=0.1,
        description="SVR epsilon parameter")
    svr_gamma: str = Field(default="scale", description="SVR gamma parameter")

    # Random Forest
    rf_n_estimators: int = Field(
        default=100, description="RF number of estimators")
    rf_max_depth: Optional[int] = Field(
        default=None, description="RF max depth")
    rf_min_samples_split: int = Field(
        default=2, description="RF min samples split")
    rf_min_samples_leaf: int = Field(
        default=1, description="RF min samples leaf")

    # XGBoost
    xgb_n_estimators: int = Field(
        default=100, description="XGB number of estimators")
    xgb_learning_rate: float = Field(
        default=0.1, description="XGB learning rate")
    xgb_max_depth: int = Field(default=3, description="XGB max depth")
    xgb_subsample: float = Field(default=0.8, description="XGB subsample")
    xgb_colsample_bytree: float = Field(
        default=0.8, description="XGB colsample_bytree")

    # LightGBM
    lgbm_n_estimators: int = Field(
        default=100, description="LGBM number of estimators")
    lgbm_learning_rate: float = Field(
        default=0.1, description="LGBM learning rate")
    lgbm_max_depth: int = Field(default=-1, description="LGBM max depth")
    lgbm_num_leaves: int = Field(default=31, description="LGBM num leaves")


class DataConfig(BaseModel):
    """Configuration for data loading and preprocessing."""

    data_path: str = Field(..., description="Path to input CSV file")
    target_column: str = Field(..., description="Name of target column")
    exclude_columns: List[str] = Field(
        default=[
            "Instrument Serial Number",
            "Notes",
            "Timestamp",
            "Integration Time",
            "wetlab ID",
            "Lab",
        ],
        description="Columns to exclude from processing",
    )
    transform: TransformType = Field(
        default=TransformType.SNV,
        description="Spectral transformation to apply")
    savgol: SavGolConfig = Field(
        default_factory=SavGolConfig,
        description="Savitzky-Golay filter configuration")
    remove_outliers: bool = Field(
        default=False, description="Whether to detect and remove outliers"
    )


class MLflowConfig(BaseModel):
    """Configuration for MLflow tracking."""

    enabled: bool = Field(
        default=False,
        description="Whether to use MLflow tracking")
    tracking_uri: Optional[str] = Field(
        default=None, description="MLflow tracking URI")
    experiment_name: str = Field(
        default="nirs-tomato", description="MLflow experiment name"
    )


class ExperimentConfig(BaseModel):
    """Main configuration for experiments."""

    model_config = ConfigDict(extra="allow")

    name: str = Field(..., description="Experiment name")
    description: Optional[str] = Field(
        default=None, description="Experiment description"
    )
    output_dir: str = Field(
        default="models", description="Directory to save trained model"
    )
    results_dir: str = Field(
        default="results", description="Directory to save experiment results"
    )
    verbose: bool = Field(
        default=False,
        description="Whether to enable verbose output")
    data: DataConfig = Field(..., description="Data configuration")
    feature_selection: FeatureSelectionConfig = Field(
        default_factory=FeatureSelectionConfig,
        description="Feature selection configuration",
    )
    model: ModelConfig = Field(
        default_factory=ModelConfig, description="Model configuration"
    )
    mlflow: MLflowConfig = Field(
        default_factory=MLflowConfig, description="MLflow configuration"
    )

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ExperimentConfig":
        """
        Load configuration from a YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            ExperimentConfig object
        """
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(
                f"Configuration file not found: {yaml_path}")

        with open(yaml_path, "r") as file:
            config_dict = yaml.safe_load(file)

        # Convert string values to corresponding Enum types
        if "data" in config_dict and "transform" in config_dict["data"]:
            transform_value = config_dict["data"]["transform"]
            if isinstance(transform_value, str):
                try:
                    config_dict["data"]["transform"] = TransformType(
                        transform_value)
                except ValueError:
                    pass  # Leave as is if conversion fails

        if (
            "feature_selection" in config_dict
            and "method" in config_dict["feature_selection"]
        ):
            method_value = config_dict["feature_selection"]["method"]
            if isinstance(method_value, str):
                try:
                    config_dict["feature_selection"]["method"] = FeatureSelectionMethod(
                        method_value)
                except ValueError:
                    pass

        if "model" in config_dict and "model_type" in config_dict["model"]:
            model_type_value = config_dict["model"]["model_type"]
            if isinstance(model_type_value, str):
                try:
                    config_dict["model"]["model_type"] = ModelType(
                        model_type_value)
                except ValueError:
                    pass

        return cls(**config_dict)

    def to_yaml(self, yaml_path: str) -> None:
        """
        Save configuration to a YAML file.

        Args:
            yaml_path: Path to save YAML configuration
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(yaml_path)), exist_ok=True)

        # Convert Enum values to strings for proper YAML serialization
        config_dict = self.model_dump()

        # Convert Enum values to strings
        if "data" in config_dict and "transform" in config_dict["data"]:
            config_dict["data"]["transform"] = str(
                config_dict["data"]["transform"])

        if (
            "feature_selection" in config_dict
            and "method" in config_dict["feature_selection"]
        ):
            config_dict["feature_selection"]["method"] = str(
                config_dict["feature_selection"]["method"]
            )

        if "model" in config_dict and "model_type" in config_dict["model"]:
            config_dict["model"]["model_type"] = str(
                config_dict["model"]["model_type"])

        with open(yaml_path, "w") as file:
            yaml.dump(
                config_dict,
                file,
                default_flow_style=False,
                sort_keys=False)

    def get_experiment_name(self) -> str:
        """
        Generate a standard experiment name based on configuration.

        Returns:
            Formatted experiment name
        """
        components = [self.model.model_type.value, self.data.transform.value]

        if self.data.savgol.enabled:
            components.append(f"sg{self.data.savgol.window_length}")

        if self.feature_selection.method != FeatureSelectionMethod.NONE:
            components.append(f"{self.feature_selection.method.value}{
                self.feature_selection.n_features}")

        if self.model.tune_hyperparams:
            components.append("tuned")

        return "_".join(components)
