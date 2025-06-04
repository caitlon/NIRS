#!/usr/bin/env python
"""
Configuration Template Generator for NIR Spectroscopy Experiments.

This script generates a template configuration file for NIR spectroscopy experiments.
It provides a convenient way to create new experiment configurations with default
values that can be customized.

Examples:
    # Create a basic PLS configuration template
    python experiments/create_config.py --name my_pls_experiment --model pls --output configs/my_pls_experiment.yaml

    # Create a Random Forest configuration with MSC transformation
    python experiments/create_config.py --name rf_msc_experiment --model rf --transform msc --output configs/rf_msc_experiment.yaml

    # Create an XGBoost configuration with feature selection
    python experiments/create_config.py --name xgb_feature_selection --model xgb --feature_selection vip --output configs/xgb_feature_selection.yaml
"""

import argparse
import os
import sys
from typing import Optional

# Add parent directory to the module search path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from nirs_tomato.config import (  # noqa: E402
    DataConfig,
    ExperimentConfig,
    FeatureSelectionConfig,
    MLflowConfig,
    ModelConfig,
    SavGolConfig,
)


def create_config_template(
    name: str,
    description: Optional[str] = None,
    data_path: str = "data/raw/Tomato_Viavi_Brix_model_pulp.csv",
    target_column: str = "Brix",
    transform: str = "snv",
    model_type: str = "pls",
    feature_selection: str = "none",
    output_file: str = None,
) -> ExperimentConfig:
    """
    Create a configuration template for an experiment.

    Args:
        name: Name of the experiment
        description: Description of the experiment
        data_path: Path to input data CSV file
        target_column: Name of target column in data
        transform: Spectral transformation type (snv, msc, none)
        model_type: Type of regression model (pls, svr, rf, xgb, lgbm)
        feature_selection: Feature selection method (none, ga, cars, vip)
        output_file: Path to save the YAML configuration

    Returns:
        ExperimentConfig object with the template configuration
    """
    # Generate description if not provided
    if not description:
        transform_name = (
            "SNV"
            if transform == "snv"
            else "MSC"
            if transform == "msc"
            else "no"
        )
        fs_name = (
            "without feature selection"
            if feature_selection == "none"
            else f"with {feature_selection.upper()} feature selection"
        )
        description = f"{model_type.upper()} regression with {transform_name} transformation {fs_name} for {target_column} prediction"

    # Create data configuration
    data_config = DataConfig(
        data_path=data_path,
        target_column=target_column,
        transform=transform,
        savgol=(
            SavGolConfig(enabled=True, window_length=15, polyorder=2, deriv=0)
            if transform != "none"
            else SavGolConfig(enabled=False)
        ),
    )

    # Create feature selection configuration
    fs_config = FeatureSelectionConfig(
        method=feature_selection,
        n_features=30 if feature_selection != "none" else 0,
        plot_selection=True if feature_selection != "none" else False,
    )

    # Create model configuration based on type
    model_config = ModelConfig(
        model_type=model_type,
        tune_hyperparams=False,
        test_size=0.2,
        random_state=42,
    )

    # Create MLflow configuration
    mlflow_config = MLflowConfig(
        enabled=True, experiment_name="nirs-tomato-experiments"
    )

    # Create the full experiment configuration
    config = ExperimentConfig(
        name=name,
        description=description,
        data=data_config,
        feature_selection=fs_config,
        model=model_config,
        mlflow=mlflow_config,
        output_dir="models",
        results_dir="results",
        verbose=False,
    )

    # Save to YAML if output file is specified
    if output_file:
        # Ensure directory exists
        os.makedirs(
            os.path.dirname(os.path.abspath(output_file)), exist_ok=True
        )
        config.to_yaml(output_file)
        print(f"Configuration template saved to: {output_file}")

    return config


def main():
    """Parse command line arguments and create configuration template."""
    parser = argparse.ArgumentParser(
        description="Create a configuration template for NIR spectroscopy experiments."
    )

    parser.add_argument(
        "--name", type=str, required=True, help="Name of the experiment"
    )
    parser.add_argument(
        "--description", type=str, help="Description of the experiment"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/raw/Tomato_Viavi_Brix_model_pulp.csv",
        help="Path to input data CSV file",
    )
    parser.add_argument(
        "--target_column",
        type=str,
        default="Brix",
        help="Name of target column in data",
    )
    parser.add_argument(
        "--transform",
        type=str,
        default="snv",
        choices=["snv", "msc", "none"],
        help="Spectral transformation type",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="pls",
        choices=["pls", "svr", "rf", "xgb", "lgbm"],
        help="Type of regression model",
    )
    parser.add_argument(
        "--feature_selection",
        type=str,
        default="none",
        choices=["none", "ga", "cars", "vip"],
        help="Feature selection method",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save the YAML configuration",
    )

    args = parser.parse_args()

    create_config_template(
        name=args.name,
        description=args.description,
        data_path=args.data_path,
        target_column=args.target_column,
        transform=args.transform,
        model_type=args.model,
        feature_selection=args.feature_selection,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()
