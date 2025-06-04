#!/usr/bin/env python
"""
Analyze trained models and extract metrics.

This script loads trained models from the models directory,
extracts their metrics, and provides a summary of model performance.

Example:
    $ python analyze_models.py
"""

import argparse
import logging
import os
from typing import Any, Dict

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze trained models and extract metrics."
    )

    parser.add_argument(
        "--models_dir",
        type=str,
        default="models",
        help="Directory containing trained models",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save analysis results",
    )

    return parser.parse_args()


def extract_model_info(model_file: str) -> Dict[str, Any]:
    """
    Extract information from a trained model file.

    Args:
        model_file: Path to the trained model file

    Returns:
        Dictionary with model information
    """
    try:
        # Load model data
        model_data = joblib.load(model_file)

        # Extract basic information
        model_info = {
            "model_file": model_file,
            "model_name": os.path.basename(model_file),
        }

        # Extract model type (xgb, rf, svr, etc.)
        model_type = os.path.basename(model_file).split("_")[0]
        model_info["model_type"] = model_type

        # Extract target variable
        target = os.path.basename(model_file).split("_")[1]
        model_info["target"] = target

        # Extract training date if available
        date_parts = os.path.basename(model_file).split("_")[2:]
        if date_parts:
            training_date = "_".join(date_parts).replace(".pkl", "")
            model_info["training_date"] = training_date

        # Extract metrics from model data
        if isinstance(model_data, dict):
            # Look for metrics in different locations
            if "metrics" in model_data:
                metrics = model_data["metrics"]
                for metric_name, metric_value in metrics.items():
                    model_info[metric_name] = metric_value
            else:
                # Try to extract individual metrics
                for metric in ["rmse", "r2", "mae", "explained_variance"]:
                    if metric in model_data:
                        model_info[metric] = model_data[metric]

        # Extract feature importances if available
        if model_type in ["rf", "xgb"]:
            if "model" in model_data:
                model = model_data["model"]
                if hasattr(model, "feature_importances_"):
                    importances = model.feature_importances_
                    feature_names = model_data.get(
                        "feature_names",
                        [f"feature_{i}" for i in range(len(importances))],
                    )

                    # Store top 10 feature importances
                    importance_dict = dict(zip(feature_names, importances))
                    top_features = sorted(
                        importance_dict.items(),
                        key=lambda x: x[1],
                        reverse=True,
                    )[:10]
                    model_info["top_features"] = dict(top_features)

        return model_info

    except Exception as e:
        logger.error(f"Error extracting info from {model_file}: {e}")
        return {"model_file": model_file, "error": str(e)}


def analyze_models(models_dir: str) -> pd.DataFrame:
    """
    Analyze all models in the specified directory.

    Args:
        models_dir: Directory containing trained models

    Returns:
        DataFrame with model information and metrics
    """
    # Find all model files
    model_files = [
        os.path.join(models_dir, f)
        for f in os.listdir(models_dir)
        if f.endswith(".pkl") and os.path.isfile(os.path.join(models_dir, f))
    ]

    if not model_files:
        logger.warning(f"No model files found in {models_dir}")
        return pd.DataFrame()

    logger.info(f"Found {len(model_files)} model files")

    # Extract information from each model
    model_info_list = []
    for model_file in model_files:
        logger.info(f"Analyzing model: {os.path.basename(model_file)}")
        model_info = extract_model_info(model_file)
        model_info_list.append(model_info)

    # Create DataFrame
    models_df = pd.DataFrame(model_info_list)

    return models_df


def plot_model_metrics(models_df: pd.DataFrame, output_dir: str):
    """
    Generate plots of model metrics.

    Args:
        models_df: DataFrame with model information and metrics
        output_dir: Directory to save plots
    """
    if models_df.empty:
        logger.warning("No model data to plot")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Sort models by RMSE
    if "rmse" in models_df.columns:
        models_df = models_df.sort_values("rmse")

    # Bar plot of RMSE by model
    plt.figure(figsize=(12, 6))
    if "rmse" in models_df.columns:
        sns.barplot(x="model_name", y="rmse", data=models_df)
        plt.xticks(rotation=45, ha="right")
        plt.title("RMSE by Model")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "rmse_by_model.png"))
        plt.close()

    # Bar plot of R² by model
    plt.figure(figsize=(12, 6))
    if "r2" in models_df.columns:
        sns.barplot(x="model_name", y="r2", data=models_df)
        plt.xticks(rotation=45, ha="right")
        plt.title("R² by Model")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "r2_by_model.png"))
        plt.close()

    # Group by model type and compute average metrics
    if "model_type" in models_df.columns:
        metrics_by_type = (
            models_df.groupby("model_type")
            .mean(numeric_only=True)
            .reset_index()
        )

        # Bar plot of average RMSE by model type
        plt.figure(figsize=(10, 6))
        if "rmse" in metrics_by_type.columns:
            sns.barplot(x="model_type", y="rmse", data=metrics_by_type)
            plt.title("Average RMSE by Model Type")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "rmse_by_model_type.png"))
            plt.close()

        # Bar plot of average R² by model type
        plt.figure(figsize=(10, 6))
        if "r2" in metrics_by_type.columns:
            sns.barplot(x="model_type", y="r2", data=metrics_by_type)
            plt.title("Average R² by Model Type")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "r2_by_model_type.png"))
            plt.close()


def main():
    """Main function for analyzing models."""
    # Parse command line arguments
    args = parse_args()

    # Analyze models
    models_df = analyze_models(args.models_dir)

    if models_df.empty:
        logger.warning("No models found or analysis failed")
        return

    # Print model summary
    logger.info("\nModel Summary:")
    logger.info("-" * 80)

    # Sort by RMSE if available
    if "rmse" in models_df.columns:
        models_df_sorted = models_df.sort_values("rmse")

        # Print table of metrics
        logger.info(f"{'Model':<30} {'RMSE':<10} {'R²':<10} {'MAE':<10}")
        logger.info("-" * 80)

        for _, row in models_df_sorted.iterrows():
            rmse = row.get("rmse", np.nan)
            r2 = row.get("r2", np.nan)
            mae = row.get("mae", np.nan)

            logger.info(
                f"{row['model_name']:<30} {rmse:<10.4f} {r2:<10.4f} {mae:<10.4f}"
            )

        # Find best model
        best_model = models_df_sorted.iloc[0]
        logger.info("\nBest Model:")
        logger.info(f"Model: {best_model['model_name']}")
        logger.info(f"RMSE: {best_model.get('rmse', np.nan):.4f}")
        logger.info(f"R²: {best_model.get('r2', np.nan):.4f}")
        logger.info(f"MAE: {best_model.get('mae', np.nan):.4f}")
    else:
        logger.info("No RMSE metrics found in models")

    # Save model analysis to CSV
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "model_analysis.csv")
    models_df.to_csv(output_file, index=False)
    logger.info(f"\nModel analysis saved to {output_file}")

    # Generate plots
    plot_model_metrics(models_df, args.output_dir)
    logger.info(f"Model visualizations saved to {args.output_dir}")


if __name__ == "__main__":
    main()
