#!/usr/bin/env python
"""
Run experiments using configuration files.

This script runs experiments based on YAML configuration files,
making it easy to reproduce experiments with different parameters.

Example:
    $ python run_from_config.py --config configs/pls_snv_savgol.yaml

    $ python run_from_config.py --config configs/rf_msc_feature_selection.yaml

    $ python run_from_config.py --config_dir configs/
"""

import argparse
import logging
import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from nirs_tomato.config import ExperimentConfig
from nirs_tomato.data_processing.feature_selection import (
    CARSSelector,
    GeneticAlgorithmSelector,
    PLSVIPSelector,
)
from nirs_tomato.data_processing.pipeline import preprocess_spectra
from nirs_tomato.data_processing.transformers import (
    MSCTransformer,
    SavGolTransformer,
    SNVTransformer,
)
from nirs_tomato.modeling.hyperparameter_tuning import (
    bayesian_hyperparameter_search,
)
from nirs_tomato.modeling.regression_models import (
    evaluate_regression_model,
    save_model,
    train_regression_model,
)
from nirs_tomato.modeling.tracking import (
    end_run,
    log_artifact,
    log_metrics,
    log_model,
    log_parameters,
    start_run,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run experiments using configuration files."
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--config", type=str, help="Path to YAML configuration file"
    )
    group.add_argument(
        "--config_dir",
        type=str,
        help="Directory containing YAML configuration files",
    )

    return parser.parse_args()


def run_single_experiment(config_path: str) -> None:
    """
    Run a single experiment based on a configuration file.

    Args:
        config_path: Path to YAML configuration file
    """
    try:
        # Load configuration
        logger.info(f"Loading configuration from {config_path}")
        config = ExperimentConfig.from_yaml(config_path)

        # Set logging level
        if config.verbose:
            logger.setLevel(logging.DEBUG)

        # Generate experiment name if not provided
        if not config.name:
            config.name = config.get_experiment_name()

        logger.info(f"Running experiment: {config.name}")

        # Initialize MLflow if enabled
        if config.mlflow.enabled:
            logger.info("Initializing MLflow tracking")
            # Create a unique run name
            from datetime import datetime

            run_name = (
                f"{config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )

            # Start the run
            start_run(
                run_name=run_name,
                experiment_name=config.mlflow.experiment_name,
                tracking_uri=config.mlflow.tracking_uri,
            )

            # Log configuration parameters
            log_parameters(config.model_dump())

        # Load data
        logger.info(f"Loading data from {config.data.data_path}")
        df = pd.read_csv(config.data.data_path)
        logger.info(f"Loaded dataframe with shape {df.shape}")

        # Create transformers based on configuration
        transformers = []

        if config.data.transform == "snv":
            logger.info("Using SNV transformation")
            transformers.append(SNVTransformer())
        elif config.data.transform == "msc":
            logger.info("Using MSC transformation")
            transformers.append(MSCTransformer())

        if config.data.savgol.enabled:
            logger.info(
                f"Using Savitzky-Golay filter (window_length={config.data.savgol.window_length}, "
                f"polyorder={config.data.savgol.polyorder}, deriv={config.data.savgol.deriv})"
            )
            transformers.append(
                SavGolTransformer(
                    window_length=config.data.savgol.window_length,
                    polyorder=config.data.savgol.polyorder,
                    deriv=config.data.savgol.deriv,
                )
            )

        # Preprocess data
        logger.info("Preprocessing data")
        data = preprocess_spectra(
            df=df,
            target_column=config.data.target_column,
            transformers=transformers,
            exclude_columns=config.data.exclude_columns,
            remove_outliers=config.data.remove_outliers,
            verbose=config.verbose,
        )

        # Extract processed data
        X = data["X"]
        y = data["y"]

        if y is None or len(y) == 0:
            raise ValueError(
                f"Target column '{config.data.target_column}' not found or empty after preprocessing"
            )

        logger.info(f"Preprocessed features shape: {X.shape}")
        logger.info(f"Target values shape: {y.shape}")

        # Apply feature selection if configured
        feature_selector = None
        if config.feature_selection.method != "none":
            logger.info(
                f"Using {config.feature_selection.method.upper()} feature selection "
                f"with {config.feature_selection.n_features} features"
            )

            if config.feature_selection.method == "ga":
                from sklearn.ensemble import RandomForestRegressor

                estimator = RandomForestRegressor(
                    n_estimators=50, random_state=config.model.random_state
                )
                feature_selector = GeneticAlgorithmSelector(
                    estimator=estimator,
                    n_features_to_select=config.feature_selection.n_features,
                    population_size=config.feature_selection.ga_population_size,
                    n_generations=config.feature_selection.ga_n_generations,
                    crossover_prob=config.feature_selection.ga_crossover_prob,
                    mutation_prob=config.feature_selection.ga_mutation_prob,
                    random_state=config.model.random_state,
                )
            elif config.feature_selection.method == "cars":
                feature_selector = CARSSelector(
                    n_pls_components=config.feature_selection.vip_n_components,
                    n_sampling_runs=config.feature_selection.cars_n_sampling_runs,
                    n_features_to_select=config.feature_selection.n_features,
                    exponential_decay=config.feature_selection.cars_exponential_decay,
                    random_state=config.model.random_state,
                )
            elif config.feature_selection.method == "vip":
                feature_selector = PLSVIPSelector(
                    n_components=config.feature_selection.vip_n_components,
                    n_features_to_select=config.feature_selection.n_features,
                )

            # Fit feature selector and transform data
            feature_selector.fit(X, y)
            X = feature_selector.transform(X)

            logger.info(f"After feature selection, X shape: {X.shape}")

            # If feature selection is enabled and plotting is requested, create plot but don't save
            if (
                config.feature_selection.method != "none"
                and config.feature_selection.plot_selection
                and hasattr(feature_selector, "plot_selected_wavelengths")
            ):
                logger.info("Plotting selected features")
                fig = feature_selector.plot_selected_wavelengths()

                # Save selected features to CSV
                selected_features_path = os.path.join(
                    config.results_dir, f"{config.name}_selected_features.csv"
                )

                # Get the selected wavelengths
                if hasattr(feature_selector, "selected_features_indices_"):
                    selected_indices = (
                        feature_selector.selected_features_indices_
                    )
                    if (
                        hasattr(feature_selector, "wavelengths")
                        and feature_selector.wavelengths is not None
                    ):
                        selected_wavelengths = [
                            feature_selector.wavelengths[i]
                            for i in selected_indices
                        ]
                    else:
                        # Use column names from X as wavelengths
                        selected_wavelengths = [
                            X.columns[i] for i in selected_indices
                        ]

                    # Save to CSV
                    pd.DataFrame(
                        {
                            "feature_index": selected_indices,
                            "wavelength": selected_wavelengths,
                        }
                    ).to_csv(selected_features_path, index=False)
                    logger.info(
                        f"Saved selected features to {selected_features_path}"
                    )

                # Log plot to MLflow if enabled (but don't save to disk)
                if config.mlflow.enabled:
                    log_figure(fig, f"{config.name}_selected_features.png")  # noqa: F821

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=config.model.test_size,
            random_state=config.model.random_state,
        )

        logger.info(f"Training set shape: {X_train.shape}")
        logger.info(f"Test set shape: {X_test.shape}")

        # Set up model parameters based on configuration
        model_params = {}

        if config.model.model_type == "pls":
            model_params = {"n_components": config.model.pls_n_components}
        elif config.model.model_type == "svr":
            model_params = {
                "kernel": config.model.svr_kernel,
                "C": config.model.svr_C,
                "epsilon": config.model.svr_epsilon,
                "gamma": config.model.svr_gamma,
            }
        elif config.model.model_type == "rf":
            model_params = {"n_estimators": config.model.rf_n_estimators,
                "max_depth": config.model.rf_max_depth,
                "min_samples_split": config.model.rf_min_samples_split,
                "min_samples_leaf": config.model.rf_min_samples_leaf, }
        elif config.model.model_type == "xgb":
            model_params = {
                "n_estimators": config.model.xgb_n_estimators,
                "learning_rate": config.model.xgb_learning_rate,
                "max_depth": config.model.xgb_max_depth,
                "subsample": config.model.xgb_subsample,
                "colsample_bytree": config.model.xgb_colsample_bytree,
            }
        elif config.model.model_type == "lgbm":
            model_params = {
                "n_estimators": config.model.lgbm_n_estimators,
                "learning_rate": config.model.lgbm_learning_rate,
                "max_depth": config.model.lgbm_max_depth,
                "num_leaves": config.model.lgbm_num_leaves,
            }

        # Train model
        logger.info(f"Training {config.model.model_type.upper()} model")

        if config.model.tune_hyperparams:
            logger.info("Hyperparameter tuning enabled")

            # Use Bayesian optimization for hyperparameter tuning
            n_trials = 30  # Default
            cv_folds = 3  # Default

            # Try to get custom values from config if available
            if hasattr(config.model, "n_trials"):
                n_trials = config.model.n_trials
            if hasattr(config.model, "cv_folds"):
                cv_folds = config.model.cv_folds

            model, search_results = bayesian_hyperparameter_search(
                X_train=X_train,
                y_train=y_train,
                X_val=X_test,
                y_val=y_test,
                model_type=config.model.model_type,
                n_trials=n_trials,
                cv=cv_folds,
                scoring="neg_root_mean_squared_error",
                random_state=config.model.random_state,
                verbose=config.verbose,
            )

            # Log best hyperparameters
            logger.info("Best hyperparameters found:")
            for param, value in search_results["best_params"].items():
                logger.info(f"  {param}: {value}")

            # Save hyperparameter optimization results to CSV
            hp_results_path = os.path.join(
                config.results_dir, f"{config.name}_hp_optimization.csv"
            )

            # Convert trials to DataFrame
            trials_df = pd.DataFrame(
                [
                    {
                        **{
                            "trial_number": t.number,
                            "value": t.value,
                            "state": t.state.name,
                        },
                        **t.params,
                    }
                    for t in search_results["study"].trials
                ]
            )

            # Save to CSV
            trials_df.to_csv(hp_results_path, index=False)
            logger.info(
                f"Saved hyperparameter optimization results to {hp_results_path}"
            )

            # Log as artifact if MLflow enabled
            if config.mlflow.enabled:
                log_artifact(hp_results_path)
        else:
            model = train_regression_model(
                X_train=X_train,
                y_train=y_train,
                model_type=config.model.model_type,
                model_params=model_params,
                random_state=config.model.random_state,
                verbose=config.verbose,
            )

        # Evaluate model
        logger.info("Evaluating model")
        evaluation = evaluate_regression_model(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_test,
            y_val=y_test,
        )

        # Print metrics
        logger.info("Training metrics:")
        for metric, value in evaluation["train_metrics"].items():
            logger.info(f"  {metric}: {value:.4f}")

        logger.info("Validation metrics:")
        for metric, value in evaluation["val_metrics"].items():
            logger.info(f"  {metric}: {value:.4f}")

        # Create regression plot
        logger.info("Creating regression plot")

        # Save results to CSV
        os.makedirs(config.results_dir, exist_ok=True)
        results_path = os.path.join(
            config.results_dir, f"{config.name}_results.csv"
        )

        # Create DataFrame with actual and predicted values
        results_df = pd.DataFrame(
            {
                "actual": y_test,
                "predicted": evaluation["y_val_pred"],
                "error": y_test - evaluation["y_val_pred"],
            }
        )

        # Add metrics as metadata at the top of the file
        with open(results_path, "w") as f:
            f.write("# Results for experiment: {}\n".format(config.name))
            f.write("# Train metrics:\n")
            for metric, value in evaluation["train_metrics"].items():
                f.write("#   {}: {:.4f}\n".format(metric, value))
            f.write("# Validation metrics:\n")
            for metric, value in evaluation["val_metrics"].items():
                f.write("#   {}: {:.4f}\n".format(metric, value))
            f.write("\n")

        # Append the DataFrame to the file
        results_df.to_csv(results_path, mode="a", index=True)
        logger.info(f"Saved results to {results_path}")

        # Save model
        os.makedirs(config.output_dir, exist_ok=True)
        model_path = os.path.join(
            config.output_dir,
            f"{config.model.model_type}_{config.data.target_column}_{config.name}.pkl",
        )

        # Create model info dictionary
        model_info = {
            "name": config.name,
            "description": config.description,
            "model_type": config.model.model_type,
            "transform": config.data.transform,
            "savgol": config.data.savgol.enabled,
            "feature_selection": config.feature_selection.method,
            "n_features": X.shape[1],
            "train_metrics": evaluation["train_metrics"],
            "val_metrics": evaluation["val_metrics"],
            "config": config.model_dump(),
        }

        logger.info(f"Saving model to {model_path}")
        save_model(model, model_path, model_info)

        # Log to MLflow if enabled
        if config.mlflow.enabled:
            # Log metrics
            for metric, value in evaluation["train_metrics"].items():
                log_metrics({f"train_{metric}": value})

            for metric, value in evaluation["val_metrics"].items():
                log_metrics({f"val_{metric}": value})

            # Log model
            log_model(model, config.name)

            # Log artifacts
            log_artifact(model_path)
            log_artifact(results_path)

            # End the run
            end_run()

        logger.info(f"Experiment {config.name} completed successfully")

    except Exception as e:
        logger.error(f"Error running experiment: {str(e)}")
        logger.exception(e)
        if config.mlflow.enabled:
            end_run()
        sys.exit(1)


def run_experiments_from_directory(config_dir: str) -> None:
    """
    Run experiments for all YAML configurations in a directory.

    Args:
        config_dir: Directory containing YAML configuration files
    """
    if not os.path.isdir(config_dir):
        logger.error(f"Directory not found: {config_dir}")
        sys.exit(1)

    yaml_files = [
        os.path.join(config_dir, f)
        for f in os.listdir(config_dir)
        if f.endswith((".yaml", ".yml"))
    ]

    if not yaml_files:
        logger.error(f"No YAML configuration files found in {config_dir}")
        sys.exit(1)

    logger.info(f"Found {len(yaml_files)} configuration files")

    for config_path in yaml_files:
        logger.info(f"Running experiment from {config_path}")
        run_single_experiment(config_path)


def main():
    """Main entry point."""
    args = parse_args()

    if args.config:
        run_single_experiment(args.config)
    elif args.config_dir:
        run_experiments_from_directory(args.config_dir)


if __name__ == "__main__":
    main()
