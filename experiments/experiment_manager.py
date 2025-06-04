#!/usr/bin/env python
"""
Experiment Manager for NIR Spectroscopy Analysis.

This module provides a clean interface for running experiments in the NIRS project.
It serves as the main entry point for all experiment-related operations.

Example:
    from experiments.experiment_manager import ExperimentManager

    # Run a single experiment from a config file
    manager = ExperimentManager()
    manager.run_from_config("configs/pls_snv_savgol.yaml")

    # Or run all experiments in a directory
    manager.run_from_config_dir("configs/")

    # Create a new experiment configuration
    from nirs_tomato.config import ExperimentConfig, DataConfig, ModelConfig
    config = ExperimentConfig(
        name="my_custom_experiment",
        data=DataConfig(
            data_path="data/raw/my_data.csv",
            target_column="Brix"
        ),
        model=ModelConfig(model_type="rf")
    )
    manager.run_from_config_object(config)
"""

import logging
import os
import sys
from typing import Any, Dict

import pandas as pd

# Add parent directory to the module search path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from nirs_tomato.config import ExperimentConfig  # noqa: E402
from nirs_tomato.data_processing.feature_selection import (  # noqa: E402
    CARSSelector,
    GeneticAlgorithmSelector,
    PLSVIPSelector,
)
from nirs_tomato.data_processing.pipeline import (
    preprocess_spectra,  # noqa: E402
)
from nirs_tomato.data_processing.transformers import (  # noqa: E402
    MSCTransformer,
    SavGolTransformer,
    SNVTransformer,
)
from nirs_tomato.modeling.evaluation import (
    evaluate_regression_model,  # noqa: E402
)
from nirs_tomato.modeling.hyperparameter_tuning import (  # noqa: E402
    bayesian_hyperparameter_search,
)
from nirs_tomato.modeling.regression_models import (  # noqa: E402
    save_model,
    train_regression_model,
)
from nirs_tomato.modeling.tracking import (  # noqa: E402
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


class ExperimentManager:
    """
    Manager for running NIR spectroscopy experiments.

    This class provides a clean interface for running experiments with different
    configurations, tracking results, and managing experiment lifecycle.
    """

    def __init__(self, log_level: int = logging.INFO):
        """
        Initialize the experiment manager.

        Args:
            log_level: Logging level to use (default: logging.INFO)
        """
        self.logger = logger
        self.logger.setLevel(log_level)

    def run_from_config(self, config_path: str) -> Dict[str, Any]:
        """
        Run a single experiment from a configuration file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            Dictionary with experiment results
        """
        try:
            # Load configuration
            self.logger.info(f"Loading configuration from {config_path}")
            config = ExperimentConfig.from_yaml(config_path)

            # Run experiment with the loaded config
            return self.run_from_config_object(config)

        except Exception as e:
            self.logger.error(
                f"Error running experiment from {config_path}: {str(e)}"
            )
            raise

    def run_from_config_dir(
        self, config_dir: str
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run multiple experiments from a directory of configuration files.

        Args:
            config_dir: Directory containing YAML configuration files

        Returns:
            Dictionary mapping experiment names to their results
        """
        if not os.path.isdir(config_dir):
            raise ValueError(f"Config directory not found: {config_dir}")

        results = {}
        config_files = [
            os.path.join(config_dir, f)
            for f in os.listdir(config_dir)
            if f.endswith(".yaml") or f.endswith(".yml")
        ]

        if not config_files:
            self.logger.warning(
                f"No YAML configuration files found in {config_dir}"
            )
            return results

        self.logger.info(f"Found {len(config_files)} configuration files")

        for config_file in config_files:
            try:
                experiment_name = os.path.basename(config_file).split(".")[0]
                self.logger.info(f"Running experiment from {config_file}")
                results[experiment_name] = self.run_from_config(config_file)
            except Exception as e:
                self.logger.error(
                    f"Error running experiment from {config_file}: {str(e)}"
                )
                # Continue with next experiment instead of stopping
                results[experiment_name] = {"error": str(e)}

        return results

    def run_from_config_object(
        self, config: ExperimentConfig
    ) -> Dict[str, Any]:
        """
        Run an experiment from a configuration object.

        This is the core method that actually runs the experiment based on
        the provided configuration.

        Args:
            config: ExperimentConfig object with experiment settings

        Returns:
            Dictionary with experiment results
        """
        # Set logging level
        if config.verbose:
            self.logger.setLevel(logging.DEBUG)

        # Generate experiment name if not provided
        if not config.name:
            config.name = config.get_experiment_name()

        self.logger.info(f"Running experiment: {config.name}")

        # Create result directories if they don't exist
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.results_dir, exist_ok=True)

        # Initialize MLflow if enabled
        if config.mlflow.enabled:
            self._setup_mlflow(config)

        try:
            # Process data
            processed_data = self._process_data(config)

            # Apply feature selection if configured
            processed_data = self._apply_feature_selection(
                config, processed_data
            )

            # Train and evaluate model
            results = self._train_and_evaluate(config, processed_data)

            # Save results
            self._save_results(config, results)

            # End MLflow run if active
            if config.mlflow.enabled:
                end_run()

            return results

        except Exception as e:
            self.logger.error(f"Error in experiment {config.name}: {str(e)}")
            # End MLflow run if active
            if config.mlflow.enabled:
                end_run()
            raise

    def _setup_mlflow(self, config: ExperimentConfig) -> None:
        """Set up MLflow tracking for this experiment."""
        self.logger.info("Initializing MLflow tracking")
        # Create a unique run name
        from datetime import datetime

        run_name = f"{config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Start the run
        start_run(
            run_name=run_name,
            experiment_name=config.mlflow.experiment_name,
            tracking_uri=config.mlflow.tracking_uri,
        )

        # Log configuration parameters
        log_parameters(config.model_dump())

    def _process_data(self, config: ExperimentConfig) -> Dict[str, Any]:
        """Process data according to configuration."""
        # Load data
        self.logger.info(f"Loading data from {config.data.data_path}")
        df = pd.read_csv(config.data.data_path)
        self.logger.info(f"Loaded dataframe with shape {df.shape}")

        # Create transformers based on configuration
        transformers = []

        if config.data.transform == "snv":
            self.logger.info("Using SNV transformation")
            transformers.append(SNVTransformer())
        elif config.data.transform == "msc":
            self.logger.info("Using MSC transformation")
            transformers.append(MSCTransformer())

        if config.data.savgol.enabled:
            self.logger.info(
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
        self.logger.info("Preprocessing data")
        processed_data = preprocess_spectra(
            df=df,
            target_column=config.data.target_column,
            transformers=transformers,
            exclude_columns=config.data.exclude_columns,
            remove_outliers=config.data.remove_outliers,
            verbose=config.verbose,
        )

        # Extract processed data
        X = processed_data["X"]
        y = processed_data["y"]

        if y is None or len(y) == 0:
            raise ValueError(
                f"Target column '{config.data.target_column}' not found or empty after preprocessing"
            )

        self.logger.info(f"Preprocessed features shape: {X.shape}")
        self.logger.info(f"Target values shape: {y.shape}")

        return processed_data

    def _apply_feature_selection(
        self, config: ExperimentConfig, processed_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply feature selection if configured."""
        X = processed_data["X"]
        y = processed_data["y"]

        feature_selector = None
        if config.feature_selection.method != "none":
            self.logger.info(
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

            self.logger.info(f"After feature selection, X shape: {X.shape}")

            # If feature selection is enabled and plotting is requested
            if config.feature_selection.plot_selection and hasattr(
                feature_selector, "plot_selected_wavelengths"
            ):
                self.logger.info("Plotting selected features")
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
                    self.logger.info(
                        f"Saved selected features to {selected_features_path}"
                    )

                # Log plot to MLflow if enabled
                if config.mlflow.enabled:
                    log_artifact(fig, f"{config.name}_selected_features.png")

        # Update X in processed_data
        processed_data["X"] = X

        return processed_data

    def _train_and_evaluate(
        self, config: ExperimentConfig, processed_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Train and evaluate model based on configuration."""
        from sklearn.model_selection import train_test_split

        X = processed_data["X"]
        y = processed_data["y"]

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=config.model.test_size,
            random_state=config.model.random_state,
        )

        self.logger.info(f"Training set shape: {X_train.shape}")
        self.logger.info(f"Test set shape: {X_test.shape}")

        # Train model
        model_type = config.model.model_type
        self.logger.info(f"Training {model_type.upper()} model")

        model_params = self._get_model_params(config)

        # Tune hyperparameters if requested
        if config.model.tune_hyperparams:
            self.logger.info("Performing hyperparameter tuning")
            model, best_params = bayesian_hyperparameter_search(
                model_type=model_type,
                X_train=X_train,
                y_train=y_train,
                random_state=config.model.random_state,
                verbose=config.verbose,
            )

            # Update model_params with tuned parameters
            model_params.update(best_params)

            # Log tuned parameters
            if config.mlflow.enabled:
                log_parameters(
                    {f"tuned_{k}": v for k, v in best_params.items()}
                )
        else:
            # Train model with default or specified parameters
            model = train_regression_model(
                model_type=model_type,
                X_train=X_train,
                y_train=y_train,
                model_params=model_params,
                verbose=config.verbose,
                random_state=config.model.random_state,
            )

        # Evaluate model
        self.logger.info("Evaluating model")
        metrics, y_pred = evaluate_regression_model(model, X_test, y_test)

        for metric_name, metric_value in metrics.items():
            self.logger.info(f"{metric_name}: {metric_value:.4f}")

            # Log metrics to MLflow if enabled
            if config.mlflow.enabled:
                log_metrics({metric_name: metric_value})

        # Save model
        model_path = os.path.join(config.output_dir, f"{config.name}.pkl")
        save_model(model, model_path)
        self.logger.info(f"Model saved to {model_path}")

        # Log model to MLflow if enabled
        if config.mlflow.enabled:
            log_model(model, f"{config.name}")

        # Return results
        results = {
            "model": model,
            "metrics": metrics,
            "predictions": y_pred,
            "model_path": model_path,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        }

        return results

    def _get_model_params(self, config: ExperimentConfig) -> Dict[str, Any]:
        """Extract model parameters from configuration."""
        model_type = config.model.model_type
        params = {}

        if model_type == "pls":
            params = {"n_components": config.model.pls_n_components}
        elif model_type == "svr":
            params = {
                "kernel": config.model.svr_kernel,
                "C": config.model.svr_C,
                "epsilon": config.model.svr_epsilon,
                "gamma": config.model.svr_gamma,
            }
        elif model_type == "rf":
            params = {
                "n_estimators": config.model.rf_n_estimators,
                "max_depth": config.model.rf_max_depth,
                "min_samples_split": config.model.rf_min_samples_split,
                "min_samples_leaf": config.model.rf_min_samples_leaf,
                "random_state": config.model.random_state,
            }
        elif model_type == "xgb":
            params = {
                "n_estimators": config.model.xgb_n_estimators,
                "learning_rate": config.model.xgb_learning_rate,
                "max_depth": config.model.xgb_max_depth,
                "subsample": config.model.xgb_subsample,
                "colsample_bytree": config.model.xgb_colsample_bytree,
                "random_state": config.model.random_state,
            }
        elif model_type == "lgbm":
            params = {
                "n_estimators": config.model.lgbm_n_estimators,
                "learning_rate": config.model.lgbm_learning_rate,
                "max_depth": config.model.lgbm_max_depth,
                "num_leaves": config.model.lgbm_num_leaves,
                "random_state": config.model.random_state,
            }

        return params

    def _save_results(
        self, config: ExperimentConfig, results: Dict[str, Any]
    ) -> None:
        """Save experiment results."""
        # Save predictions to CSV
        predictions_path = os.path.join(
            config.results_dir, f"{config.name}_predictions.csv"
        )
        pd.DataFrame(
            {"y_true": results["y_test"], "y_pred": results["predictions"]}
        ).to_csv(predictions_path, index=False)
        self.logger.info(f"Predictions saved to {predictions_path}")

        # Save metrics to text file
        metrics_path = os.path.join(
            config.results_dir, f"{config.name}_metrics.txt"
        )
        with open(metrics_path, "w") as f:
            for metric_name, metric_value in results["metrics"].items():
                f.write(f"{metric_name}: {metric_value:.4f}\n")
        self.logger.info(f"Metrics saved to {metrics_path}")

        # Save experiment configuration
        config_path = os.path.join(
            config.results_dir, f"{config.name}_config.yaml"
        )
        config.to_yaml(config_path)
        self.logger.info(f"Configuration saved to {config_path}")


def run_experiment(config_path: str) -> Dict[str, Any]:
    """
    Convenience function to run a single experiment from a configuration file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Dictionary with experiment results
    """
    manager = ExperimentManager()
    return manager.run_from_config(config_path)


def run_experiments(config_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    Convenience function to run multiple experiments from a directory.

    Args:
        config_dir: Directory containing YAML configuration files

    Returns:
        Dictionary mapping experiment names to their results
    """
    manager = ExperimentManager()
    return manager.run_from_config_dir(config_dir)


if __name__ == "__main__":
    import argparse

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

    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose output"
    )

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    manager = ExperimentManager(log_level=log_level)

    if args.config:
        manager.run_from_config(args.config)
    elif args.config_dir:
        manager.run_from_config_dir(args.config_dir)
