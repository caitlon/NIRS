#!/usr/bin/env python
"""
Train a model on NIR tomato spectroscopy data.

This script trains a regression model on tomato NIR spectroscopy data,
optimizing hyperparameters and evaluating performance. It saves the
trained model to disk for later use.

Example:
    $ python train_model.py --data data/tomato_spectra.csv --target SSC
                           --model pls --transform snv --savgol

    # With feature selection:
    $ python train_model.py --data data/tomato_spectra.csv --target SSC
                           --model pls --transform snv --feature_selection vip --n_features 20
"""

import argparse
import logging
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

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
from nirs_tomato.modeling.evaluation import (
    evaluate_regression_model,
    print_regression_metrics,
)
from nirs_tomato.modeling.model_factory import (
    create_pls_model,
    create_rf_model,
    create_svr_model,
    create_xgb_model,
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
        description="Train a model on NIR tomato spectroscopy data."
    )

    # Data options
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to CSV file containing NIR spectroscopy data",
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Target column name (e.g., 'SSC', 'lycopene')",
    )
    parser.add_argument(
        "--exclude_columns",
        type=str,
        nargs="+",
        default=[],
        help="Columns to exclude from processing",
    )

    # Preprocessing options
    parser.add_argument(
        "--transform",
        type=str,
        choices=["snv", "msc", "none"],
        default="snv",
        help="Spectral transformation to apply (snv, msc, or none)",
    )
    parser.add_argument(
        "--savgol", action="store_true", help="Apply Savitzky-Golay filtering"
    )
    parser.add_argument(
        "--window_length",
        type=int,
        default=15,
        help="Window length for Savitzky-Golay filter",
    )
    parser.add_argument(
        "--polyorder",
        type=int,
        default=2,
        help="Polynomial order for Savitzky-Golay filter",
    )
    parser.add_argument(
        "--remove_outliers",
        action="store_true",
        help="Detect and remove outliers",
    )

    # Feature selection options
    parser.add_argument(
        "--feature_selection",
        type=str,
        choices=["none", "ga", "cars", "vip"],
        default="none",
        help="Feature selection method to apply (ga=Genetic Algorithm, cars=Competitive Adaptive Reweighted Sampling, vip=Variable Importance in Projection)",
    )
    parser.add_argument(
        "--n_features",
        type=int,
        default=20,
        help="Number of features to select when using feature selection",
    )
    parser.add_argument(
        "--plot_selection",
        action="store_true",
        help="Plot selected features (saved to output_dir)",
    )

    # Model options
    parser.add_argument(
        "--model",
        type=str,
        choices=["pls", "svr", "rf", "xgb"],
        default="pls",
        help="Regression model to train (pls, svr, rf, xgb)",
    )
    parser.add_argument(
        "--tune_hyperparams",
        action="store_true",
        help="Perform hyperparameter tuning",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Proportion of data to use for testing",
    )

    # Output options
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models",
        help="Directory to save trained model",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose output"
    )

    return parser.parse_args()


def main():
    """Main function for training a model."""
    # Parse command line arguments
    args = parse_args()

    # Set verbose logging
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Load data
    logger.info(f"Loading data from {args.data}")
    df = pd.read_csv(args.data)
    logger.info(f"Loaded dataframe with shape {df.shape}")

    # Create list of transformers based on arguments
    transformers = []

    if args.transform == "snv":
        logger.info("Using SNV transformation")
        transformers.append(SNVTransformer())
    elif args.transform == "msc":
        logger.info("Using MSC transformation")
        transformers.append(MSCTransformer())

    if args.savgol:
        logger.info(
            f"Using Savitzky-Golay filter (window_length={args.window_length}, polyorder={args.polyorder})"
        )
        transformers.append(
            SavGolTransformer(
                window_length=args.window_length, polyorder=args.polyorder
            )
        )

    # Include common non-numeric columns that should be excluded
    common_non_numeric = [
        "Instrument Serial Number",
        "Notes",
        "Timestamp",
        "Integration Time",
        "wetlab ID",
        "Lab",
    ]

    exclude_columns = list(set(args.exclude_columns + common_non_numeric))

    # Preprocess data
    logger.info("Preprocessing data")
    data = preprocess_spectra(
        df=df,
        target_column=args.target,
        transformers=transformers,
        exclude_columns=exclude_columns,
        remove_outliers=args.remove_outliers,
        verbose=args.verbose,
    )

    # Extract processed data
    X = data["X"]
    y = data["y"]

    if y is None or len(y) == 0:
        raise ValueError(
            f"Target column '{args.target}' not found or empty after preprocessing"
        )

    logger.info(f"Preprocessed features shape: {X.shape}")
    logger.info(f"Target values shape: {y.shape}")

    # Split data into train and test sets
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42
    )

    logger.info(f"Training set shape: {X_train.shape}")
    logger.info(f"Test set shape: {X_test.shape}")

    # Apply feature selection if requested
    feature_selector = None
    if args.feature_selection != "none":
        logger.info(
            f"Applying {args.feature_selection.upper()} feature selection"
        )

        # Identify wavelength columns
        from nirs_tomato.data_processing.utils import identify_spectral_columns

        spectral_cols, _ = identify_spectral_columns(X)
        wavelengths = np.array([float(col) for col in spectral_cols])

        if args.feature_selection == "ga":
            # For GA, we need a model to evaluate feature subsets
            rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
            feature_selector = GeneticAlgorithmSelector(
                estimator=rf_model,
                n_features_to_select=args.n_features,
                population_size=30,
                n_generations=10,
                wavelengths=wavelengths,
                random_state=42,
            )
        elif args.feature_selection == "cars":
            feature_selector = CARSSelector(
                n_pls_components=10,
                n_sampling_runs=30,
                n_features_to_select=args.n_features,
                wavelengths=wavelengths,
                random_state=42,
            )
        elif args.feature_selection == "vip":
            feature_selector = PLSVIPSelector(
                n_components=10,
                n_features_to_select=args.n_features,
                wavelengths=wavelengths,
            )

        # Fit and transform data
        feature_selector.fit(X_train, y_train)
        X_train_selected = feature_selector.transform(X_train)
        X_test_selected = feature_selector.transform(X_test)

        # Update data for modeling
        X_train = X_train_selected
        X_test = X_test_selected

        logger.info(
            f"Selected {X_train.shape[1]} features using {args.feature_selection.upper()}"
        )

        # Plot selected features if requested
        if args.plot_selection:
            os.makedirs(args.output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            if args.feature_selection == "ga":
                plot = feature_selector.plot_selected_wavelengths()
                plot_path = os.path.join(
                    args.output_dir, f"ga_selected_features_{timestamp}.png"
                )
                plot.savefig(plot_path, dpi=300, bbox_inches="tight")
                logger.info(f"Saved feature selection plot to {plot_path}")
            elif args.feature_selection == "cars":
                # Plot selection history
                plot_history = feature_selector.plot_selection_history()
                plot_history_path = os.path.join(
                    args.output_dir, f"cars_selection_history_{timestamp}.png"
                )
                plot_history.savefig(
                    plot_history_path, dpi=300, bbox_inches="tight"
                )

                # Plot selected wavelengths
                plot_wavelengths = feature_selector.plot_selected_wavelengths()
                plot_wavelengths_path = os.path.join(
                    args.output_dir, f"cars_selected_features_{timestamp}.png"
                )
                plot_wavelengths.savefig(
                    plot_wavelengths_path, dpi=300, bbox_inches="tight"
                )

                logger.info(f"Saved CARS plots to {args.output_dir}")
            elif args.feature_selection == "vip":
                plot = feature_selector.plot_vip_scores()
                plot_path = os.path.join(
                    args.output_dir, f"vip_scores_{timestamp}.png"
                )
                plot.savefig(plot_path, dpi=300, bbox_inches="tight")
                logger.info(f"Saved VIP scores plot to {plot_path}")

    # Create model
    logger.info(f"Creating {args.model.upper()} model")

    if args.model == "pls":
        model, param_grid = create_pls_model()
    elif args.model == "svr":
        model, param_grid = create_svr_model()
    elif args.model == "rf":
        model, param_grid = create_rf_model()
    elif args.model == "xgb":
        model, param_grid = create_xgb_model()
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    # Train model with or without hyperparameter tuning
    if args.tune_hyperparams:
        logger.info("Performing hyperparameter tuning")
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=5,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
        )
        grid_search.fit(X_train, y_train)

        # Get best model
        model = grid_search.best_estimator_
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {-grid_search.best_score_:.4f} MSE")
    else:
        logger.info("Training model with default parameters")
        model.fit(X_train, y_train)

    # Evaluate model
    logger.info("Evaluating model")
    metrics, y_pred = evaluate_regression_model(model, X_test, y_test)
    print_regression_metrics(metrics)

    # Save model
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Include feature selection in model filename
    fs_suffix = ""
    if args.feature_selection != "none":
        fs_suffix = f"_{args.feature_selection}{args.n_features}"

    model_filename = f"{args.model}_{args.target}{fs_suffix}_{timestamp}.pkl"
    model_path = os.path.join(args.output_dir, model_filename)

    logger.info(f"Saving model to {model_path}")

    # Save model and metadata
    model_data = {
        "model": model,
        "preprocessing_pipeline": data["preprocessing_pipeline"],
        "spectral_columns": data["spectral_columns"],
        "feature_names": list(
            X_train.columns
            if hasattr(X_train, "columns")
            else [f"feature_{i}" for i in range(X_train.shape[1])]
        ),
        "target": args.target,
        "metrics": metrics,
        "transform_type": args.transform,
        "savgol": args.savgol,
        "feature_selection": {
            "method": args.feature_selection,
            "n_features": (
                args.n_features if args.feature_selection != "none" else None
            ),
            "selector": feature_selector,
        },
        "training_date": timestamp,
    }

    joblib.dump(model_data, model_path)
    logger.info("Model training completed successfully")


if __name__ == "__main__":
    main()
