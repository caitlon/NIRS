#!/usr/bin/env python
"""
Train a model on NIR tomato spectroscopy data.

This script trains a regression model on tomato NIR spectroscopy data,
optimizing hyperparameters and evaluating performance. It saves the
trained model to disk for later use.

Example:
    $ python train_model.py --data data/tomato_spectra.csv --target SSC 
                           --model pls --transform snv --savgol
"""

import argparse
import logging
import os
import pickle
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from src.data_processing.transformers import SNVTransformer, MSCTransformer, SavGolTransformer
from src.data_processing.pipeline import preprocess_spectra
from src.modeling.model_factory import (
    create_pls_model,
    create_svr_model,
    create_rf_model,
    create_xgb_model
)
from src.modeling.evaluation import evaluate_regression_model, print_regression_metrics

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
        "--data", type=str, required=True, 
        help="Path to CSV file containing NIR spectroscopy data"
    )
    parser.add_argument(
        "--target", type=str, required=True, 
        help="Target column name (e.g., 'SSC', 'lycopene')"
    )
    parser.add_argument(
        "--exclude_columns", type=str, nargs="+", default=[],
        help="Columns to exclude from processing"
    )
    
    # Preprocessing options
    parser.add_argument(
        "--transform", type=str, choices=["snv", "msc", "none"], default="snv",
        help="Spectral transformation to apply (snv, msc, or none)"
    )
    parser.add_argument(
        "--savgol", action="store_true", 
        help="Apply Savitzky-Golay filtering"
    )
    parser.add_argument(
        "--window_length", type=int, default=15,
        help="Window length for Savitzky-Golay filter"
    )
    parser.add_argument(
        "--polyorder", type=int, default=2,
        help="Polynomial order for Savitzky-Golay filter"
    )
    parser.add_argument(
        "--remove_outliers", action="store_true", 
        help="Detect and remove outliers"
    )
    
    # Model options
    parser.add_argument(
        "--model", type=str, choices=["pls", "svr", "rf", "xgb"], default="pls",
        help="Regression model to train (pls, svr, rf, xgb)"
    )
    parser.add_argument(
        "--tune_hyperparams", action="store_true", 
        help="Perform hyperparameter tuning"
    )
    parser.add_argument(
        "--test_size", type=float, default=0.2,
        help="Proportion of data to use for testing"
    )
    
    # Output options
    parser.add_argument(
        "--output_dir", type=str, default="models",
        help="Directory to save trained model"
    )
    parser.add_argument(
        "--verbose", action="store_true", 
        help="Enable verbose output"
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
        logger.info(f"Using Savitzky-Golay filter (window_length={args.window_length}, polyorder={args.polyorder})")
        transformers.append(SavGolTransformer(window_length=args.window_length, polyorder=args.polyorder))

    # Include common non-numeric columns that should be excluded
    common_non_numeric = [
        'Instrument Serial Number', 'Notes', 'Timestamp', 'Integration Time', 
        'wetlab ID', 'Lab'
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
        verbose=args.verbose
    )
    
    # Extract processed data
    X = data['X']
    y = data['y']
    
    if y is None or len(y) == 0:
        raise ValueError(f"Target column '{args.target}' not found or empty after preprocessing")
    
    logger.info(f"Preprocessed features shape: {X.shape}")
    logger.info(f"Target values shape: {y.shape}")
    
    # Split data into train and test sets
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42
    )
    
    logger.info(f"Training set shape: {X_train.shape}")
    logger.info(f"Test set shape: {X_test.shape}")
    
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
            model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1
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
    model_filename = f"{args.model}_{args.target}_{timestamp}.pkl"
    model_path = os.path.join(args.output_dir, model_filename)
    
    logger.info(f"Saving model to {model_path}")
    
    # Save model and metadata
    model_data = {
        'model': model,
        'preprocessing_pipeline': data['preprocessing_pipeline'],
        'spectral_columns': data['spectral_columns'],
        'feature_names': list(X.columns),
        'target': args.target,
        'metrics': metrics,
        'transform_type': args.transform,
        'savgol': args.savgol,
        'training_date': timestamp
    }
    
    joblib.dump(model_data, model_path)
    logger.info("Model training completed successfully")

if __name__ == "__main__":
    main() 