#!/usr/bin/env python
"""
Simple script to train a regression model on processed NIR tomato data.

Uses the joblib file created by process_data.py which contains both features and target.

Example:
    $ python simple_train.py --data data/processed/processed_tomato_snv_20250528_160744.joblib --model rf
"""

import argparse
import logging
import os
import joblib
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR 
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a regression model on processed NIR tomato data."
    )
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to joblib file containing processed data"
    )
    parser.add_argument(
        "--model", type=str, choices=['rf', 'svr'], default='rf',
        help="Model to use (rf=RandomForest, svr=SupportVectorRegression)"
    )
    parser.add_argument(
        "--test_size", type=float, default=0.2,
        help="Proportion of data to use for testing"
    )
    parser.add_argument(
        "--output_dir", type=str, default="models",
        help="Directory to save trained model"
    )
    parser.add_argument(
        "--verbose", action="store_true", 
        help="Enable verbose output"
    )
    return parser.parse_args()

def print_metrics(y_true, y_pred):
    """Print regression metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print("\nRegression Model Performance:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R²:   {r2:.4f}")
    
    return {'rmse': rmse, 'mae': mae, 'r2': r2}

def main():
    """Main function for training a model."""
    # Parse command line arguments
    args = parse_args()
    
    # Set verbose logging
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    
    # Load data
    logger.info(f"Loading data from {args.data}")
    data = joblib.load(args.data)
    
    # Extract features and target
    X = data['X']
    y = data['y']
    
    logger.info(f"Data loaded successfully. Features shape: {X.shape}")
    logger.info(f"Target shape: {len(y)}")
    
    # Check for columns with all NaN values and drop them
    nan_cols = X.columns[X.isna().all()].tolist()
    if nan_cols:
        logger.info(f"Dropping {len(nan_cols)} columns with all NaN values")
        X = X.drop(columns=nan_cols)
    
    # Handle remaining missing values
    logger.info("Checking for remaining missing values")
    nan_count = X.isna().sum().sum()
    if nan_count > 0:
        logger.info(f"Found {nan_count} missing values. Applying imputation.")
        imputer = SimpleImputer(strategy='mean')
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
    
    logger.info(f"Final feature set shape: {X.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42
    )
    
    logger.info(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")
    
    # Create and train model
    if args.model == 'rf':
        logger.info("Training Random Forest model")
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif args.model == 'svr':
        logger.info("Training Support Vector Regression model")
        model = SVR(kernel='rbf', C=1.0, gamma='scale')
    else:
        raise ValueError(f"Unsupported model: {args.model}")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Evaluate model
    logger.info("Evaluating model on test set")
    y_pred = model.predict(X_test)
    metrics = print_metrics(y_test, y_pred)
    
    # Save model
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"{args.model}_model_{timestamp}.joblib"
    model_path = os.path.join(args.output_dir, model_filename)
    
    model_data = {
        'model': model,
        'metrics': metrics,
        'feature_names': list(X.columns)
    }
    
    logger.info(f"Saving model to {model_path}")
    joblib.dump(model_data, model_path)
    logger.info("Model training completed successfully")

if __name__ == "__main__":
    main() 