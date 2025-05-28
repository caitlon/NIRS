#!/usr/bin/env python
"""
Process NIR tomato spectroscopy data.

This script processes raw NIR spectroscopy data, applying transformations
and preprocessing steps, and saves the processed data for later use.

Example:
    $ python process_data.py --input data/raw/tomato_spectra.csv --output data/processed/
                            --transform snv --savgol
                            
    # With feature selection:
    $ python process_data.py --input data/raw/tomato_spectra.csv --output data/processed/
                            --transform snv --feature_selection vip --n_features 20
"""

import argparse
import logging
import os
import joblib
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression

from nirs_tomato.data_processing.transformers import SNVTransformer, MSCTransformer, SavGolTransformer
from nirs_tomato.data_processing.pipeline import preprocess_spectra
from nirs_tomato.data_processing.feature_selection import (
    GeneticAlgorithmSelector, 
    CARSSelector, 
    PLSVIPSelector
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
        description="Process NIR tomato spectroscopy data."
    )
    
    # Input/output options
    parser.add_argument(
        "--input", type=str, required=True, 
        help="Path to CSV file containing NIR spectroscopy data"
    )
    parser.add_argument(
        "--output", type=str, default="data/processed/", 
        help="Directory to save processed data"
    )
    parser.add_argument(
        "--target", type=str, default=None, 
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
    
    # Feature selection options
    parser.add_argument(
        "--feature_selection", type=str, choices=["none", "ga", "cars", "vip"], default="none",
        help="Feature selection method to apply (ga=Genetic Algorithm, cars=Competitive Adaptive Reweighted Sampling, vip=Variable Importance in Projection)"
    )
    parser.add_argument(
        "--n_features", type=int, default=20,
        help="Number of features to select when using feature selection"
    )
    parser.add_argument(
        "--plot_selection", action="store_true",
        help="Plot selected features and save to output directory"
    )
    
    # Misc options
    parser.add_argument(
        "--verbose", action="store_true", 
        help="Enable verbose output"
    )
    
    return parser.parse_args()

def main():
    """Main function for processing NIR spectroscopy data."""
    # Parse command line arguments
    args = parse_args()
    
    # Set verbose logging
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Load data
    logger.info(f"Loading data from {args.input}")
    df = pd.read_csv(args.input)
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
    
    # Process data
    logger.info("Processing data")
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
    y = data.get('y', None)
    
    # Apply feature selection if requested
    feature_selector = None
    if args.feature_selection != "none":
        if args.target is None or y is None:
            logger.warning("Feature selection requires a target column. Skipping feature selection.")
        else:
            logger.info(f"Applying {args.feature_selection.upper()} feature selection")
            
            # Identify spectral columns
            from nirs_tomato.data_processing.utils import identify_spectral_columns
            spectral_cols, non_spectral_cols = identify_spectral_columns(df)
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
                    random_state=42
                )
            elif args.feature_selection == "cars":
                feature_selector = CARSSelector(
                    n_pls_components=10,
                    n_sampling_runs=30,
                    n_features_to_select=args.n_features,
                    wavelengths=wavelengths,
                    random_state=42
                )
            elif args.feature_selection == "vip":
                feature_selector = PLSVIPSelector(
                    n_components=10,
                    n_features_to_select=args.n_features,
                    wavelengths=wavelengths
                )
            
            # Fit and transform data
            feature_selector.fit(X, y)
            X_selected = feature_selector.transform(X)
            
            # Update data for output
            X = X_selected
            
            logger.info(f"Selected {X.shape[1]} features using {args.feature_selection.upper()}")
            
            # Plot selected features if requested
            if args.plot_selection:
                os.makedirs(args.output, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                if args.feature_selection == "ga":
                    plot = feature_selector.plot_selected_wavelengths()
                    plot_path = os.path.join(args.output, f"ga_selected_features_{timestamp}.png")
                    plot.savefig(plot_path, dpi=300, bbox_inches='tight')
                    logger.info(f"Saved feature selection plot to {plot_path}")
                elif args.feature_selection == "cars":
                    # Plot selection history
                    plot_history = feature_selector.plot_selection_history()
                    plot_history_path = os.path.join(args.output, f"cars_selection_history_{timestamp}.png")
                    plot_history.savefig(plot_history_path, dpi=300, bbox_inches='tight')
                    
                    # Plot selected wavelengths
                    plot_wavelengths = feature_selector.plot_selected_wavelengths()
                    plot_wavelengths_path = os.path.join(args.output, f"cars_selected_features_{timestamp}.png")
                    plot_wavelengths.savefig(plot_wavelengths_path, dpi=300, bbox_inches='tight')
                    
                    logger.info(f"Saved CARS plots to {args.output}")
                elif args.feature_selection == "vip":
                    plot = feature_selector.plot_vip_scores()
                    plot_path = os.path.join(args.output, f"vip_scores_{timestamp}.png")
                    plot.savefig(plot_path, dpi=300, bbox_inches='tight')
                    logger.info(f"Saved VIP scores plot to {plot_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Save processed data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    transform_name = args.transform
    if args.savgol:
        transform_name += "_savgol"
    
    # Add feature selection to output filename
    if args.feature_selection != "none":
        transform_name += f"_{args.feature_selection}{args.n_features}"
    
    output_filename = f"processed_tomato_{transform_name}_{timestamp}.joblib"
    output_path = os.path.join(args.output, output_filename)
    
    # Update the data dictionary with feature selection info
    if feature_selector is not None:
        data['feature_selection'] = {
            'method': args.feature_selection,
            'n_features': args.n_features,
            'selector': feature_selector,
            'selected_features_mask': feature_selector.selected_features_mask_ if hasattr(feature_selector, 'selected_features_mask_') else None,
            'selected_features_indices': feature_selector.selected_features_indices_ if hasattr(feature_selector, 'selected_features_indices_') else None
        }
    
    # Update X in the data dictionary
    data['X'] = X
    
    logger.info(f"Saving processed data to {output_path}")
    joblib.dump(data, output_path)
    
    # Also save as CSV for easier inspection
    csv_filename = f"processed_tomato_{transform_name}_{timestamp}.csv"
    csv_path = os.path.join(args.output, csv_filename)
    X.to_csv(csv_path, index=False)
    logger.info(f"Saved processed features as CSV to {csv_path}")
    
    if args.target and data.get('y') is not None:
        logger.info(f"Processed data contains {len(X)} samples with {X.shape[1]} features")
        logger.info(f"Target column: {args.target}")
    else:
        logger.info(f"Processed data contains {len(X)} samples with {X.shape[1]} features (no target specified)")
    
    logger.info("Data processing completed successfully")

if __name__ == "__main__":
    main() 