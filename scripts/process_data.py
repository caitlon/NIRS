#!/usr/bin/env python
"""
Process NIR tomato spectroscopy data.

This script processes raw NIR spectroscopy data, applying transformations
and preprocessing steps, and saves the processed data for later use.

Example:
    $ python process_data.py --input data/raw/tomato_spectra.csv --output data/processed/
                            --transform snv --savgol
"""

import argparse
import logging
import os
import joblib
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any

import pandas as pd

from src.data_processing.transformers import SNVTransformer, MSCTransformer, SavGolTransformer
from src.data_processing.pipeline import preprocess_spectra

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
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Save processed data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    transform_name = args.transform
    if args.savgol:
        transform_name += "_savgol"
    
    output_filename = f"processed_tomato_{transform_name}_{timestamp}.joblib"
    output_path = os.path.join(args.output, output_filename)
    
    logger.info(f"Saving processed data to {output_path}")
    joblib.dump(data, output_path)
    
    # Also save as CSV for easier inspection
    csv_filename = f"processed_tomato_{transform_name}_{timestamp}.csv"
    csv_path = os.path.join(args.output, csv_filename)
    X.to_csv(csv_path, index=False)
    logger.info(f"Saved processed features as CSV to {csv_path}")
    
    if args.target and data['y'] is not None:
        logger.info(f"Processed data contains {len(X)} samples with {X.shape[1]} features")
        logger.info(f"Target column: {args.target}")
    else:
        logger.info(f"Processed data contains {len(X)} samples with {X.shape[1]} features (no target specified)")
    
    logger.info("Data processing completed successfully")

if __name__ == "__main__":
    main() 