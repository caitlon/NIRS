#!/usr/bin/env python
"""
Script for processing NIR tomato spectroscopy data.

This script processes raw NIR spectroscopy data and saves the processed data
for later use in modeling.

Example:
    python process_tomato_data.py --input data/raw/Tomato_Viavi_Brix_model_pulp.csv --output data/processed --method snv

"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path to import the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nirs_tomato.data_processing.constants import (
    AVAILABLE_PREPROCESSING_METHODS,
)
from nirs_tomato.data_processing.pipeline import process_and_save_data


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process NIR tomato spectroscopy data."
    )

    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to the input data file (CSV or Excel)",
    )

    parser.add_argument(
        "--output",
        "-o",
        default="data/processed",
        help="Directory where processed data will be saved",
    )

    parser.add_argument(
        "--output-filename",
        "-f",
        default="processed_tomato_nir_data",
        help="Base filename for the output files",
    )

    parser.add_argument(
        "--method",
        "-m",
        choices=AVAILABLE_PREPROCESSING_METHODS,
        default="raw",
        help="Preprocessing method to use",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of data to use for test set",
    )

    parser.add_argument(
        "--val-size",
        type=float,
        default=0.2,
        help="Proportion of data to use for validation set",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    parser.add_argument(
        "--no-outlier-detection",
        action="store_true",
        help="Disable outlier detection",
    )

    parser.add_argument(
        "--format",
        choices=["joblib", "pickle", "csv"],
        default="joblib",
        help="Format to save the processed data",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print verbose output"
    )

    return parser.parse_args()


def main():
    """Main function."""
    # Parse command line arguments
    args = parse_args()

    # Configure logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Log arguments
    logger.info(
        "Processing NIR tomato spectroscopy data with the following parameters:"
    )
    logger.info(f"  Input file: {args.input}")
    logger.info(f"  Output directory: {args.output}")
    logger.info(f"  Output filename: {args.output_filename}")
    logger.info(f"  Preprocessing method: {args.method}")
    logger.info(f"  Test size: {args.test_size}")
    logger.info(f"  Validation size: {args.val_size}")
    logger.info(f"  Random state: {args.random_state}")
    logger.info(f"  Outlier detection: {not args.no_outlier_detection}")
    logger.info(f"  Save format: {args.format}")

    # Process data
    try:
        saved_files = process_and_save_data(
            data_path=args.input,
            output_dir=args.output,
            output_filename=args.output_filename,
            test_size=args.test_size,
            val_size=args.val_size,
            random_state=args.random_state,
            preprocessing_method=args.method,
            outlier_detection=not args.no_outlier_detection,
            save_format=args.format,
            verbose=args.verbose,
        )

        # Log results
        logger.info("Data processing completed successfully.")
        logger.info("Saved files:")
        for file_type, file_path in saved_files.items():
            logger.info(f"  {file_type}: {file_path}")

        # Return success
        return 0

    except Exception as e:
        logger.error(f"Error processing data: {e}")
        import traceback

        logger.error(traceback.format_exc())

        # Return error
        return 1


if __name__ == "__main__":
    sys.exit(main())
