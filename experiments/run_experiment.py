#!/usr/bin/env python
"""
NIR Spectroscopy Experiment Runner

This script provides a simple command-line interface for running NIR
spectroscopy experiments using the ExperimentManager.

Examples:
    # Run a single experiment from a config file
    python experiments/run_experiment.py --config configs/pls_snv_savgol.yaml
    
    # Run all experiments in a directory
    python experiments/run_experiment.py --config_dir configs/
    
    # Run an experiment with verbose output
    python experiments/run_experiment.py --config configs/rf_msc_feature_selection.yaml --verbose
"""

import argparse
import logging
import sys
import os
from typing import Optional

# Add parent directory to the module search path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from experiments.experiment_manager import ExperimentManager


def main() -> None:
    """Run the experiment based on command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run NIR spectroscopy experiments using configuration files."
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--config", type=str,
        help="Path to YAML configuration file"
    )
    group.add_argument(
        "--config_dir", type=str,
        help="Directory containing YAML configuration files"
    )
    
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Create experiment manager
    manager = ExperimentManager(log_level=log_level)
    
    # Run experiments
    if args.config:
        manager.run_from_config(args.config)
    elif args.config_dir:
        manager.run_from_config_dir(args.config_dir)


if __name__ == "__main__":
    main() 