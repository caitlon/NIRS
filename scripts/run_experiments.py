#!/usr/bin/env python
"""
Run multiple model training experiments with different parameters.

This script automates the process of training multiple models with different
hyperparameters and transformations, helping to find the best performing model.

Example:
    $ python run_experiments.py --data data/raw/Tomato_Viavi_Brix_model_pulp.csv
"""

import os
import argparse
import subprocess
import logging
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run multiple model training experiments."
    )
    
    # Data options
    parser.add_argument(
        "--data", type=str, default="data/raw/Tomato_Viavi_Brix_model_pulp.csv", 
        help="Path to CSV file containing NIR spectroscopy data"
    )
    parser.add_argument(
        "--target", type=str, default="Brix", 
        help="Target column name (e.g., 'Brix', 'lycopene')"
    )
    parser.add_argument(
        "--results_dir", type=str, default="results", 
        help="Directory to save experiment results"
    )
    parser.add_argument(
        "--verbose", action="store_true", 
        help="Enable verbose output"
    )
    
    return parser.parse_args()

def run_experiment(
    data_path: str,
    target_column: str,
    model_type: str,
    transform_type: str,
    use_savgol: bool = True,
    tune_hyperparams: bool = False,
    verbose: bool = False
) -> str:
    """
    Run a single experiment by calling train_model.py with specified parameters.
    
    Args:
        data_path: Path to input data file
        target_column: Name of target column
        model_type: Model type (pls, svr, rf, xgb)
        transform_type: Transform type (snv, msc, none)
        use_savgol: Whether to apply Savitzky-Golay filtering
        tune_hyperparams: Whether to tune hyperparameters
        verbose: Whether to use verbose output
        
    Returns:
        Output filename of the trained model
    """
    # Build command
    cmd = [
        "python", "scripts/train_model.py",
        "--data", data_path,
        "--target", target_column,
        "--model", model_type,
        "--transform", transform_type
    ]
    
    # Add optional flags
    if use_savgol:
        cmd.append("--savgol")
    
    if tune_hyperparams:
        cmd.append("--tune_hyperparams")
    
    if verbose:
        cmd.append("--verbose")
    
    # Execute command
    experiment_name = f"{model_type}_{transform_type}"
    if use_savgol:
        experiment_name += "_savgol"
    if tune_hyperparams:
        experiment_name += "_tuned"
    
    logger.info(f"Running experiment: {experiment_name}")
    logger.info(f"Command: {' '.join(cmd)}")
    
    # Run the command and capture output
    process = subprocess.run(cmd, capture_output=True, text=True)
    
    # Check if command was successful
    if process.returncode == 0:
        logger.info(f"Experiment {experiment_name} completed successfully")
        
        # Get the most recent model file with the naming pattern
        model_pattern = f"{model_type}_{target_column}_*.pkl"
        model_files = sorted(
            [f for f in os.listdir("models") if f.startswith(f"{model_type}_{target_column}_")], 
            key=lambda x: os.path.getmtime(os.path.join("models", x)),
            reverse=True
        )
        
        if model_files:
            model_file = os.path.join("models", model_files[0])
            logger.info(f"Found model file: {model_file}")
            return model_file
        else:
            logger.warning(f"No model file found for experiment {experiment_name}")
            return None
    else:
        logger.error(f"Experiment {experiment_name} failed with code {process.returncode}")
        logger.error(f"Error: {process.stderr}")
        return None

def extract_metrics(model_file: str) -> Dict[str, float]:
    """
    Extract metrics from the model output.
    
    Args:
        model_file: Path to saved model file
        
    Returns:
        Dictionary with model metrics
    """
    import joblib
    
    try:
        model_data = joblib.load(model_file)
        # Check saved model format
        if isinstance(model_data, dict) and 'metrics' in model_data:
            return model_data['metrics']
        else:
            # Try to extract metrics from other formats
            # Assume metrics might be in the root of the dictionary
            metrics = {}
            possible_metrics = ['rmse', 'r2', 'mae', 'explained_variance']
            
            for metric in possible_metrics:
                if metric in model_data:
                    metrics[metric] = model_data[metric]
            
            if metrics:
                return metrics
            else:
                logger.warning(f"Could not extract metrics from model file: {model_file}")
                # Return empty values
                return {metric: float('nan') for metric in possible_metrics}
    except Exception as e:
        logger.error(f"Error loading model metrics: {e}")
        return {'rmse': float('nan'), 'r2': float('nan'), 'mae': float('nan'), 'explained_variance': float('nan')}

def main():
    """Main function to run experiments."""
    # Parse command line arguments
    args = parse_args()
    
    # Create results directory if it doesn't exist
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Define experiments to run
    experiments = [
        # XGBoost experiments
        {"model_type": "xgb", "transform_type": "snv", "use_savgol": True, "tune_hyperparams": False},
        {"model_type": "xgb", "transform_type": "snv", "use_savgol": True, "tune_hyperparams": False}, # Removing tune for XGBoost due to errors
        {"model_type": "xgb", "transform_type": "msc", "use_savgol": True, "tune_hyperparams": False},
        
        # Random Forest experiments
        {"model_type": "rf", "transform_type": "snv", "use_savgol": True, "tune_hyperparams": False},
        {"model_type": "rf", "transform_type": "snv", "use_savgol": True, "tune_hyperparams": True},
        {"model_type": "rf", "transform_type": "msc", "use_savgol": True, "tune_hyperparams": False},
        
        # SVR experiments
        {"model_type": "svr", "transform_type": "snv", "use_savgol": True, "tune_hyperparams": False},
        {"model_type": "svr", "transform_type": "msc", "use_savgol": True, "tune_hyperparams": False},
    ]
    
    # Run experiments and collect results
    results = []
    
    for exp in experiments:
        exp_name = f"{exp['model_type']}_{exp['transform_type']}"
        if exp['use_savgol']:
            exp_name += "_savgol"
        if exp['tune_hyperparams']:
            exp_name += "_tuned"
            
        logger.info(f"Starting experiment: {exp_name}")
        
        # Run the experiment
        model_file = run_experiment(
            data_path=args.data,
            target_column=args.target,
            model_type=exp['model_type'],
            transform_type=exp['transform_type'],
            use_savgol=exp['use_savgol'],
            tune_hyperparams=exp['tune_hyperparams'],
            verbose=args.verbose
        )
        
        if model_file:
            # Extract metrics
            metrics = extract_metrics(model_file)
            
            # Add to results
            result = {
                "experiment": exp_name,
                "model_type": exp['model_type'],
                "transform_type": exp['transform_type'],
                "use_savgol": exp['use_savgol'],
                "tune_hyperparams": exp['tune_hyperparams'],
                "model_file": model_file,
                "rmse": metrics.get('rmse', float('nan')),
                "r2": metrics.get('r2', float('nan')),
                "mae": metrics.get('mae', float('nan')),
                "explained_variance": metrics.get('explained_variance', float('nan'))
            }
            
            results.append(result)
            logger.info(f"Experiment {exp_name} completed, RMSE: {result['rmse']:.4f}, R²: {result['r2']:.4f}")
        else:
            logger.warning(f"Experiment {exp_name} failed to produce a model file")
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Save results only if there is data
    if not results_df.empty:
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(args.results_dir, f"experiment_results_{timestamp}.csv")
        results_df.to_csv(results_file, index=False)
        
        # Print summary
        logger.info("\nExperiment Results Summary:")
        logger.info("-" * 80)
        logger.info(f"{'Experiment':<25} {'RMSE':<10} {'R²':<10} {'MAE':<10}")
        logger.info("-" * 80)
        
        # Check if rmse column exists before sorting
        if 'rmse' in results_df.columns and not results_df.empty:
            for _, row in results_df.sort_values('rmse').iterrows():
                logger.info(f"{row['experiment']:<25} {row['rmse']:<10.4f} {row['r2']:<10.4f} {row['mae']:<10.4f}")
            
            # Find best model
            best_model = results_df.loc[results_df['rmse'].idxmin()]
            logger.info("\nBest Model:")
            logger.info(f"Experiment: {best_model['experiment']}")
            logger.info(f"RMSE: {best_model['rmse']:.4f}")
            logger.info(f"R²: {best_model['r2']:.4f}")
            logger.info(f"MAE: {best_model['mae']:.4f}")
            logger.info(f"Model File: {best_model['model_file']}")
            
            logger.info(f"\nResults saved to {results_file}")
        else:
            logger.warning("No valid results to display")
    else:
        logger.warning("No successful experiments to save")

if __name__ == "__main__":
    main() 