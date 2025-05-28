#!/usr/bin/env python
"""
Run multiple model training experiments with different parameters.

This script automates the process of training multiple models with different
hyperparameters and transformations, helping to find the best performing model.

Example:
    $ python run_experiments.py --data data/raw/Tomato_Viavi_Brix_model_pulp.csv
    
    # With feature selection methods:
    $ python run_experiments.py --data data/raw/Tomato_Viavi_Brix_model_pulp.csv --feature_selection
    
    # With MLflow tracking:
    $ python run_experiments.py --data data/raw/Tomato_Viavi_Brix_model_pulp.csv --use_mlflow
    
    # With remote MLflow tracking:
    $ python run_experiments.py --data data/raw/Tomato_Viavi_Brix_model_pulp.csv --use_mlflow --tracking_uri http://server:5000
"""

import os
import argparse
import subprocess
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional

from nirs_tomato.modeling.tracking import (
    start_run, 
    log_parameters, 
    log_metrics, 
    log_model, 
    log_figure,
    log_artifact,
    end_run
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
    parser.add_argument(
        "--feature_selection", action="store_true",
        help="Run experiments with feature selection methods"
    )
    
    # MLflow options
    parser.add_argument(
        "--use_mlflow", action="store_true",
        help="Track experiments with MLflow"
    )
    parser.add_argument(
        "--tracking_uri", type=str, default=None,
        help="MLflow tracking URI (for remote tracking)"
    )
    parser.add_argument(
        "--experiment_name", type=str, default="nirs-tomato",
        help="MLflow experiment name"
    )
    
    return parser.parse_args()

def run_experiment(
    data_path: str,
    target_column: str,
    model_type: str,
    transform_type: str,
    use_savgol: bool = True,
    tune_hyperparams: bool = False,
    feature_selection_method: Optional[str] = None,
    n_features: int = 20,
    verbose: bool = False,
    use_mlflow: bool = False,
    tracking_uri: Optional[str] = None,
    experiment_name: str = "nirs-tomato"
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
        feature_selection_method: Method for feature selection (ga, cars, vip, None)
        n_features: Number of features to select
        verbose: Whether to use verbose output
        use_mlflow: Whether to track with MLflow
        tracking_uri: MLflow tracking URI
        experiment_name: MLflow experiment name
        
    Returns:
        Output filename of the trained model
    """
    # Build command
    cmd = [
        "python", "experiments/train_model.py",
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
    
    if feature_selection_method:
        cmd.extend(["--feature_selection", feature_selection_method])
        cmd.extend(["--n_features", str(n_features)])
    
    if verbose:
        cmd.append("--verbose")
    
    # Execute command
    experiment_name_str = f"{model_type}_{transform_type}"
    if use_savgol:
        experiment_name_str += "_savgol"
    if tune_hyperparams:
        experiment_name_str += "_tuned"
    if feature_selection_method:
        experiment_name_str += f"_{feature_selection_method}{n_features}"
    
    logger.info(f"Running experiment: {experiment_name_str}")
    logger.info(f"Command: {' '.join(cmd)}")
    
    # Start MLflow run if requested
    if use_mlflow:
        # Create a unique run name
        run_name = f"{experiment_name_str}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Start the run
        start_run(
            run_name=run_name,
            experiment_name=experiment_name,
            tracking_uri=tracking_uri
        )
        
        # Log parameters
        mlflow_params = {
            "data_path": os.path.basename(data_path),
            "target_column": target_column,
            "model_type": model_type,
            "transform_type": transform_type,
            "use_savgol": use_savgol,
            "tune_hyperparams": tune_hyperparams,
            "feature_selection_method": feature_selection_method if feature_selection_method else "none",
            "n_features": n_features if feature_selection_method else 0
        }
        log_parameters(mlflow_params)
    
    # Run the command and capture output
    process = subprocess.run(cmd, capture_output=True, text=True)
    
    # Check if command was successful
    if process.returncode == 0:
        logger.info(f"Experiment {experiment_name_str} completed successfully")
        
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
            
            # Extract metrics and log to MLflow
            if use_mlflow:
                import joblib
                import json
                import tempfile
                
                try:
                    # Load model
                    model_data = joblib.load(model_file)
                    
                    # Extract metrics
                    if isinstance(model_data, dict):
                        if 'metrics' in model_data:
                            metrics = model_data['metrics']
                        else:
                            metrics = {k: v for k, v in model_data.items() 
                                     if k in ['rmse', 'r2', 'mae', 'explained_variance']}
                        
                        # Log metrics
                        log_metrics(metrics)
                        
                        # Log model if available
                        if 'model' in model_data and 'feature_names' in model_data:
                            model = model_data['model']
                            feature_names = model_data['feature_names']
                            
                            # Create input example with correct feature names
                            input_example = pd.DataFrame(np.random.random((1, len(feature_names))), 
                                                      columns=feature_names)
                            
                            # Log model
                            log_model(
                                model, 
                                "model", 
                                registered_model_name=f"nirs-tomato-{model_type}",
                                input_example=input_example
                            )
                            
                            # Log hyperparameters
                            if hasattr(model, 'get_params'):
                                hyperparams = model.get_params()
                                # Add prefix to distinguish from other parameters
                                hyperparams_with_prefix = {f"hyperparams.{k}": v for k, v in hyperparams.items()
                                                        if not isinstance(v, (list, dict, set))}
                                log_parameters(hyperparams_with_prefix)
                            
                            # Log feature importances for tree-based models
                            if hasattr(model, 'feature_importances_'):
                                feature_importance = model.feature_importances_
                                # Log top 10 feature importances as metrics
                                importance_indices = np.argsort(feature_importance)[-10:]
                                for idx in importance_indices:
                                    if idx < len(feature_names):
                                        log_metrics({
                                            f"feature_importance.{feature_names[idx]}": 
                                            float(feature_importance[idx])
                                        })
                                
                                # Create and log feature importance plot
                                import matplotlib.pyplot as plt
                                fig = plt.figure(figsize=(10, 6))
                                sorted_idx = feature_importance.argsort()[-20:]  # Top 20 features
                                plot_feature_names = []
                                for i in sorted_idx:
                                    if i < len(feature_names):
                                        name = feature_names[i]
                                        # Truncate long names
                                        if len(name) > 15:
                                            name = name[:12] + "..."
                                        plot_feature_names.append(name)
                                    else:
                                        plot_feature_names.append(f"feature_{i}")
                                
                                plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
                                plt.yticks(range(len(sorted_idx)), plot_feature_names)
                                plt.title(f'Feature Importance for {model_type}')
                                plt.tight_layout()
                                
                                # Log the figure
                                log_figure(fig, f"artifacts/feature_importance_{model_type}.png")
                            
                            # Log detailed model info as JSON
                            with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
                                model_info = {
                                    "model_type": model_type,
                                    "parameters": {k: str(v) for k, v in model.get_params().items()},
                                    "metrics": metrics,
                                    "features": feature_names,
                                    "data_file": os.path.basename(data_path),
                                    "preprocessing": {
                                        "transform_type": transform_type,
                                        "use_savgol": use_savgol,
                                        "feature_selection_method": feature_selection_method,
                                        "n_features": n_features
                                    }
                                }
                                json.dump(model_info, f, indent=2)
                                temp_file = f.name
                            
                            # Log the JSON file
                            log_artifact(temp_file, "model_details")
                            # Remove temporary file
                            os.unlink(temp_file)
                            
                        elif 'model' in model_data:
                            model = model_data['model']
                            
                            # Log the model
                            log_model(
                                model, 
                                "model", 
                                registered_model_name=f"nirs-tomato-{model_type}"
                            )
                            
                            # Log hyperparameters
                            if hasattr(model, 'get_params'):
                                hyperparams = model.get_params()
                                # Add prefix to distinguish from other parameters
                                hyperparams_with_prefix = {f"hyperparams.{k}": v for k, v in hyperparams.items()
                                                        if not isinstance(v, (list, dict, set))}
                                log_parameters(hyperparams_with_prefix)
                        
                        # End the run
                        end_run()
                except Exception as e:
                    logger.error(f"Error logging to MLflow: {e}")
                    if use_mlflow:
                        end_run()
            
            return model_file
        else:
            logger.warning(f"No model file found for experiment {experiment_name_str}")
            if use_mlflow:
                end_run()
            return None
    else:
        logger.error(f"Experiment {experiment_name_str} failed with code {process.returncode}")
        logger.error(f"Error: {process.stderr}")
        if use_mlflow:
            # Log error
            log_metrics({"error": 1.0})
            end_run()
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
    experiments = []
    
    # Standard experiments without feature selection
    standard_experiments = [
        # XGBoost experiments
        {"model_type": "xgb", "transform_type": "snv", "use_savgol": True, "tune_hyperparams": False},
        {"model_type": "xgb", "transform_type": "msc", "use_savgol": True, "tune_hyperparams": False},
        
        # Random Forest experiments
        {"model_type": "rf", "transform_type": "snv", "use_savgol": True, "tune_hyperparams": False},
        {"model_type": "rf", "transform_type": "snv", "use_savgol": True, "tune_hyperparams": True},
        {"model_type": "rf", "transform_type": "msc", "use_savgol": True, "tune_hyperparams": False},
        
        # SVR experiments
        {"model_type": "svr", "transform_type": "snv", "use_savgol": True, "tune_hyperparams": False},
        {"model_type": "svr", "transform_type": "msc", "use_savgol": True, "tune_hyperparams": False},
        
        # PLS experiments
        {"model_type": "pls", "transform_type": "snv", "use_savgol": True, "tune_hyperparams": True},
        {"model_type": "pls", "transform_type": "msc", "use_savgol": True, "tune_hyperparams": True},
    ]
    
    # Add standard experiments
    experiments.extend(standard_experiments)
    
    # Add feature selection experiments if enabled
    if args.feature_selection:
        feature_selection_experiments = []
        
        # Models to test with feature selection
        models = ["pls", "xgb", "rf"]
        
        # Feature selection methods
        fs_methods = [
            {"method": "vip", "n_features": 20},
            {"method": "vip", "n_features": 50},
            {"method": "cars", "n_features": 20},
            {"method": "cars", "n_features": 50},
            {"method": "ga", "n_features": 20},
            {"method": "ga", "n_features": 50},
        ]
        
        # Create experiments with feature selection
        for model in models:
            for fs in fs_methods:
                feature_selection_experiments.append({
                    "model_type": model,
                    "transform_type": "snv",
                    "use_savgol": True,
                    "tune_hyperparams": False,
                    "feature_selection_method": fs["method"],
                    "n_features": fs["n_features"]
                })
        
        # Add feature selection experiments
        experiments.extend(feature_selection_experiments)
    
    # Run experiments and collect results
    results = []
    
    # If using MLflow, start a parent run for the entire experiment set
    parent_run_id = None
    if args.use_mlflow:
        parent_run_name = f"experiment_set_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        parent_run_id = start_run(
            run_name=parent_run_name,
            experiment_name=args.experiment_name,
            tracking_uri=args.tracking_uri
        )
        log_parameters({
            "num_experiments": len(experiments),
            "data_file": os.path.basename(args.data),
            "target_column": args.target,
            "feature_selection_enabled": args.feature_selection,
        })
        # End parent run - we'll create child runs for each experiment
        end_run()
    
    for exp in experiments:
        exp_name = f"{exp['model_type']}_{exp['transform_type']}"
        if exp['use_savgol']:
            exp_name += "_savgol"
        if exp.get('tune_hyperparams', False):
            exp_name += "_tuned"
        if exp.get('feature_selection_method'):
            exp_name += f"_{exp['feature_selection_method']}{exp['n_features']}"
            
        logger.info(f"Starting experiment: {exp_name}")
        
        # Run the experiment
        model_file = run_experiment(
            data_path=args.data,
            target_column=args.target,
            model_type=exp['model_type'],
            transform_type=exp['transform_type'],
            use_savgol=exp['use_savgol'],
            tune_hyperparams=exp.get('tune_hyperparams', False),
            feature_selection_method=exp.get('feature_selection_method'),
            n_features=exp.get('n_features', 20),
            verbose=args.verbose,
            use_mlflow=args.use_mlflow,
            tracking_uri=args.tracking_uri,
            experiment_name=args.experiment_name
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
                "tune_hyperparams": exp.get('tune_hyperparams', False),
                "feature_selection_method": exp.get('feature_selection_method', 'none'),
                "n_features": exp.get('n_features', 0),
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
            # Sort by RMSE for display
            sorted_results = results_df.sort_values('rmse')
            for _, row in sorted_results.iterrows():
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
            
            # Log best model to MLflow if enabled
            if args.use_mlflow:
                # Start a final run for the summary
                summary_run_name = f"experiment_summary_{timestamp}"
                start_run(
                    run_name=summary_run_name,
                    experiment_name=args.experiment_name,
                    tracking_uri=args.tracking_uri
                )
                
                # Log summary metrics
                log_metrics({
                    "best_rmse": float(best_model['rmse']),
                    "best_r2": float(best_model['r2']),
                    "best_mae": float(best_model['mae']),
                    "num_experiments": len(results),
                    "successful_experiments": len(results)
                })
                
                # Log parameters about best model
                log_parameters({
                    "best_model_type": best_model['model_type'],
                    "best_transform_type": best_model['transform_type'],
                    "best_use_savgol": bool(best_model['use_savgol']),
                    "best_tune_hyperparams": bool(best_model['tune_hyperparams']),
                    "best_feature_selection": best_model['feature_selection_method'],
                    "best_n_features": int(best_model['n_features']),
                    "results_file": results_file
                })
                
                # Log the results CSV as an artifact
                log_artifact(results_file)
                
                # End the summary run
                end_run()
        else:
            logger.warning("No valid results to display")
    else:
        logger.warning("No successful experiments to save")

if __name__ == "__main__":
    main() 