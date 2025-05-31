# Experiment Configurations

This directory contains YAML configuration files for reproducible NIR spectroscopy experiments.

## Available Configurations

1. **pls_snv_savgol.yaml** - PLS regression with SNV transformation and Savitzky-Golay filtering
2. **rf_msc_feature_selection.yaml** - Random Forest with MSC transformation and VIP feature selection
3. **xgb_genetic_algorithm.yaml** - XGBoost with SNV transformation and Genetic Algorithm feature selection

## Running Experiments

Use the `run_from_config.py` script to run experiments based on these configurations:

```bash
# Run a single experiment
python experiments/run_from_config.py --config configs/pls_snv_savgol.yaml

# Run all experiments in the configs directory
python experiments/run_from_config.py --config_dir configs/
```

## Configuration Schema

Each YAML configuration file follows this structure:

```yaml
name: experiment_name
description: Experiment description

# Data configuration
data:
  data_path: path/to/data.csv
  target_column: Brix
  transform: snv  # snv, msc, or none
  savgol:
    enabled: true
    window_length: 15
    polyorder: 2
    deriv: 1
  remove_outliers: false

# Feature selection configuration
feature_selection:
  method: none  # none, ga, cars, or vip
  n_features: 20  # number of features to select

# Model configuration
model:
  model_type: pls  # pls, svr, rf, xgb, or lgbm
  tune_hyperparams: false
  test_size: 0.2

# Output configuration
output_dir: models
results_dir: results
verbose: false

# MLflow configuration
mlflow:
  enabled: true
  experiment_name: experiment-name
```

## Creating New Configurations

To create a new configuration:

1. Copy an existing configuration file
2. Modify the parameters as needed
3. Save with a descriptive name
4. Run the experiment with `run_from_config.py`

All parameters are validated automatically based on the Pydantic schema in `nirs_tomato/config.py`.

## Best Practices

- Use clear, descriptive names for configuration files
- Include documentation comments in your YAML files
- Keep experiments organized by target (e.g., Brix, lycopene)
- Commit configuration files to version control to ensure reproducibility
- Use MLflow tracking to compare experiment results 