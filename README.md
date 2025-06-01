# 🍅 NIRS - NIR Spectroscopy Analysis for Tomatoes

> Analysis of tomato quality using Near-Infrared Spectroscopy (NIR)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![MLflow](https://img.shields.io/badge/MLflow-2.11%2B-brightgreen)](https://mlflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/your-username/NIRS/actions/workflows/tests.yml/badge.svg)](https://github.com/your-username/NIRS/actions/workflows/tests.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<p align="center">
  <img src="images/tomato.png" alt="Tomato NIR Spectroscopy" width="600">
</p>

## 📑 Table of Contents

- [✨ Features](#-features)
- [📂 Project Structure](#-project-structure)
- [🚀 Installation](#-installation)
- [📊 Usage](#-usage)
  - [Data Processing](#data-processing)
  - [Model Training](#model-training)
  - [Running Experiments](#running-experiments)
  - [Data Visualization](#data-visualization)
- [📈 Experiment Tracking with MLflow](#-experiment-tracking-with-mlflow)
  - [Setup and Installation](#setup-and-installation)
  - [Running Experiments with MLflow](#running-experiments-with-mlflow)
  - [Viewing Results via MLflow UI](#viewing-results-via-mlflow-ui)
- [🧪 Testing](#-testing)
  - [Running Tests](#running-tests)
  - [Continuous Integration](#continuous-integration)
- [📝 License](#-license)

## ✨ Features

- 🔍 **Pre-processing of NIR spectral data**:
  - Spectral transformations (SNV, MSC)
  - Savitzky-Golay filtering
  - Automatic detection and filtering of non-numeric columns
  - Outlier detection and removal

- 🧠 **Modeling of NIR data**:
  - PLS regression
  - Support Vector Regression (SVR)
  - Random Forest regression
  - XGBoost regression
  - LightGBM regression

- 📊 **Advanced model optimization**:
  - Hyperparameter tuning with Optuna
  - Feature selection methods (Genetic Algorithm, CARS, VIP)
  - Integrated cross-validation

- 📈 **Experiment tracking with MLflow**:
  - Parameter logging
  - Metrics tracking
  - Model artifacts storage
  - Feature importance visualization

- 🧪 **Quality assurance**:
  - Comprehensive test suite
  - CI/CD with GitHub Actions
  - Code quality checks with black, isort, flake8, and mypy

## 📂 Project Structure

```
NIRS/
├── configs/                    # Configuration files for experiments
│   ├── pls_snv_savgol.yaml     # PLS model with SNV and Savitzky-Golay
│   ├── rf_msc_feature_selection.yaml # Random Forest with feature selection
│   ├── xgb_genetic_algorithm.yaml    # XGBoost with genetic algorithm
│   ├── rf_hyperparams_tuning.yaml    # Random Forest with hyperparameter tuning
│   └── README.md               # Documentation for config files
├── data/                       # Data directory
│   ├── raw/                    # Raw input data files
│   └── processed/              # Processed data files
├── experiments/                # Experiment scripts
│   ├── analyze_models.py       # Script for analyzing model performance
│   ├── create_config.py        # Tool for creating experiment configs
│   ├── experiment_manager.py   # Manages experiment execution
│   ├── process_data.py         # Data processing utilities
│   ├── process_tomato_data.py  # Tomato-specific data processing
│   ├── run_experiment.py       # Main experiment runner
│   ├── run_from_config.py      # Run experiments from config files
│   ├── run_experiments.py      # Extended experiment runner
│   ├── run_mlflow_server.py    # MLflow server launcher
│   └── train_model.py          # Model training script
├── images/                     # Images for documentation
├── mlruns/                     # MLflow experiment tracking data
├── models/                     # Saved model files
├── nirs_tomato/                # Main package
│   ├── config.py               # Configuration utilities
│   ├── data_processing/        # Data processing modules
│   │   ├── constants.py        # Constant definitions
│   │   ├── feature_selection.py # Feature selection methods
│   │   ├── feature_selection/  # Feature selection implementations
│   │   ├── pipeline.py         # Pipeline definitions
│   │   ├── pipeline/           # Pipeline implementations
│   │   ├── transformers.py     # Spectral transformers
│   │   └── utils.py            # Utility functions
│   ├── modeling/               # Modeling and evaluation modules
│   │   ├── evaluation.py       # Model evaluation tools
│   │   ├── hyperparameter_tuning.py # Hyperparameter optimization
│   │   ├── model_factory.py    # Model creation factory
│   │   ├── regression_models.py # Regression model implementations
│   │   └── tracking.py         # MLflow experiment tracking
│   └── __init__.py             # Package initialization
├── results/                    # Experiment results and outputs
├── tests/                      # Test files
│   ├── conftest.py             # Test fixtures and configuration
│   ├── test_data_processing/   # Tests for data processing
│   └── test_modeling/          # Tests for modeling
├── docs/                       # Documentation
├── .gitignore                  # Git ignore file
├── .coverage                   # Coverage report
├── pyproject.toml              # Project configuration
└── README.md                   # This file
```

## 🚀 Installation

Clone this repository and install the package using pip:

```bash
git clone https://github.com/your-username/NIRS.git
cd NIRS
pip install -e ".[dev]"
```

## 📊 Usage

### Quick Start

Here's a quick example to get you started with analyzing tomato NIR spectra:

```python
from nirs_tomato.data_processing.transformers import SNVTransformer
from nirs_tomato.data_processing.utils import preprocess_spectra
from nirs_tomato.modeling.model_factory import create_model
from nirs_tomato.modeling.evaluation import evaluate_regression_model
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Load your NIR data
df = pd.read_csv('data/raw/Tomato_Viavi_Brix_model_pulp.csv')

# 2. Process the spectral data
results = preprocess_spectra(
    df=df,
    target_column='Brix',
    transformers=[SNVTransformer()],
    exclude_columns=['Instrument Serial Number', 'Notes', 'Timestamp'],
    remove_outliers=True,
    verbose=True
)

# 3. Get processed features and target
X, y = results['X'], results['y']

# 4. Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Create and train a model
model = create_model(model_type="pls", n_components=10)
model.fit(X_train, y_train)

# 6. Evaluate the model
metrics, y_pred = evaluate_regression_model(model, X_test, y_test)
print(f"Model performance:")
print(f"  R² score: {metrics['r2']:.4f}")
print(f"  RMSE: {metrics['rmse']:.4f}")
print(f"  MAE: {metrics['mae']:.4f}")
```

### Data Processing

The package provides tools for data processing, including transformers for spectral data, utilities for data cleaning, and pipelines for complete data processing workflows.

```python
from nirs_tomato.data_processing.transformers import SNVTransformer
from nirs_tomato.data_processing.utils import preprocess_spectra
import pandas as pd

# Load data
df = pd.read_csv('data/raw/Tomato_Viavi_Brix_model_pulp.csv')

# Process data
results = preprocess_spectra(
    df=df,
    target_column='Brix',
    transformers=[SNVTransformer()],
    exclude_columns=['Instrument Serial Number', 'Notes', 'Timestamp'],
    remove_outliers=True,
    verbose=True
)

# Get processed features and target
X = results['X']
y = results['y']
```

### Model Training

The package provides command-line scripts for model training, as well as Python functions for creating and evaluating models.

#### Using the command-line script

```bash
# Train a PLS model with SNV transformation
python experiments/train_model.py --data data/raw/Tomato_Viavi_Brix_model_pulp.csv --target Brix --model pls --transform snv

# Train an XGBoost model with MSC transformation and Savitzky-Golay filtering
python experiments/train_model.py --data data/raw/Tomato_Viavi_Brix_model_pulp.csv --target Brix --model xgb --transform msc --savgol --window_length 15 --polyorder 2 --tune_hyperparams

# Train a Random Forest model excluding specific columns
python experiments/train_model.py --data data/raw/Tomato_Viavi_Brix_model_pulp.csv --target Brix --model rf --transform snv --exclude_columns "Notes" "Timestamp" "Instrument Serial Number"
```

#### Using Python functions

```python
from nirs_tomato.modeling.model_factory import create_model
from nirs_tomato.modeling.evaluation import evaluate_regression_model
from sklearn.model_selection import train_test_split

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create model
model = create_model(model_type="pls")

# Train model
model.fit(X_train, y_train)

# Evaluate model
metrics, y_pred = evaluate_regression_model(model, X_test, y_test)
print(f"R2 score: {metrics['r2']:.4f}")
```

### Running Experiments

The package provides multiple ways to run experiments:

#### 1. Using Configuration Files (Recommended)

The simplest way to run experiments is using YAML configuration files:

```bash
# Run a single experiment from a config file
python experiments/run_experiment.py --config configs/pls_snv_savgol.yaml

# Run all experiments in the configs directory
python experiments/run_experiment.py --config_dir configs/

# Run an experiment with verbose output
python experiments/run_experiment.py --config configs/rf_msc_feature_selection.yaml --verbose
```

#### 2. Creating Custom Configuration Files

You can create your own experiment configuration files using the `create_config.py` script:

```bash
# Create a new configuration file
python experiments/create_config.py --name my_experiment --data_path data/raw/Tomato_Viavi_Brix_model_pulp.csv --target_column Brix --model rf --transform snv --output configs/my_experiment.yaml
```

#### 3. Programmatic Interface

You can also run experiments programmatically:

```python
from experiments.experiment_manager import ExperimentManager
from nirs_tomato.config import ExperimentConfig

# Create experiment manager
manager = ExperimentManager()

# Run from config file
results = manager.run_from_config("configs/pls_snv_savgol.yaml")

# Or create a config object programmatically
config = ExperimentConfig.from_yaml("configs/pls_snv_savgol.yaml")
config.model.model_type = "rf"  # Change model type
config.data.transform = "msc"   # Change transformation

# Run with modified config
results = manager.run_from_config_object(config)
```

### Data Visualization

The package includes functions for visualizing results:

```python
from nirs_tomato.modeling.regression_models import plot_regression_results

# Plot regression results
fig = plot_regression_results(y_test, y_pred, title="Predicted vs Actual Brix")

# Save the plot
fig.savefig("results/regression_plot.png")
```

## 📈 Experiment Tracking with MLflow

The package integrates with MLflow for experiment tracking, which helps you organize and compare your experiments.

### Setup and Installation

MLflow is included in the project dependencies. To start the MLflow tracking server:

```bash
# Start MLflow server with local storage
python experiments/run_mlflow_server.py --host 0.0.0.0 --port 5000

# Start with custom backend store
python experiments/run_mlflow_server.py --backend-store-uri sqlite:///mlflow.db

# Start with S3 artifact storage
python experiments/run_mlflow_server.py --artifacts-uri s3://my-bucket/mlflow-artifacts --endpoint-url http://localhost:9000
```

### Running Experiments with MLflow

To enable MLflow tracking in your experiments, set `mlflow.enabled: true` in your configuration file:

```yaml
# In your YAML config file
mlflow:
  enabled: true
  experiment_name: nirs-tomato-brix
```

Or enable it programmatically:

```python
from nirs_tomato.modeling.tracking import start_run, log_parameters, log_metrics, log_model, end_run

# Start a run
with start_run(experiment_name="nirs-tomato-test"):
    # Log parameters
    log_parameters({"model_type": "pls", "n_components": 10})
    
    # Train your model
    # ...
    
    # Log metrics
    log_metrics({"r2": 0.95, "rmse": 0.05})
    
    # Log model
    log_model(model, "pls_model")
```

### Viewing Results via MLflow UI

Once your MLflow server is running, you can view your experiments in the MLflow UI:

1. Open your browser and navigate to `http://localhost:5000` (or your custom host:port)
2. Browse experiments by name
3. Compare runs, view metrics, and download models

## 🧪 Testing

### Running Tests

Run the test suite using pytest:

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=nirs_tomato

# Run specific test module
pytest tests/test_data_processing/test_transformers.py
```

### Continuous Integration

This project uses GitHub Actions for continuous integration. The CI pipeline:

1. Runs on multiple Python versions (3.9, 3.10, 3.11)
2. Installs dependencies
3. Runs the test suite
4. Generates and uploads coverage reports

## 📝 License

This project is licensed under the MIT License. See the LICENSE file for details.

---

<p align="center">
  <i>Built with ❤️ for tomato quality analysis using NIR spectroscopy</i>
</p>
