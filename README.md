# 🍅 NIRS - NIR Spectroscopy Analysis for Tomatoes

> Analysis of tomato quality using Near-Infrared Spectroscopy (NIR)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![MLflow](https://img.shields.io/badge/MLflow-2.0%2B-brightgreen)](https://mlflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<p align="center">
  <img src="https://github.com/user/NIRS/raw/main/images/tomato_nir_banner.png" alt="Tomato NIR Spectroscopy" width="600">
</p>

## 📑 Table of Contents

- [✨ Features](#-features)
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
  - [Setting up Remote Tracking](#setting-up-remote-tracking)
  - [Programmatic Usage of MLflow API](#programmatic-usage-of-mlflow-api)
  - [Structure of Experiment Tracking](#structure-of-experiment-tracking)
  - [MLflow Server Setup](#mlflow-server-setup)
  - [Security for Remote Access](#security-for-remote-access)
- [📂 Project Structure](#-project-structure)

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

- 📊 **Evaluation and visualization of results**

- 📈 **Experiment tracking with MLflow**:
  - Parameter logging
  - Metrics tracking
  - Model artifacts storage
  - Feature importance visualization

## 🚀 Installation

Clone this repository and install the package using pip:

```bash
git clone https://github.com/username/NIRS.git
cd NIRS
pip install -e ".[dev]"
```

## 📊 Usage

### Data Processing

The package provides tools for data processing, including transformers for spectral data, utilities for data cleaning, and pipelines for complete data processing workflows.

```python
from nirs.data_processing.transformers import SNVTransformer
from nirs.data_processing.pipeline import preprocess_spectra
import pandas as pd

# Load data
df = pd.read_csv('data/tomato_spectra.csv')

# Process data
results = preprocess_spectra(
    df=df,
    target_column='SSC',
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
python scripts/train_model.py --data data/tomato_spectra.csv --target SSC --model pls --transform snv

# Train an XGBoost model with MSC transformation and Savitzky-Golay filtering
python scripts/train_model.py --data data/tomato_spectra.csv --target SSC --model xgb --transform msc --savgol --window_length 15 --polyorder 2 --tune_hyperparams

# Train a Random Forest model excluding specific columns
python scripts/train_model.py --data data/tomato_spectra.csv --target SSC --model rf --transform snv --exclude_columns "Notes" "Timestamp" "Instrument Serial Number"
```

> **Note**: The script automatically filters out common non-numeric columns that can't be used for modeling.

#### Using Python functions

```python
from nirs.modeling.models import create_pls_model
from nirs.modeling.evaluation import evaluate_regression_model, print_regression_metrics
from sklearn.model_selection import train_test_split

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create model
model, param_grid = create_pls_model()

# Train model
model.fit(X_train, y_train)

# Evaluate model
metrics, y_pred = evaluate_regression_model(model, X_test, y_test)
print_regression_metrics(metrics)
```

### Running Experiments

The package provides a script for running multiple experiments with different models and preprocessing methods:

```bash
# Run a standard set of experiments
python scripts/run_experiments.py --data data/tomato_spectra.csv

# Run experiments with feature selection methods
python scripts/run_experiments.py --data data/tomato_spectra.csv --feature_selection

# Track experiments with MLflow
python scripts/run_experiments.py --data data/tomato_spectra.csv --use_mlflow
```

### Data Visualization

The package provides tools for visualizing spectral data and model results.

```python
from nirs.visualization.spectra import plot_spectra, plot_transformed_spectra
import matplotlib.pyplot as plt

# Plot raw spectra
plot_spectra(df, title='Raw Tomato NIR Spectra')
plt.show()

# Plot transformed spectra
plot_transformed_spectra(X, title='Transformed Tomato NIR Spectra')
plt.show()
```

## 📈 Experiment Tracking with MLflow

The project integrates MLflow for experiment tracking, allowing you to monitor and compare model performance, hyperparameters, and feature importance.

### Setup and Installation

MLflow is already added to `requirements.txt`, so you just need to update the dependencies:

```bash
pip install -r requirements.txt
```

Or update the package installation in development mode:

```bash
pip install -e ".[dev]"
```

### Running Experiments with MLflow

To track experiments through MLflow, you can use the `run_experiments.py` script:

```bash
# Run experiments with MLflow tracking
python scripts/run_experiments.py --data data/raw/Tomato_Viavi_Brix_model_pulp.csv --use_mlflow

# Run experiments with feature selection and MLflow tracking
python scripts/run_experiments.py --data data/raw/Tomato_Viavi_Brix_model_pulp.csv --feature_selection --use_mlflow
```

### Viewing Results via MLflow UI

To launch a local MLflow server with interface:

```bash
python scripts/run_mlflow_server.py --host 127.0.0.1 --port 5000
```

Then open in browser: http://127.0.0.1:5000

#### MLflow Interface Screenshots

<p align="center">
  <img src="images/mlflow_experiments.png" alt="MLflow Experiments Overview" width="800"><br>
  <em>MLflow Experiments overview with comparison of multiple models</em>
</p>

<p align="center">
  <img src="images/mlflow_experiment_details.png" alt="MLflow Experiment Details" width="800"><br>
  <em>Detailed view of a specific experiment showing hyperparameters and metrics</em>
</p>

### Setting up Remote Tracking

For collaborative use of results, MLflow can be configured to store data in a remote storage (S3, MinIO, etc.).

#### Configuration with S3 (AWS)

1. Set up necessary AWS credentials:

```bash
export AWS_ACCESS_KEY_ID=<your_access_key>
export AWS_SECRET_ACCESS_KEY=<your_secret_key>
export AWS_DEFAULT_REGION=<aws_region>
```

2. Launch MLflow server with S3 storage:

```bash
python scripts/run_mlflow_server.py --host 0.0.0.0 --port 5000 \
    --backend-store-uri sqlite:///mlflow.db \
    --artifacts-uri s3://bucket-name/artifacts-path
```

3. Training models with remote tracking:

```bash
python scripts/run_experiments.py --data <path_to_data> --use_mlflow \
    --tracking_uri http://server-address:5000
```

#### Configuration with MinIO (S3-compatible storage)

1. Install MinIO Server (locally or on a server):
   - Follow instructions at: https://min.io/docs/minio/container/index.html

2. Set up credentials:

```bash
export AWS_ACCESS_KEY_ID=<minio_access_key>
export AWS_SECRET_ACCESS_KEY=<minio_secret_key>
```

3. Launch MLflow server with MinIO:

```bash
python scripts/run_mlflow_server.py --host 0.0.0.0 --port 5000 \
    --backend-store-uri sqlite:///mlflow.db \
    --artifacts-uri s3://mlflow \
    --endpoint-url http://minio-address:9000
```

### Programmatic Usage of MLflow API

The project includes a module `src.modeling.tracking` with functions for working with MLflow:

```python
from src.modeling.tracking import start_run, log_parameters, log_metrics, log_model, end_run

# Start a new experiment
start_run(run_name="experiment_name")

# Log parameters
log_parameters({"param1": value1, "param2": value2})

# Log metrics
log_metrics({"rmse": 0.123, "mae": 0.089, "r2": 0.95})

# Log model
log_model(trained_model, "model")

# End experiment
end_run()
```

### Structure of Experiment Tracking

When running a series of experiments through `run_experiments.py` with the `--use_mlflow` flag:

1. **Parent Experiment** - a general experiment is created for the entire series with basic parameters
2. **Individual Experiments** - each model/preprocessing variant runs as a separate experiment
3. **Final Experiment** - saves information about the best model from the entire series

This allows:
- Seeing the full picture in one place
- Comparing all experiments
- Quickly finding the best models

### MLflow Server Setup

For long-term use, it is recommended to configure the MLflow server with:

1. A database for metadata (e.g., PostgreSQL)
2. Reliable storage for artifacts (S3, MinIO)
3. A proxy server (e.g., nginx) with SSL encryption

Example of launching with PostgreSQL:

```bash
python scripts/run_mlflow_server.py --host 0.0.0.0 --port 5000 \
    --backend-store-uri postgresql://username:password@host:port/database \
    --artifacts-uri s3://bucket-name/artifacts-path
```

### Security for Remote Access

If you're opening MLflow for internet access, be sure to:

1. Set up HTTPS via nginx or another proxy server
2. Add basic authentication
3. Restrict access by IP if possible
4. Monitor MLflow updates to address vulnerabilities

## 📂 Project Structure

```
NIRS/
├── data/                  # Data directory
├── models/                # Saved models
├── scripts/               # Command-line scripts
├── src/                   # Source code
│   ├── data_processing/   # Data processing modules
│   ├── modeling/          # Modeling modules
│   │   └── tracking.py    # MLflow integration module
│   └── visualization/     # Visualization modules
├── tests/                 # Unit tests
├── images/                # Images for documentation
├── README.md              # This file
├── setup.py               # Package installation script
└── requirements.txt       # Package dependencies
```