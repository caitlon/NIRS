"""
NIR Preprocessing Module

This module contains preprocessing functions and transformer classes
for NIR spectral data analysis
"""

import numpy as np
import pandas as pd
from scipy import signal
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline


class SNVTransformer(BaseEstimator, TransformerMixin):
    """
    Standard Normal Variate transformation for spectral data.
    
    For each spectrum (row), this subtracts the mean and divides by the standard deviation.
    This helps reduce scatter effects and baseline variations.
    """
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Convert to numpy array if DataFrame
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X.copy()
        
        # Apply SNV: subtract mean and divide by std for each row
        X_snv = np.zeros_like(X_array)
        for i in range(X_array.shape[0]):
            X_snv[i,:] = (X_array[i,:] - np.mean(X_array[i,:])) / np.std(X_array[i,:])
        
        return X_snv


class SavGolTransformer(BaseEstimator, TransformerMixin):
    """
    Savitzky-Golay filter for spectral derivatives.
    
    Applies a polynomial smoothing filter and calculates derivatives.
    Derivatives help enhance spectral features and reduce baseline effects.
    """
    def __init__(self, window_length=15, polyorder=2, deriv=1):
        self.window_length = window_length
        self.polyorder = polyorder
        self.deriv = deriv
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Convert to numpy array if DataFrame
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X.copy()
        
        # Apply Savitzky-Golay filter
        X_sg = np.zeros_like(X_array)
        for i in range(X_array.shape[0]):
            X_sg[i,:] = signal.savgol_filter(X_array[i,:], 
                                          window_length=self.window_length, 
                                          polyorder=self.polyorder, 
                                          deriv=self.deriv)
        return X_sg


class MSCTransformer(BaseEstimator, TransformerMixin):
    """
    Multiplicative Scatter Correction for spectral data.
    
    Corrects for scatter effects by regressing each spectrum against a reference 
    spectrum (usually the mean) and then correcting using the slope and intercept.
    """
    def __init__(self):
        self.mean_spectrum = None
    
    def fit(self, X, y=None):
        # Convert to numpy array if DataFrame
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X.copy()
            
        # Calculate mean spectrum as reference
        self.mean_spectrum = np.mean(X_array, axis=0)
        return self
    
    def transform(self, X):
        # Convert to numpy array if DataFrame
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X.copy()
        
        # Apply MSC correction
        X_msc = np.zeros_like(X_array)
        for i in range(X_array.shape[0]):
            # Linear regression of spectrum against reference
            slope, intercept = np.polyfit(self.mean_spectrum, X_array[i,:], 1)
            # Apply correction
            X_msc[i,:] = (X_array[i,:] - intercept) / slope
        
        return X_msc


def create_preprocessing_pipelines():
    """
    Create preprocessing pipelines for NIR spectral data.
    
    Returns a dictionary of preprocessing pipelines for different methods.
    """
    # Raw data with scaling
    raw_pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])

    # SNV preprocessing
    snv_pipeline = Pipeline([
        ('snv', SNVTransformer())
    ])

    # First derivative preprocessing
    sg1_pipeline = Pipeline([
        ('sg1', SavGolTransformer(window_length=15, polyorder=2, deriv=1))
    ])

    # Second derivative preprocessing
    sg2_pipeline = Pipeline([
        ('sg2', SavGolTransformer(window_length=15, polyorder=2, deriv=2))
    ])

    # MSC preprocessing
    msc_pipeline = Pipeline([
        ('msc', MSCTransformer())
    ])

    # SNV followed by PCA
    snv_pca_pipeline = Pipeline([
        ('snv', SNVTransformer()),
        ('pca', PCA(n_components=10))
    ])

    # First derivative followed by PCA
    sg1_pca_pipeline = Pipeline([
        ('sg1', SavGolTransformer(window_length=15, polyorder=2, deriv=1)),
        ('pca', PCA(n_components=10))
    ])

    # Store all pipelines in a dictionary
    preprocessing_pipelines = {
        'raw': raw_pipeline,
        'snv': snv_pipeline,
        'sg1': sg1_pipeline,
        'sg2': sg2_pipeline,
        'msc': msc_pipeline,
        'snv_pca': snv_pca_pipeline,
        'sg1_pca': sg1_pca_pipeline
    }
    
    return preprocessing_pipelines

def save_processed_data(data, output_path):
    """
    Save the processed dataset to a CSV file.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The processed dataset to save.
    output_path : str
        Path where the processed file will be saved.
    """
    data.to_csv(output_path, index=False)