"""
Visualization functions for NIR tomato spectroscopy data.

This module contains functions for visualizing NIR spectral data and analysis results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple, Union
from scipy import stats
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.2)

def plot_raw_spectra(
    spectra: pd.DataFrame,
    wavelengths: Optional[List[float]] = None,
    sample_ids: Optional[List[str]] = None,
    n_samples: int = 10,
    title: str = 'Raw NIR Spectra',
    figsize: Tuple[int, int] = (12, 6),
    color: str = 'viridis',
    alpha: float = 0.7,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot raw NIR spectra.
    
    Args:
        spectra: DataFrame with spectral data (samples as rows, wavelengths as columns)
        wavelengths: List of wavelength values (if None, use column names)
        sample_ids: List of sample IDs to highlight (if None, randomly select n_samples)
        n_samples: Number of samples to plot if sample_ids is None
        title: Plot title
        figsize: Figure size (width, height)
        color: Color or colormap name
        alpha: Transparency of plotted lines
        save_path: Path to save the plot (optional)
            
    Returns:
        Matplotlib figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get wavelengths if not provided
    if wavelengths is None:
        # Try to convert column names to float
        try:
            wavelengths = [float(col) for col in spectra.columns]
        except ValueError:
            # If conversion fails, use column indices
            wavelengths = list(range(len(spectra.columns)))
    
    # Select samples to plot
    if sample_ids is None:
        # Randomly select samples
        if n_samples >= len(spectra):
            samples_to_plot = spectra
        else:
            samples_to_plot = spectra.sample(n=n_samples, random_state=42)
    else:
        # Use provided sample IDs
        samples_to_plot = spectra.loc[sample_ids]
    
    # Set up colormap if using viridis or other colormaps
    if color in plt.colormaps():
        cmap = plt.get_cmap(color)
        colors = [cmap(i) for i in np.linspace(0, 1, len(samples_to_plot))]
    else:
        colors = [color] * len(samples_to_plot)
    
    # Plot each spectrum
    for i, (idx, spectrum) in enumerate(samples_to_plot.iterrows()):
        ax.plot(wavelengths, spectrum, color=colors[i], alpha=alpha, label=f'Sample {idx}')
    
    # Add axis labels and title
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Absorbance')
    ax.set_title(title)
    
    # Add legend if not too many samples
    if len(samples_to_plot) <= 10:
        ax.legend(loc='best')
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_mean_spectrum(
    spectra: pd.DataFrame,
    wavelengths: Optional[List[float]] = None,
    with_std: bool = True,
    title: str = 'Mean NIR Spectrum',
    figsize: Tuple[int, int] = (12, 6),
    color: str = 'blue',
    std_color: str = 'lightblue',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot mean NIR spectrum with optional standard deviation.
    
    Args:
        spectra: DataFrame with spectral data (samples as rows, wavelengths as columns)
        wavelengths: List of wavelength values (if None, use column names)
        with_std: Whether to show standard deviation
        title: Plot title
        figsize: Figure size (width, height)
        color: Color for mean spectrum
        std_color: Color for standard deviation area
        save_path: Path to save the plot (optional)
            
    Returns:
        Matplotlib figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get wavelengths if not provided
    if wavelengths is None:
        # Try to convert column names to float
        try:
            wavelengths = [float(col) for col in spectra.columns]
        except ValueError:
            # If conversion fails, use column indices
            wavelengths = list(range(len(spectra.columns)))
    
    # Calculate mean and standard deviation
    mean_spectrum = spectra.mean(axis=0)
    std_spectrum = spectra.std(axis=0)
    
    # Plot mean spectrum
    ax.plot(wavelengths, mean_spectrum, color=color, linewidth=2, label='Mean Spectrum')
    
    # Add standard deviation area if requested
    if with_std:
        ax.fill_between(
            wavelengths,
            mean_spectrum - std_spectrum,
            mean_spectrum + std_spectrum,
            color=std_color,
            alpha=0.3,
            label='± 1 Std Dev'
        )
    
    # Add axis labels and title
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Absorbance')
    ax.set_title(title)
    
    # Add legend
    ax.legend(loc='best')
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_preprocessed_spectra(
    raw_spectra: pd.DataFrame,
    preprocessed_spectra: pd.DataFrame,
    wavelengths: Optional[List[float]] = None,
    sample_ids: Optional[List[str]] = None,
    n_samples: int = 5,
    preprocessing_method: str = 'Unknown',
    figsize: Tuple[int, int] = (14, 8),
    cmap: str = 'viridis',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot raw and preprocessed NIR spectra for comparison.
    
    Args:
        raw_spectra: DataFrame with raw spectral data
        preprocessed_spectra: DataFrame with preprocessed spectral data
        wavelengths: List of wavelength values (if None, use column names)
        sample_ids: List of sample IDs to highlight (if None, randomly select n_samples)
        n_samples: Number of samples to plot if sample_ids is None
        preprocessing_method: Name of preprocessing method used
        figsize: Figure size (width, height)
        cmap: Colormap name
        save_path: Path to save the plot (optional)
            
    Returns:
        Matplotlib figure
    """
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Get wavelengths if not provided
    if wavelengths is None:
        # Try to convert column names to float
        try:
            wavelengths = [float(col) for col in raw_spectra.columns]
        except ValueError:
            # If conversion fails, use column indices
            wavelengths = list(range(len(raw_spectra.columns)))
    
    # Select samples to plot
    if sample_ids is None:
        # Randomly select samples
        if n_samples >= len(raw_spectra):
            indices = raw_spectra.index
        else:
            indices = raw_spectra.sample(n=n_samples, random_state=42).index
    else:
        # Use provided sample IDs
        indices = [idx for idx in sample_ids if idx in raw_spectra.index and idx in preprocessed_spectra.index]
    
    # Set up colormap
    cmap_obj = plt.get_cmap(cmap)
    colors = [cmap_obj(i) for i in np.linspace(0, 1, len(indices))]
    
    # Plot raw spectra
    for i, idx in enumerate(indices):
        if idx in raw_spectra.index:
            axes[0].plot(wavelengths, raw_spectra.loc[idx], color=colors[i], label=f'Sample {idx}')
    
    axes[0].set_title('Raw NIR Spectra')
    axes[0].set_ylabel('Absorbance')
    
    # Plot preprocessed spectra
    for i, idx in enumerate(indices):
        if idx in preprocessed_spectra.index:
            axes[1].plot(wavelengths, preprocessed_spectra.loc[idx], color=colors[i], label=f'Sample {idx}')
    
    axes[1].set_title(f'Preprocessed NIR Spectra ({preprocessing_method})')
    axes[1].set_xlabel('Wavelength (nm)')
    axes[1].set_ylabel('Absorbance')
    
    # Add legend if not too many samples
    if len(indices) <= 10:
        axes[1].legend(loc='best')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_brix_distribution(
    brix_values: pd.Series,
    bins: int = 20,
    kde: bool = True,
    title: str = 'Brix Distribution',
    figsize: Tuple[int, int] = (10, 6),
    color: str = 'skyblue',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot the distribution of Brix values.
    
    Args:
        brix_values: Series with Brix values
        bins: Number of histogram bins
        kde: Whether to show kernel density estimate
        title: Plot title
        figsize: Figure size (width, height)
        color: Histogram color
        save_path: Path to save the plot (optional)
            
    Returns:
        Matplotlib figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot histogram with KDE
    sns.histplot(brix_values, bins=bins, kde=kde, color=color, ax=ax)
    
    # Add statistics as text
    stats_text = (
        f"Mean: {brix_values.mean():.2f}\n"
        f"Median: {brix_values.median():.2f}\n"
        f"Std Dev: {brix_values.std():.2f}\n"
        f"Min: {brix_values.min():.2f}\n"
        f"Max: {brix_values.max():.2f}\n"
        f"Samples: {len(brix_values)}"
    )
    
    ax.text(0.95, 0.95, stats_text,
            transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add axis labels and title
    ax.set_xlabel('Brix Value')
    ax.set_ylabel('Count')
    ax.set_title(title)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_correlation_heatmap(
    spectra: pd.DataFrame,
    target: pd.Series,
    wavelengths: Optional[List[float]] = None,
    method: str = 'pearson',
    cmap: str = 'coolwarm',
    figsize: Tuple[int, int] = (12, 6),
    title: str = 'Correlation Between Wavelengths and Brix',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot correlation heatmap between spectral data and target values.
    
    Args:
        spectra: DataFrame with spectral data
        target: Series with target values (Brix)
        wavelengths: List of wavelength values (if None, use column names)
        method: Correlation method ('pearson', 'spearman', or 'kendall')
        cmap: Colormap for heatmap
        figsize: Figure size (width, height)
        title: Plot title
        save_path: Path to save the plot (optional)
            
    Returns:
        Matplotlib figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get wavelengths if not provided
    if wavelengths is None:
        # Try to convert column names to float
        try:
            wavelengths = [float(col) for col in spectra.columns]
        except ValueError:
            # If conversion fails, use column indices
            wavelengths = list(range(len(spectra.columns)))
    
    # Calculate correlations
    correlations = []
    for col in spectra.columns:
        if method == 'pearson':
            corr, _ = stats.pearsonr(spectra[col], target)
        elif method == 'spearman':
            corr, _ = stats.spearmanr(spectra[col], target)
        elif method == 'kendall':
            corr, _ = stats.kendalltau(spectra[col], target)
        else:
            raise ValueError(f"Unknown correlation method: {method}")
        correlations.append(corr)
    
    # Plot correlation vs wavelength
    scatter = ax.scatter(wavelengths, correlations, c=correlations, cmap=cmap, alpha=0.7)
    ax.plot(wavelengths, correlations, 'k-', alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(f'{method.capitalize()} Correlation')
    
    # Add axis labels and title
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel(f'{method.capitalize()} Correlation with Brix')
    ax.set_title(title)
    
    # Add horizontal line at zero
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_pca_components(
    spectra: pd.DataFrame,
    n_components: int = 3,
    standardize: bool = True,
    wavelengths: Optional[List[float]] = None,
    figsize: Tuple[int, int] = (12, 8),
    colors: List[str] = ['blue', 'red', 'green', 'purple', 'orange'],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot PCA components of spectral data.
    
    Args:
        spectra: DataFrame with spectral data
        n_components: Number of PCA components to plot
        standardize: Whether to standardize the data before PCA
        wavelengths: List of wavelength values (if None, use column names)
        figsize: Figure size (width, height)
        colors: List of colors for PCA components
        save_path: Path to save the plot (optional)
            
    Returns:
        Matplotlib figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get wavelengths if not provided
    if wavelengths is None:
        # Try to convert column names to float
        try:
            wavelengths = [float(col) for col in spectra.columns]
        except ValueError:
            # If conversion fails, use column indices
            wavelengths = list(range(len(spectra.columns)))
    
    # Prepare data for PCA
    X = spectra.values
    if standardize:
        X = StandardScaler().fit_transform(X)
    
    # Perform PCA
    pca = PCA(n_components=min(n_components, min(X.shape)))
    pca.fit(X)
    
    # Plot each component
    for i in range(min(n_components, len(pca.components_))):
        color = colors[i % len(colors)]
        ax.plot(wavelengths, pca.components_[i], color=color, 
                label=f'PC{i+1} ({pca.explained_variance_ratio_[i]*100:.1f}%)')
    
    # Add axis labels and title
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Loading')
    ax.set_title(f'PCA Components (Total Variance Explained: {sum(pca.explained_variance_ratio_[:n_components])*100:.1f}%)')
    
    # Add legend
    ax.legend(loc='best')
    
    # Add horizontal line at zero
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_spectra_by_brix(
    spectra: pd.DataFrame,
    brix_values: pd.Series,
    wavelengths: Optional[List[float]] = None,
    n_groups: int = 3,
    figsize: Tuple[int, int] = (14, 8),
    cmap: str = 'viridis',
    alpha: float = 0.7,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot spectra colored by Brix value groups.
    
    Args:
        spectra: DataFrame with spectral data
        brix_values: Series with Brix values
        wavelengths: List of wavelength values (if None, use column names)
        n_groups: Number of Brix groups to create
        figsize: Figure size (width, height)
        cmap: Colormap name
        alpha: Transparency of lines
        save_path: Path to save the plot (optional)
            
    Returns:
        Matplotlib figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get wavelengths if not provided
    if wavelengths is None:
        # Try to convert column names to float
        try:
            wavelengths = [float(col) for col in spectra.columns]
        except ValueError:
            # If conversion fails, use column indices
            wavelengths = list(range(len(spectra.columns)))
    
    # Create Brix groups
    brix_bins = pd.qcut(brix_values, n_groups, labels=False)
    
    # Set up colormap
    cmap_obj = plt.get_cmap(cmap)
    norm = Normalize(vmin=brix_values.min(), vmax=brix_values.max())
    
    # Plot spectra for each sample, colored by Brix value
    for idx, spectrum in spectra.iterrows():
        if idx in brix_values.index:
            brix = brix_values.loc[idx]
            color = cmap_obj(norm(brix))
            ax.plot(wavelengths, spectrum, color=color, alpha=alpha)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Brix Value')
    
    # Add axis labels and title
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Absorbance')
    ax.set_title('NIR Spectra Colored by Brix Value')
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig 