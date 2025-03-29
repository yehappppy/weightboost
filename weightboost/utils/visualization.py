"""
Visualization utilities for the WeightBoost package.

This module contains functions for visualizing evaluation results.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_results(results, noise_level=0):
    """
    Plot results at a specific noise level.
    
    Parameters
    ----------
    results : dict
        Dictionary containing evaluation results for each dataset and noise level.
        
    noise_level : float, default=0
        The noise level to plot results for.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the plot.
    """
    noise_key = f"{int(noise_level*100)}% Noise"
    if noise_key not in results:
        noise_key = "0% Noise"  # Default to no noise
        
    data = results[noise_key]
    datasets = list(data.keys())
    methods = list(data[datasets[0]].keys())
    
    x = np.arange(len(datasets))
    width = 0.15
    
    fig, ax = plt.subplots(figsize=(15, 10))
    
    for i, method in enumerate(methods):
        error_rates = [data[d][method] * 100 for d in datasets]  # Convert to percentage
        ax.bar(x + (i - len(methods)/2 + 0.5) * width, error_rates, width, label=method)
    
    ax.set_xlabel('Data Sets')
    ax.set_ylabel('Error Rate (%)')
    ax.set_title(f'Classification Errors with {noise_key}')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45)
    ax.legend()
    
    plt.tight_layout()
    return fig
