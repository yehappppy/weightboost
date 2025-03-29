"""
Data utility functions for the WeightBoost package.

This module contains utility functions for data processing and manipulation.
"""

import numpy as np


def add_noise(y, noise_level):
    """
    Add noise to labels by flipping a percentage of them.
    
    Parameters
    ----------
    y : array-like of shape (n_samples,)
        The target values to add noise to.
        
    noise_level : float
        The proportion of labels to flip, between 0 and 1.
        
    Returns
    -------
    noisy_y : array-like of shape (n_samples,)
        The noisy target values.
    """
    noisy_y = y.copy()
    n_samples = len(y)
    n_noise = int(n_samples * noise_level)
    noise_idx = np.random.choice(n_samples, n_noise, replace=False)
    noisy_y[noise_idx] = -noisy_y[noise_idx]  # Flip labels
    return noisy_y
