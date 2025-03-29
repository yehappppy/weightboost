"""
ε-Boost algorithm implementation.

This module implements the ε-Boost variant of AdaBoost which uses
small fixed weights for base classifiers.
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone
from weightboost.algorithms.adaboost import AdaBoost


class EpsilonBoost(AdaBoost):
    """
    ε-Boost classifier.
    
    ε-Boost is a variant of AdaBoost that uses small fixed weights (epsilon)
    for combining base classifiers instead of adaptive weights.
    
    Parameters
    ----------
    base_classifier : object, default=DecisionTreeClassifier(max_depth=1)
        The base estimator from which the boosted ensemble is built.
        Support for sample weighting is required.
        
    n_estimators : int, default=100
        The maximum number of estimators at which boosting is terminated.
        
    epsilon : float, default=0.1
        The fixed weight used for all base classifiers.
        
    Attributes
    ----------
    models : list
        The collection of fitted base estimators.
        
    alphas : list
        The weights for each estimator in the boosted ensemble.
    """
    
    def __init__(self, base_classifier=DecisionTreeClassifier(max_depth=1), n_estimators=100, epsilon=0.1):
        super().__init__(base_classifier, n_estimators)
        self.epsilon = epsilon  # Small weight factor
        
    def fit(self, X, y):
        """
        Build a boosted classifier from the training data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
            
        y : array-like of shape (n_samples,)
            The target values, with values in {-1, 1}.
            
        Returns
        -------
        self : object
            Returns self.
        """
        n_samples = X.shape[0]
        w = np.ones(n_samples) / n_samples  # Initialize weights
        
        for t in range(self.n_estimators):
            # Train base classifier
            model = clone(self.base_classifier)
            model.fit(X, y, sample_weight=w)
            pred = model.predict(X)
            
            # Calculate weighted error
            err = np.sum(w * (pred != y)) / np.sum(w)
            
            # Check if error rate is too high
            if err >= 0.5:
                break
                
            # Fixed small weight
            alpha = self.epsilon
            
            # Save model and weight
            self.models.append(model)
            self.alphas.append(alpha)
            
            # Update sample weights
            w = w * np.exp(-alpha * y * pred)
            
            # Prevent division by zero or very small numbers
            sum_w = np.sum(w)
            if sum_w > 1e-10:  # Add a small threshold check
                w = w / sum_w  # Normalize
            else:
                # If weights are too small, reinitialize with uniform weights
                w = np.ones(n_samples) / n_samples
        
        return self
