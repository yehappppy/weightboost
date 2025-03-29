"""
AdaBoost algorithm implementation.

This module implements the original AdaBoost algorithm by Freund and Schapire.
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone
from sklearn.utils.validation import check_X_y, check_array

class AdaBoost:
    """
    AdaBoost (Adaptive Boosting) classifier.
    
    AdaBoost is an ensemble method that fits a sequence of weak learners on 
    repeatedly modified versions of the data, where the weights are adjusted
    based on the previous prediction results.
    
    Parameters
    ----------
    base_classifier : object, default=DecisionTreeClassifier(max_depth=1)
        The base estimator from which the boosted ensemble is built.
        Support for sample weighting is required.
        
    n_estimators : int, default=100
        The maximum number of estimators at which boosting is terminated.
        
    Attributes
    ----------
    models : list
        The collection of fitted base estimators.
        
    alphas : list
        The weights for each estimator in the boosted ensemble.
    """
    
    def __init__(self, base_classifier=DecisionTreeClassifier(max_depth=1), n_estimators=100):
        self.base_classifier = base_classifier
        self.n_estimators = n_estimators
        self.alphas = []
        self.models = []
        
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
            err = np.clip(np.sum(w * (pred != y)) / np.sum(w), 1e-10, 1 - 1e-10)
            
            # Check if error rate is too high
            if err >= 0.5:
                break
                
            # Calculate model weight
            alpha = 0.5 * np.log((1 - err) / err)
            
            # Save model and weight
            self.models.append(model)
            self.alphas.append(alpha)
            
            # Update sample weights
            w *= np.exp(-alpha * y * pred)
            
            # Prevent division by zero or very small numbers
            sum_w = np.sum(w)
            if sum_w > 1e-10:  
                w /= sum_w  # Normalize
            else:
                # If weights are too small, reinitialize with uniform weights
                w = np.ones(n_samples) / n_samples
        
        return self
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
            
        Returns
        -------
        y : array-like of shape (n_samples,)
            The predicted classes, with values in {-1, 1}.
        """
        X = check_array(X)
        H = np.zeros(X.shape[0])
        
        for alpha, model in zip(self.alphas, self.models):
            H += alpha * model.predict(X)
            
        return np.sign(H)
