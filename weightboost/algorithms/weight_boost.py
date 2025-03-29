"""
WeightBoost algorithm implementation.

This module implements the WeightBoost algorithm as described in the paper
"A New Boosting Algorithm Using Input-Dependent Regularizer" by Jin et al. (2003).
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone


class WeightBoost:
    """
    WeightBoost classifier.
    
    WeightBoost is a boosting algorithm that uses an input-dependent regularizer
    to make the combination weights a function of the input. This allows each
    base classifier to contribute only in input regions where it performs well.
    
    Parameters
    ----------
    base_classifier : object, default=DecisionTreeClassifier(max_depth=1)
        The base estimator from which the boosted ensemble is built.
        Support for sample weighting is required.
        
    n_estimators : int, default=100
        The maximum number of estimators at which boosting is terminated.
        
    beta : float, default=0.5
        The parameter that controls the strength of the input-dependent regularization.
        
    Attributes
    ----------
    models : list
        The collection of fitted base estimators.
        
    alphas : list
        The weights for each estimator in the boosted ensemble.
    """
    
    def __init__(self, base_classifier=DecisionTreeClassifier(max_depth=1), n_estimators=100, beta=0.5):
        self.base_classifier = base_classifier
        self.n_estimators = n_estimators
        self.beta = beta  # Input-dependent regularizer parameter
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
        H = np.zeros(n_samples)  # Cumulative classifier output
        
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
                
            # Calculate model weight
            alpha = 0.5 * np.log((1 - err) / max(err, 1e-10))
            
            # Save model and weight
            self.models.append(model)
            self.alphas.append(alpha)
            
            # Update cumulative classifier output
            H += alpha * pred
            
            # Calculate regularization factor
            reg = np.exp(-np.abs(self.beta * H))
            
            # Update sample weights (with input-dependent regularizer)
            w = np.exp(-y * H) * reg  # Apply regularization to the exponential loss
            
            # Prevent division by zero or very small numbers
            sum_w = np.sum(w)
            if sum_w > 1e-10:  # Add a small threshold check
                w = w / sum_w  # Normalize
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
        n_samples = X.shape[0]
        H = np.zeros(n_samples)
        
        # Apply classifiers sequentially
        for i, (alpha, model) in enumerate(zip(self.alphas, self.models)):
            if i == 0:
                H = alpha * model.predict(X)
            else:
                # Calculate regularization factor
                reg = np.exp(-np.abs(self.beta * H))
                
                # Combine base classifier prediction with input-dependent regularizer
                H = H + alpha * reg * model.predict(X)
        
        return np.sign(H)
