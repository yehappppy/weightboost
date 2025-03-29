"""
Evaluator module for boosting algorithms.

This module contains functions to evaluate boosting algorithms on various datasets.
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score

from weightboost.algorithms.adaboost import AdaBoost
from weightboost.algorithms.weight_decay import WeightDecay
from weightboost.algorithms.epsilon_boost import EpsilonBoost
from weightboost.algorithms.weight_boost import WeightBoost
from weightboost.utils.data_utils import add_noise


def evaluate_uci_datasets(datasets, noise_levels=[0, 0.1, 0.2, 0.3]):
    """
    Evaluate all algorithms on UCI datasets.
    
    Parameters
    ----------
    datasets : dict
        Dictionary mapping dataset names to (X, y) tuples.
        
    noise_levels : list, default=[0, 0.1, 0.2, 0.3]
        List of noise levels to evaluate.
        
    Returns
    -------
    results : dict
        Dictionary containing evaluation results for each dataset and noise level.
    """
    results = {}
    
    for noise_level in noise_levels:
        noise_results = {}
        
        for name, (X, y) in datasets.items():
            dataset_results = {}
            
            # Create noisy labels
            if noise_level > 0:
                noisy_y = add_noise(y, noise_level)
            else:
                noisy_y = y
                
            # Baseline: C4.5 decision tree
            base_score = np.mean(cross_val_score(
                DecisionTreeClassifier(), X, noisy_y, cv=10, scoring='accuracy'))
            dataset_results['C4.5'] = 1 - base_score
            
            # AdaBoost
            adaboost_score = np.mean(cross_val_score(
                AdaBoost(n_estimators=100), X, noisy_y, cv=10, scoring='accuracy'))
            dataset_results['AdaBoost'] = 1 - adaboost_score
            
            # Weight Decay
            wd_score = np.mean(cross_val_score(
                WeightDecay(n_estimators=100), X, noisy_y, cv=10, scoring='accuracy'))
            dataset_results['Weight Decay'] = 1 - wd_score
            
            # ε-Boost
            eps_score = np.mean(cross_val_score(
                EpsilonBoost(n_estimators=100), X, noisy_y, cv=10, scoring='accuracy'))
            dataset_results['ε-Boost'] = 1 - eps_score
            
            # WeightBoost
            wb_score = np.mean(cross_val_score(
                WeightBoost(n_estimators=100, beta=0.5), X, noisy_y, cv=10, scoring='accuracy'))
            dataset_results['WeightBoost'] = 1 - wb_score
            
            noise_results[name] = dataset_results
        
        results[f"{int(noise_level*100)}% Noise"] = noise_results
    
    return results


def evaluate_reuters(X_train, y_train, X_test, y_test):
    """
    Evaluate algorithms on Reuters text classification dataset.
    
    Parameters
    ----------
    X_train : array-like of shape (n_samples, n_features)
        The training input samples.
        
    y_train : array-like of shape (n_samples,)
        The training target values.
        
    X_test : array-like of shape (n_samples, n_features)
        The testing input samples.
        
    y_test : array-like of shape (n_samples,)
        The testing target values.
        
    Returns
    -------
    results : dict
        Dictionary containing evaluation results for each category.
    """
    # Select top 10 most common categories for evaluation
    top_categories = np.unique(y_train)[:10]
    results = {}
    
    for category in top_categories:
        category_results = {}
        
        # Create binary classification problem (one-vs-all)
        y_train_binary = np.where(y_train == category, 1, -1)
        y_test_binary = np.where(y_test == category, 1, -1)
        
        # Baseline: C4.5 decision tree
        base_clf = DecisionTreeClassifier()
        base_clf.fit(X_train, y_train_binary)
        base_pred = base_clf.predict(X_test)
        base_f1 = f1_score(y_test_binary, base_pred, average='binary')
        category_results['C4.5'] = base_f1
        
        # AdaBoost (10 iterations, based on paper)
        ada = AdaBoost(n_estimators=10)
        ada.fit(X_train, y_train_binary)
        ada_pred = ada.predict(X_test)
        ada_f1 = f1_score(y_test_binary, ada_pred, average='binary')
        category_results['AdaBoost'] = ada_f1
        
        # WeightBoost (25 iterations, based on paper)
        wb = WeightBoost(n_estimators=25, beta=0.5)
        wb.fit(X_train, y_train_binary)
        wb_pred = wb.predict(X_test)
        wb_f1 = f1_score(y_test_binary, wb_pred, average='binary')
        category_results['WeightBoost'] = wb_f1
        
        results[category] = category_results
    
    return results
