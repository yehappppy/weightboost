"""
Evaluation module.

This module contains functions for evaluating boosting algorithms.
"""

from weightboost.evaluation.evaluator import evaluate_uci_datasets, evaluate_reuters

__all__ = [
    'evaluate_uci_datasets',
    'evaluate_reuters'
]
