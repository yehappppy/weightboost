"""
WeightBoost: A Boosting Algorithm Using Input-Dependent Regularizer

This package implements the WeightBoost algorithm as described in the paper 
"A New Boosting Algorithm Using Input-Dependent Regularizer" by Jin et al. (2003).
"""

from weightboost.algorithms.adaboost import AdaBoost
from weightboost.algorithms.weight_decay import WeightDecay
from weightboost.algorithms.epsilon_boost import EpsilonBoost
from weightboost.algorithms.weight_boost import WeightBoost
from weightboost.evaluation.evaluator import evaluate_uci_datasets, evaluate_reuters
from weightboost.utils.visualization import plot_results
from weightboost.utils.data_utils import add_noise

__all__ = [
    'AdaBoost',
    'WeightDecay',
    'EpsilonBoost',
    'WeightBoost',
    'evaluate_uci_datasets',
    'evaluate_reuters',
    'plot_results',
    'add_noise'
]
