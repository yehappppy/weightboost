"""
Boosting algorithms module.

This module contains implementations of various boosting algorithms.
"""

from weightboost.algorithms.adaboost import AdaBoost
from weightboost.algorithms.weight_decay import WeightDecay
from weightboost.algorithms.epsilon_boost import EpsilonBoost
from weightboost.algorithms.weight_boost import WeightBoost

__all__ = [
    'AdaBoost',
    'WeightDecay',
    'EpsilonBoost',
    'WeightBoost'
]
