# WeightBoost: A Boosting Algorithm Using Input-Dependent Regularizer

This project implements the WeightBoost algorithm as described in the paper "A New Boosting Algorithm Using Input-Dependent Regularizer" by Jin et al. (2003). The implementation includes the core WeightBoost algorithm along with several other boosting methods for comparison.

## Overview

WeightBoost addresses two major issues with traditional AdaBoost:

1. **Overfitting Problem**: AdaBoost tends to overemphasize hard-to-classify samples, especially in noisy datasets.
2. **Fixed Weight Combination Problem**: AdaBoost uses fixed constants to combine base classifiers without considering input patterns.

The key innovation of WeightBoost is the introduction of an "input-dependent regularizer" that makes the combination weights a function of the input:

```
H_T(x) = Σ α_t·e^(-|βH_{t-1}(x)|)·h_t(x)
```

This approach allows each base classifier to contribute only in the input regions where it performs well, while the regularization mitigates the impact of noisy data.

## Implemented Algorithms

The project implements several boosting algorithms for comparison:

1. **AdaBoost**: The original AdaBoost algorithm by Freund and Schapire.
2. **Weight Decay**: A regularized version of AdaBoost that adds a weight decay term.
3. **ε-Boost**: A variant that uses small fixed weights for base classifiers.
4. **WeightBoost**: The novel algorithm that uses input-dependent regularization.

```bash
weightboost/
├── __init__.py                 # Package entry point with imports
├── algorithms/                 # Algorithm implementations
│   ├── __init__.py
│   ├── adaboost.py             # AdaBoost implementation
│   ├── weight_decay.py         # Weight Decay implementation
│   ├── epsilon_boost.py        # ε-Boost implementation
│   └── weight_boost.py         # WeightBoost implementation
├── evaluation/                 # Evaluation utilities
│   ├── __init__.py
│   └── evaluator.py            # Functions for evaluating algorithms
└── utils/                      # Utility functions
    ├── __init__.py
    ├── data_utils.py           # Data manipulation utilities
    └── visualization.py        # Visualization utilities
```

## Evaluation

The implementation includes evaluation functions for:

- **UCI Datasets**: Evaluates all algorithms on standard UCI datasets with different noise levels.
- **Reuters Text Classification**: Evaluates the algorithms on the Reuters-21578 corpus.

## Results

The evaluation results show that:

- WeightBoost outperforms AdaBoost on all datasets.
- WeightBoost outperforms Weight Decay on 6 out of 8 datasets.
- WeightBoost shows significant improvement on datasets like "German", "Pima", and "Contraceptive", where AdaBoost shows little improvement.
- WeightBoost maintains good performance even with noisy data, while AdaBoost tends to overfit.

## Usage

### Prerequisites

The code requires the following Python packages:
- numpy
- scikit-learn
- matplotlib
- pandas (for data loading)

### Example Usage

```python
# Import the WeightBoost algorithm
from wb import WeightBoost, AdaBoost, evaluate_uci_datasets, plot_results

# Load datasets
from sklearn.datasets import fetch_openml
X, y = fetch_openml(name='ionosphere', version=1, return_X_y=True)
y = np.where(y == 'g', 1, -1)  # Convert labels to +1/-1

# Create a WeightBoost classifier
wb = WeightBoost(n_estimators=100, beta=0.5)
wb.fit(X, y)
predictions = wb.predict(X_test)

# Evaluate multiple algorithms on datasets
datasets = {"Ionosphere": (X, y)}
results = evaluate_uci_datasets(datasets)

# Visualize results
plot_results(results)
```

## Customization

You can customize the WeightBoost algorithm by adjusting:

- `base_classifier`: The base classifier (default is a decision stump)
- `n_estimators`: Number of boosting iterations
- `beta`: The regularization parameter that controls the strength of the input-dependent regularization

## References

1. Jin, R., Liu, Y., Si, L., Carbonell, J., & Hauptmann, A. G. (2003). A New Boosting Algorithm Using Input-Dependent Regularizer. *Proceedings of the Twentieth International Conference on Machine Learning (ICML-2003)*, Washington DC.
2. Freund, Y., & Schapire, R. E. (1996). Experiments with a new boosting algorithm. *Machine Learning: Proceedings of the Thirteenth International Conference*.
3. Friedman, J., Hastie, T., & Tibshirani, R. (1998). Additive logistic regression: a statistical view of boosting. *Annals of statistics*, 28(2), 337-407.
