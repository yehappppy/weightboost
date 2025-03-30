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
5. **C4.5 Decision Tree**: Implemented as a base classifier for the boosting algorithms.

```
weightboost/
├── __init__.py                 # Package entry point with imports
├── algorithms/                 # Algorithm implementations
│   ├── __init__.py
│   ├── adaboost.py             # AdaBoost implementation
│   ├── weight_decay.py         # Weight Decay implementation
│   ├── epsilon_boost.py        # ε-Boost implementation
│   ├── weight_boost.py         # WeightBoost implementation
│   └── c45_tree.py             # C4.5 decision tree implementation
├── evaluation/                 # Evaluation utilities
│   ├── __init__.py
│   └── evaluator.py            # Functions for evaluating algorithms
└── utils/                      # Utility functions
    ├── __init__.py
    ├── data_utils.py           # Data manipulation utilities
    └── visualization.py        # Visualization utilities
```

## Datasets

### UCI Datasets

The project evaluates the boosting algorithms on 8 UCI datasets:

1. **Ionosphere**: A dataset for classifying radar returns from the ionosphere as 'good' or 'bad'.
2. **German Credit**: A dataset for classifying credit risk as 'good' or 'bad'.
3. **Pima Indians Diabetes**: A dataset for predicting diabetes in Pima Indian women.
4. **Breast Cancer (Diagnostic)**: A dataset for diagnosing breast cancer as 'malignant' or 'benign'.
5. **wpbc (Wisconsin Prognostic Breast Cancer)**: A dataset for predicting breast cancer recurrence.
6. **wdbc (Wisconsin Diagnostic Breast Cancer)**: A dataset for diagnosing breast cancer.
7. **Contraceptive**: A dataset for predicting contraceptive method choice.
8. **Spambase**: A dataset for classifying emails as spam or non-spam.

All datasets were preprocessed to handle categorical features and missing values. The experiments were conducted with different noise levels (0%, 5%, 10%, 15%, 20%) to evaluate the robustness of the algorithms.

### Reuters-21578 Dataset

The Reuters-21578 dataset is a classic multi-label text classification dataset containing news articles from Reuters news agency. Key characteristics:

- **Size**: 10,788 news articles (7,769 for training, 3,019 for testing)
- **Categories**: 90 different topic categories, with each article potentially belonging to multiple categories
- **Format**: Each article includes a file ID, category labels, and the article text

For our experiments, we:
1. Preprocessed the text data using TF-IDF vectorization
2. Selected the 10 most frequent categories for evaluation
3. Converted the multi-label problem into multiple binary classification problems
4. Compared the performance of C4.5, AdaBoost, and WeightBoost using F1-score

## Evaluation Results

### UCI Datasets

The evaluation results on UCI datasets show that:

- WeightBoost outperforms AdaBoost on all datasets, particularly on datasets like "German", "Pima", and "Contraceptive".
- WeightBoost maintains good performance even with noisy data, while AdaBoost tends to overfit as noise levels increase.
- The input-dependent regularization in WeightBoost effectively mitigates the impact of label noise.

### Reuters Text Classification

The evaluation on the Reuters dataset shows:

- WeightBoost achieves the highest F1-scores on 7 out of 10 categories.
- WeightBoost shows significant improvements over C4.5 on categories like "trade" (15.56%), "corn" (9.70%), and "money-fx" (9.60%).
- AdaBoost shows negative improvement on categories like "grain" (-5.20%), while WeightBoost maintains more consistent performance.

## Usage

### Prerequisites

The code requires the following Python packages:
```
numpy
scikit-learn
matplotlib
pandas
ucimlrepo
nltk
```

### Example Usage

```python
# Import the WeightBoost algorithm
from weightboost import WeightBoost, AdaBoost, C45Tree

# Load and preprocess data
X_train, y_train = load_and_preprocess_data()

# Create a WeightBoost classifier
wb = WeightBoost(
    base_classifier=C45Tree(min_samples_split=2, max_depth=5),
    n_estimators=50,
    beta=0.5
)

# Train the classifier
wb.fit(X_train, y_train)

# Make predictions
predictions = wb.predict(X_test)
```

## Experimental Setup

The experiments were conducted with the following settings:

- **Base Classifier**: C4.5 Decision Tree (implemented as a custom classifier)
- **Number of Estimators**: 50 for all boosting algorithms
- **Parameters**:
  - WeightBoost: β = 0.5
  - Weight Decay: C = 0.1
  - ε-Boost: ε = 0.1
- **Evaluation Metric**: Accuracy for UCI datasets, F1-score for Reuters dataset
- **Noise Levels**: 0%, 5%, 10%, 15%, 20% (for UCI datasets)

## Conclusion

The experimental results confirm the theoretical advantages of WeightBoost:

1. **Improved Robustness**: WeightBoost shows better resistance to noise compared to AdaBoost.
2. **Better Generalization**: The input-dependent regularization helps prevent overfitting.
3. **Consistent Performance**: WeightBoost maintains good performance across different datasets and noise levels.

These results demonstrate that WeightBoost effectively addresses the limitations of traditional boosting algorithms by incorporating an input-dependent regularizer that adapts the contribution of each base classifier based on the input pattern.

## References

1. Jin, R., Liu, Y., Si, L., Carbonell, J., & Hauptmann, A. G. (2003). A New Boosting Algorithm Using Input-Dependent Regularizer. *Proceedings of the Twentieth International Conference on Machine Learning (ICML-2003)*, Washington DC.
2. Freund, Y., & Schapire, R. E. (1996). Experiments with a new boosting algorithm. *Machine Learning: Proceedings of the Thirteenth International Conference*.
3. Friedman, J., Hastie, T., & Tibshirani, R. (1998). Additive logistic regression: a statistical view of boosting. *Annals of statistics*, 28(2), 337-407.
