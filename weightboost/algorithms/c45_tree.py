import numpy as np
from collections import Counter
import math

class C45Node:
    """Decision tree node class"""
    def __init__(self, feature=None, threshold=None, label=None, children=None):
        self.feature = feature      # Index of the feature to split on
        self.threshold = threshold  # Threshold for continuous features
        self.label = label          # Class label for leaf nodes
        self.children = children    # Dictionary of child nodes {feature_value: node}

class C45Tree:
    """C4.5 Decision Tree implementation"""
    
    def __init__(self, min_samples_split=2, max_depth=None, pruning_confidence=0.25):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.pruning_confidence = pruning_confidence  # Confidence level for pessimistic pruning
        self.root = None
        self.feature_types = None     # To store feature types (continuous or discrete)
        self.global_majority = None   # Majority class from training data
    
    def fit(self, X, y, features=None, feature_types=None):
        """Train the decision tree"""
        # Convert y to 1D array if needed
        y = np.asarray(y).ravel()
        
        if features is None:
            features = [f"feature_{i}" for i in range(X.shape[1])]
        self.features = features
        
        # Determine feature types if not provided
        if feature_types is None:
            # Heuristic: >10 unique values implies continuous
            self.feature_types = ['continuous' if len(np.unique(X[:, i])) > 10 else 'discrete' 
                                 for i in range(X.shape[1])]
        else:
            self.feature_types = feature_types
        
        # Store global majority class for unseen value handling
        counts = Counter(y)
        if counts:
            self.global_majority = counts.most_common(1)[0][0]
        else:
            self.global_majority = 1  # Default to positive class if empty
            
        self.root = self._build_tree(X, y, depth=0)
        self._prune_tree(self.root, X, y)
        return self
    
    def _build_tree(self, X, y, depth):
        """Recursively build the decision tree"""
        # Termination conditions
        if len(y) == 0:  # No samples
            return C45Node(label=self.global_majority)
            
        if len(set(y)) == 1:  # All samples have the same label
            return C45Node(label=y[0])
            
        if len(X) < self.min_samples_split:  # Too few samples
            return C45Node(label=self._most_common_label(y))
            
        if self.max_depth and depth >= self.max_depth:  # Maximum depth reached
            return C45Node(label=self._most_common_label(y))
        
        # Choose the best feature and threshold to split
        best_feat, best_thresh = self._choose_best_split(X, y)
        if best_feat is None:  # No valid split found
            return C45Node(label=self._most_common_label(y))
        
        # Handle discrete or continuous features
        if self.feature_types[best_feat] == 'discrete':
            feat_values = set(X[:, best_feat])
            children = {}
            for val in feat_values:
                mask = X[:, best_feat] == val
                if np.any(mask):  # Only create child if samples exist
                    children[val] = self._build_tree(X[mask], y[mask], depth + 1)
            
            # Add a default child for unseen values
            if children:
                return C45Node(feature=best_feat, children=children)
            else:
                return C45Node(label=self._most_common_label(y))
        else:  # Continuous feature
            left_mask = X[:, best_feat] <= best_thresh
            right_mask = ~left_mask
            
            # Check if split is valid (both partitions have samples)
            if np.any(left_mask) and np.any(right_mask):
                children = {
                    'left': self._build_tree(X[left_mask], y[left_mask], depth + 1),
                    'right': self._build_tree(X[right_mask], y[right_mask], depth + 1)
                }
                return C45Node(feature=best_feat, threshold=best_thresh, children=children)
            else:
                return C45Node(label=self._most_common_label(y))
    
    def _choose_best_split(self, X, y):
        """Select the best feature and threshold for splitting"""
        best_gain_ratio = -1
        best_feat = None
        best_thresh = None
        
        for feat_idx in range(X.shape[1]):
            feat_values = X[:, feat_idx]
            
            if self.feature_types[feat_idx] == 'continuous':
                thresholds = self._find_continuous_thresholds(feat_values)
                for thresh in thresholds:
                    mask = feat_values <= thresh
                    
                    # Skip if split doesn't divide the data
                    if np.all(mask) or np.all(~mask):
                        continue
                        
                    try:
                        gain_ratio = self._gain_ratio(y, mask)
                        if gain_ratio > best_gain_ratio:
                            best_gain_ratio = gain_ratio
                            best_feat = feat_idx
                            best_thresh = thresh
                    except (ValueError, ZeroDivisionError):
                        continue
            else:  # Discrete feature
                try:
                    gain_ratio = self._gain_ratio_discrete(y, feat_values)
                    if gain_ratio > best_gain_ratio:
                        best_gain_ratio = gain_ratio
                        best_feat = feat_idx
                        best_thresh = None  # No threshold for discrete features
                except (ValueError, ZeroDivisionError):
                    continue
                    
        return best_feat, best_thresh
    
    def _find_continuous_thresholds(self, values):
        """Find candidate thresholds for continuous features (midpoints of sorted unique values)"""
        sorted_vals = np.sort(np.unique(values))
        if len(sorted_vals) <= 1:
            return []
        return [(sorted_vals[i] + sorted_vals[i + 1]) / 2 for i in range(len(sorted_vals) - 1)]
    
    def _gain_ratio(self, y, mask):
        """Calculate the gain ratio for a continuous feature split"""
        entropy_before = self._entropy(y)
        n = len(y)
        
        # Check if the split is valid
        n_left = np.sum(mask)
        n_right = n - n_left
        
        if n_left == 0 or n_right == 0:
            return 0
            
        entropy_after = (n_left / n) * self._entropy(y[mask]) + \
                        (n_right / n) * self._entropy(y[~mask])
        info_gain = entropy_before - entropy_after
        
        # Calculate split information
        p_left = n_left / n
        p_right = n_right / n
        split_info = -(p_left * np.log2(p_left + 1e-10) + p_right * np.log2(p_right + 1e-10))
        
        # Avoid division by zero
        if split_info < 1e-10:
            return 0
            
        return info_gain / split_info
    
    def _gain_ratio_discrete(self, y, feat_values):
        """Calculate the gain ratio for a discrete feature"""
        entropy_before = self._entropy(y)
        unique_values = set(feat_values)
        
        if len(unique_values) <= 1:
            return 0
            
        weighted_entropy = 0
        split_info = 0
        n = len(y)
        
        for val in unique_values:
            mask = feat_values == val
            n_subset = np.sum(mask)
            
            if n_subset == 0:
                continue
                
            p = n_subset / n
            weighted_entropy += p * self._entropy(y[mask])
            split_info -= p * np.log2(p + 1e-10)
        
        info_gain = entropy_before - weighted_entropy
        
        # Avoid division by zero
        if split_info < 1e-10:
            return 0
            
        return info_gain / split_info
    
    def _entropy(self, y):
        """Calculate the entropy of a label set"""
        # Handle empty arrays
        if len(y) == 0:
            return 0
            
        # Count occurrences of each class
        counts = Counter(y)
        
        # If only one class, entropy is 0
        if len(counts) <= 1:
            return 0
            
        # Calculate entropy
        n = len(y)
        probs = [count / n for count in counts.values()]
        return -sum(p * np.log2(p + 1e-10) for p in probs)
    
    def _most_common_label(self, y):
        """Return the most common label in the set"""
        if len(y) == 0:
            return self.global_majority
            
        counts = Counter(y)
        if not counts:
            return self.global_majority
            
        return counts.most_common(1)[0][0]
    
    def _prune_tree(self, node, X, y):
        """Perform pessimistic error pruning on the tree"""
        if node is None or node.label is not None:  # Leaf node or None, no pruning needed
            return
            
        # Handle empty data
        if len(y) == 0:
            node.label = self.global_majority
            node.children = None
            node.feature = None
            node.threshold = None
            return
        
        # Recursively prune children
        if node.threshold is None:  # Discrete feature
            for val, child in node.children.items():
                mask = X[:, node.feature] == val
                if np.any(mask):  # Only prune if samples exist
                    self._prune_tree(child, X[mask], y[mask])
        else:  # Continuous feature
            left_mask = X[:, node.feature] <= node.threshold
            right_mask = ~left_mask
            
            if np.any(left_mask):
                self._prune_tree(node.children['left'], X[left_mask], y[left_mask])
            if np.any(right_mask):
                self._prune_tree(node.children['right'], X[right_mask], y[right_mask])
        
        # Calculate errors for pruning decision
        try:
            error_before = self._calculate_error(node, X, y)
            majority_label = self._most_common_label(y)
            error_leaf = np.mean(y != majority_label)
            
            n = len(y)
            if n > 0:
                std = np.sqrt(error_leaf * (1 - error_leaf) / n)
                pessimistic_error = error_leaf + self.pruning_confidence * std
                
                if pessimistic_error <= error_before:
                    node.label = majority_label
                    node.children = None
                    node.feature = None
                    node.threshold = None
        except:
            # If error calculation fails, convert to leaf
            node.label = self._most_common_label(y)
            node.children = None
            node.feature = None
            node.threshold = None
    
    def _calculate_error(self, node, X, y):
        """Calculate the classification error of the subtree"""
        if len(X) == 0 or len(y) == 0:
            return 0
            
        pred = np.array([self._predict_one(node, x) for x in X])
        return np.mean(pred != y)
    
    def _predict_one(self, node, x):
        """Predict the class for a single sample"""
        if node is None:
            return self.global_majority
            
        if node.label is not None:
            return node.label
        
        if node.threshold is None:  # Discrete feature
            val = x[node.feature]
            child = node.children.get(val)
            if child is None:  # Handle unseen feature value
                return self.global_majority
            return self._predict_one(child, x)
        else:  # Continuous feature
            if x[node.feature] <= node.threshold:
                return self._predict_one(node.children['left'], x)
            else:
                return self._predict_one(node.children['right'], x)
    
    def predict(self, X):
        """Predict classes for a batch of samples"""
        return np.array([self._predict_one(self.root, x) for x in X])