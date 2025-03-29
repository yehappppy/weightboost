import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# Import from the weightboost package
from weightboost import (
    AdaBoost,
    WeightDecay,
    EpsilonBoost,
    WeightBoost,
    evaluate_uci_datasets,
    plot_results
)

def main():
    # Load a sample dataset (Ionosphere)
    print("Loading Ionosphere dataset...")
    X, y = fetch_openml(name='ionosphere', version=1, return_X_y=True, as_frame=False)
    y = np.where(y == 'g', 1, -1)  # Convert labels to +1/-1
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Create and train classifiers
    print("\nTraining classifiers...")
    
    # Initialize classifiers
    classifiers = {
        'AdaBoost': AdaBoost(n_estimators=50),
        'WeightDecay': WeightDecay(n_estimators=50, C=0.1),
        'EpsilonBoost': EpsilonBoost(n_estimators=50, epsilon=0.1),
        'WeightBoost': WeightBoost(n_estimators=50, beta=0.5)
    }
    
    # Train and evaluate each classifier
    accuracies = {}
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        acc = np.mean(clf.predict(X_test) == y_test)
        accuracies[name] = acc
        print(f"{name} accuracy: {acc:.4f}")
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.bar(accuracies.keys(), accuracies.values(), color=['blue', 'green', 'orange', 'red'])
    plt.title('Classifier Performance Comparison on Ionosphere Dataset')
    plt.ylabel('Accuracy')
    plt.ylim(0.8, 1.0)  # Set y-axis limits to better show differences
    for i, v in enumerate(accuracies.values()):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
    
    # Example of adding noise to labels and comparing performance
    print("\nEvaluating with noisy labels...")
    from weightboost.utils.data_utils import add_noise
    
    noise_levels = [0, 0.05, 0.1, 0.15, 0.2]
    noise_results = {name: [] for name in classifiers}
    
    for noise in noise_levels:
        y_noisy = add_noise(y_train, noise_level=noise)
        for name, clf in classifiers.items():
            clf.fit(X_train, y_noisy)
            acc = np.mean(clf.predict(X_test) == y_test)
            noise_results[name].append(acc)
    
    # Plot noise sensitivity
    plt.figure(figsize=(10, 6))
    for name, results in noise_results.items():
        plt.plot(noise_levels, results, marker='o', label=name)
    
    plt.title('Classifier Robustness to Label Noise')
    plt.xlabel('Noise Level')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
    
    print("\nDone!")

if __name__ == "__main__":
    main()