import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import threading
from functools import wraps
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from weightboost import (
    AdaBoost,
    WeightDecay,
    EpsilonBoost,
    WeightBoost,
    C45Tree
)

# ==================== Timeout Decorator ====================
class TimeoutError(Exception):
    """Exception raised when a function times out."""
    pass

def timeout(seconds=60):
    """Timeout decorator using threading (works on Windows)"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = [None]
            error = [None]
            
            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    error[0] = e
            
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(seconds)
            
            if thread.is_alive():
                raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds")
            
            if error[0] is not None:
                raise error[0]
                
            return result[0]
        return wrapper
    return decorator

# ==================== Dataset Loading Functions ====================
def load_datasets():
    """Load all datasets and automatically encode categorical features"""
    datasets = {}
    
    # === Categorical feature encoding function ===
    def encode_categorical(df):
        # Create a copy to avoid SettingWithCopyWarning
        df = df.copy()
        
        # Handle missing values first
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        
        # Impute numeric columns
        if len(num_cols) > 0:
            num_imputer = SimpleImputer(strategy='mean')
            df[num_cols] = num_imputer.fit_transform(df[num_cols])
        
        # Impute and encode categorical columns
        if len(cat_cols) > 0:
            # First impute missing values with the most frequent value
            cat_imputer = SimpleImputer(strategy='most_frequent')
            df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
            
            # Then encode categorical features
            encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            df[cat_cols] = encoder.fit_transform(df[cat_cols])
        
        return df

    # 1. Ionosphere (ID 52)
    ionosphere = fetch_ucirepo(id=52)
    X = encode_categorical(ionosphere.data.features).to_numpy()
    y = np.where(ionosphere.data.targets.iloc[:, 0] == 'g', 1, -1)
    datasets['Ionosphere'] = (X, y)

    # 2. German (ID 144)
    german = fetch_ucirepo(id=144)
    X = encode_categorical(german.data.features).to_numpy()
    y = np.where(german.data.targets.iloc[:, 0] == 2, -1, 1)
    datasets['German'] = (X, y)

    # 3. Pima Indians Diabetes (local loading)
    pima = pd.read_csv('./data/pima-indians-diabetes.csv')
    X = encode_categorical(pima.iloc[:, :-1]).to_numpy()
    y = np.where(pima.iloc[:, -1] == 1, 1, -1)
    datasets['Pima'] = (X, y)

    # 4. Breast Cancer Diagnostic (ID 15)
    bc_diag = fetch_ucirepo(id=15)
    X = encode_categorical(bc_diag.data.features).to_numpy()
    y = np.where(bc_diag.data.targets.iloc[:, 0] == 'malignant', -1, 1)
    datasets['BreastCancer'] = (X, y)

    # 5. wpbc (ID 16)
    wpbc = fetch_ucirepo(id=16)
    X = encode_categorical(wpbc.data.features).to_numpy()
    y = np.where(wpbc.data.targets.iloc[:, 0] == 'R', 1, -1)
    datasets['wpbc'] = (X, y)

    # 6. wdbc (ID 17)
    wdbc = fetch_ucirepo(id=17)
    X = encode_categorical(wdbc.data.features).to_numpy()
    y = np.where(wdbc.data.targets.iloc[:, 0] == 'M', -1, 1)
    datasets['wdbc'] = (X, y)

    # 7. Contraceptive (ID 30)
    cmc = fetch_ucirepo(id=30)
    X = encode_categorical(cmc.data.features).to_numpy()
    y = np.where(cmc.data.targets.iloc[:, 0].isin([2, 3]), 1, -1)
    datasets['Contraceptive'] = (X, y)

    # 8. Spambase (ID 94)
    spambase = fetch_ucirepo(id=94)
    X = encode_categorical(spambase.data.features).to_numpy()
    y = np.where(spambase.data.targets.iloc[:, 0] == 1, 1, -1)
    datasets['Spambase'] = (X, y)
    
    return datasets

# ==================== Test Functions ====================
def run_experiment(datasets, noise_levels=[0, 0.05, 0.1, 0.15, 0.2], n_estimators=50):
    """Run experiments on multiple datasets"""
    results = {}
    
    for ds_name, (X, y) in datasets.items():
        print(f"\n=== Processing dataset: {ds_name} ===")
        try:
            # Split data BEFORE initializing classifiers to ensure consistent feature dimensions
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Initialize classifiers after data split
            classifiers = {
                'C45Tree': C45Tree(min_samples_split=2, max_depth=5),
                'AdaBoost': AdaBoost(n_estimators=n_estimators),
                'WeightDecay': WeightDecay(n_estimators=n_estimators, C=0.1),
                'EpsilonBoost': EpsilonBoost(n_estimators=n_estimators, epsilon=0.1),
                'WeightBoost': WeightBoost(n_estimators=n_estimators, beta=0.5)
            }
            
            # Store results
            ds_results = {
                'clean': {name: [] for name in classifiers},
                'noisy': {name: {noise: None for noise in noise_levels} for name in classifiers}
            }
            
            # Define a timeout wrapper for fit
            @timeout(30)  # 30 seconds timeout for training
            def fit_with_timeout(clf, X, y):
                return clf.fit(X, y)
            
            # Test on clean data
            print("Testing clean data...")
            for name, clf in classifiers.items():
                try:
                    start_time = time.time()
                    fit_with_timeout(clf, X_train, y_train)
                    acc = np.mean(clf.predict(X_test) == y_test)
                    ds_results['clean'][name] = acc
                    print(f"{name:12} Accuracy: {acc:.4f} (Time: {time.time() - start_time:.2f}s)")
                except Exception as e:
                    print(f"{name:12} Failed: {str(e)}")
                    ds_results['clean'][name] = 0.0
            
            # Test on noisy data
            print("Testing noisy data...")
            for noise in noise_levels:
                print(f"  Noise level: {noise}")
                y_noisy = add_noise(y_train, noise_level=noise)
                for name, clf in classifiers.items():
                    try:
                        start_time = time.time()
                        fit_with_timeout(clf, X_train, y_noisy)
                        acc = np.mean(clf.predict(X_test) == y_test)
                        ds_results['noisy'][name][noise] = acc
                        print(f"  {name:12} Accuracy: {acc:.4f} (Time: {time.time() - start_time:.2f}s)")
                    except Exception as e:
                        print(f"  {name:12} Failed: {str(e)}")
                        ds_results['noisy'][name][noise] = 0.0
            
            results[ds_name] = ds_results
            
            # Plot results
            plot_dataset_results(ds_name, ds_results, noise_levels)
        except Exception as e:
            print(f"Error processing dataset {ds_name}: {str(e)}")
    
    return results

def plot_dataset_results(ds_name, results, noise_levels):
    """Plot results for a single dataset"""
    plt.figure(figsize=(14, 6))
    
    # Accuracy comparison
    plt.subplot(1, 2, 1)
    plt.bar(results['clean'].keys(), results['clean'].values(), 
            color=['purple', 'blue', 'green', 'orange', 'red'])
    plt.title(f'{ds_name} - Clean Data Accuracy')
    plt.ylim(0.5, 1.05)
    for i, v in enumerate(results['clean'].values()):
        plt.text(i, v+0.02, f"{v:.4f}", ha='center')
    plt.grid(axis='y', linestyle='--')
    
    # Noise robustness
    plt.subplot(1, 2, 2)
    for name in results['noisy']:
        # Convert dictionary to list for plotting
        noise_results = [results['noisy'][name].get(noise, 0.0) for noise in noise_levels]
        plt.plot(noise_levels, noise_results, marker='o', label=name)
    plt.title(f'{ds_name} - Noise Robustness')
    plt.xlabel('Noise Level')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(f'./images/results_{ds_name}.png')
    plt.close()

# ==================== Noise Generation Function ====================
def add_noise(y, noise_level=0.1):
    """Add label noise"""
    n_samples = len(y)
    flip_indices = np.random.choice(
        n_samples, 
        size=int(n_samples * noise_level), 
        replace=False
    )
    y_noisy = y.copy()
    y_noisy[flip_indices] = -y_noisy[flip_indices]
    return y_noisy

# ==================== Main Program ====================
if __name__ == "__main__":
    # Load all datasets (ensure pima file path is correct)
    datasets = load_datasets()
    
    # Run experiments
    results = run_experiment(datasets)
    
    print("\nAll experiments completed!")