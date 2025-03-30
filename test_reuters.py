import pandas as pd
import numpy as np
import re
import ast
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import warnings
from weightboost import WeightBoost, AdaBoost
warnings.filterwarnings('ignore')
np.random.seed(66)


try:
    nltk.download('stopwords', quiet=True)
except:
    print("Failed to download NLTK resources. Please ensure NLTK is properly installed.")

def preprocess_reuters(data_path):
    """
    Preprocess Reuters dataset (multi-label task)
    
    Parameters:
        data_path (str): Path to input Excel file
    
    Returns:
        X_train, y_train, X_test, y_test: Preprocessed training and test sets
    """
    # 1. Load data
    print("Loading data...")
    df = pd.read_excel(data_path)
    df['text'] = df['text'].fillna("")  # Ensure text column has no null values
    df['categories'] = df['categories'].fillna("[]")  # Ensure categories column has no null values

    # Parse string-formatted categories into Python lists
    print("Parsing categories column...")
    df['categories'] = df['categories'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # 2. Text preprocessing
    print("Cleaning text...")
    stop_words = set(stopwords.words("english"))  # Stopwords
    stemmer = PorterStemmer()  # Stemmer

    def tokenize_and_clean(text):
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = re.sub(r"[^\w\s]", "", text)
        # Tokenize, remove stopwords, and apply stemming
        tokens = text.split()
        tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
        return " ".join(tokens)

    # Clean all text
    df['cleaned_text'] = df['text'].apply(tokenize_and_clean)

    # 3. Feature selection: TF-IDF vectorization
    print("Performing feature selection...")
    vectorizer = TfidfVectorizer(max_features=2000)  # Limit to 2000 features max
    X = vectorizer.fit_transform(df['cleaned_text'])  # Document vectorization

    # Extract category labels and perform multi-label encoding
    print("Performing multi-label encoding...")
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df['categories'])  # Convert categories to multi-label one-hot encoding

    # Save MultiLabelBinarizer classes as attribute for later decoding
    preprocess_reuters.mlb_classes = mlb.classes_

    # 4. Dataset splitting
    print("Splitting into training and test sets...")
    train_indices = df[df['ids'].str.startswith('training')].index
    test_indices = df[df['ids'].str.startswith('test')].index

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    print("Preprocessing completed!")
    return X_train, y_train, X_test, y_test, mlb.classes_

def convert_labels_to_binary(y, positive=1, negative=-1):
    """
    Convert 0/1 labels to -1/1 labels for AdaBoost and WeightBoost compatibility
    """
    return np.where(y == 1, positive, negative)

def main():
    # Data path
    data_path = "./data/reutersNLTK.xlsx"
    
    # Preprocess data
    X_train, y_train, X_test, y_test, categories = preprocess_reuters(data_path)
    
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    print(f"Number of categories: {y_train.shape[1]}")
    
    # Select top 10 most frequent categories
    category_counts = np.sum(y_train, axis=0)
    top_10_indices = np.argsort(category_counts)[-10:]
    top_10_categories = categories[top_10_indices]
    
    print(f"Selected top 10 categories: {top_10_categories}")
    
    # Prepare results table
    results = []
    
    for i, cat_idx in enumerate(top_10_indices):
        cat_name = categories[cat_idx]
        print(f"\nProcessing category {cat_name} ({i+1}/10)...")
        
        # Get labels for current category
        y_train_cat = y_train[:, cat_idx]
        y_test_cat = y_test[:, cat_idx]
        
        # Convert labels to {-1, 1} format for AdaBoost and WeightBoost
        y_train_binary = convert_labels_to_binary(y_train_cat)
        y_test_binary = convert_labels_to_binary(y_test_cat)
        
        # Train C4.5 Decision Tree (baseline)
        print("Training C4.5 Decision Tree...")
        c45 = DecisionTreeClassifier(criterion='entropy')
        c45.fit(X_train, y_train_cat)
        y_pred_c45 = c45.predict(X_test)
        f1_c45 = f1_score(y_test_cat, y_pred_c45)

        print("Training AdaBoost...")
        ada = AdaBoost(
            base_classifier=DecisionTreeClassifier(criterion='entropy'),
            n_estimators=50
        )
        ada.fit(X_train, y_train_binary)
        y_pred_ada = ada.predict(X_test)
        # Convert -1/1 predictions back to 0/1 for F1 calculation
        y_pred_ada_01 = np.where(y_pred_ada == 1, 1, 0)
        f1_ada = f1_score(y_test_cat, y_pred_ada_01)
        
        ada_impro = ((f1_ada - f1_c45) / f1_c45) * 100
        
        print("Training WeightBoost...")
        wb = WeightBoost(
            base_classifier=DecisionTreeClassifier(criterion='entropy'),
            n_estimators=50,
            beta=0.5
        )
        wb.fit(X_train, y_train_binary)
        y_pred_wb = wb.predict(X_test)
        # Convert -1/1 predictions back to 0/1 for F1 calculation
        y_pred_wb_01 = np.where(y_pred_wb == 1, 1, 0)
        f1_wb = f1_score(y_test_cat, y_pred_wb_01)
        
        wb_impro = ((f1_wb - f1_c45) / f1_c45) * 100
   
        results.append({
            'Category': cat_name,
            'C4.5_F1': f1_c45,
            'AdaBoost_F1': f1_ada,
            'AdaBoost_Impro': f"{ada_impro:.1f}%",
            'WeightBoost_F1': f1_wb,
            'WeightBoost_Impro': f"{wb_impro:.1f}%"
        })
        
    results_df = pd.DataFrame(results)
    
    # Format F1 scores to 4 decimal places
    results_df['C4.5_F1'] = results_df['C4.5_F1'].apply(lambda x: f"{x:.4f}")
    results_df['AdaBoost_F1'] = results_df['AdaBoost_F1'].apply(lambda x: f"{x:.4f}")
    results_df['WeightBoost_F1'] = results_df['WeightBoost_F1'].apply(lambda x: f"{x:.4f}")
    
    print("\nResults table:")
    print(results_df[['Category', 'C4.5_F1', 'AdaBoost_F1', 'AdaBoost_Impro', 'WeightBoost_F1', 'WeightBoost_Impro']].to_string(index=False))
    
    results_df.to_csv('./result/reuters_boosting_results.csv', index=False)
    print("\nResults saved to reuters_boosting_results.csv")

if __name__ == "__main__":
    main()