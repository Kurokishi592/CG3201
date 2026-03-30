import numpy as np
import pandas as pd
from utils import prepare_train_test, precision_recall_curve, plot_pr_curve, compute_auc, get_eval_statistics
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import os

np.random.seed(42)

class TFIDF:
    def __init__(self):
        self.idf = None

    def fit(self, X_train):
        """
        X_train: numpy array of counts
        """
        n_samples, n_features = X_train.shape
        
        # Calculate DF: count of documents where term frequency > 0
        df = np.sum(X_train > 0, axis=0)
        
        # Calculate IDF: smoothed idf = log((N + 1) / (df + 1)) + 1
        self.idf = np.log((n_samples + 1) / (df + 1)) + 1
        
        # Calculate TF-IDF
        tf_idf = X_train * self.idf
        
        # L2 Normalization
        norm = np.linalg.norm(tf_idf, axis=1, keepdims=True)
        # Avoid division by zero
        norm[norm == 0] = 1
        return tf_idf / norm

    def transform(self, X):
        """
        X: numpy array of counts
        """
        tf_idf = X * self.idf
        
        # L2 Normalization
        norm = np.linalg.norm(tf_idf, axis=1, keepdims=True)
        norm[norm == 0] = 1
        return tf_idf / norm

class SVM:
    def __init__(self, C=1.0, learning_rate=0.1, max_iter=100, batch_size=64):
        self.C = C
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.w = None
        self.b = None

    def fit(self, X_train, y_train):
        n_samples, n_features = X_train.shape
        self.w = np.zeros(n_features)
        self.b = 0
        learning_rate = self.learning_rate
        
        for it in range(self.max_iter):
            if (it != 0 and it % (it / 10) == 0):
                learning_rate *= 0.5
                
            # Shuffle indices at the start of each epoch
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            prev_w = self.w.copy()
            
            # Iterate over batches
            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i : i + self.batch_size]
                y_batch = y_shuffled[i : i + self.batch_size]
                batch_len = len(y_batch)
                
                distances = y_batch * (X_batch @ self.w + self.b)
                
                mask = distances < 1
                
                grad_w = self.w - self.C * ((mask * y_batch) @ X_batch) / batch_len
                grad_b = - self.C * np.sum(mask * y_batch) / batch_len
                
                self.w -= learning_rate * grad_w
                self.b -= learning_rate * grad_b

    
    def compute_scores(self, X):
        return X @ self.w + self.b

    def predict(self, X_test):
        return np.where(self.compute_scores(X_test) > 0, 1, 0)

def main():
    X_train, X_test, y_train, y_test, vocab = prepare_train_test()
   
    tfidf = TFIDF()
    X_train_tfidf = tfidf.fit(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    # Convert labels to {-1, 1}
    y_train_svm = np.where(y_train == 0, -1, 1)
    y_test_svm = np.where(y_test == 0, -1, 1)
    
    # Cross Validation to select C
    os.makedirs('results', exist_ok=True)
    kf = KFold(n_splits=4, shuffle=True, random_state=42)
    C_values = [1, 100, 10000, 1000000]
    results = {}
    
    for C in C_values:
        fold_f1s = []
        print(f"Evaluating C={C}")
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_tfidf)):
            X_fold_train, X_fold_val = X_train_tfidf[train_idx], X_train_tfidf[val_idx]
            y_fold_train, y_fold_val = y_train_svm[train_idx], y_train_svm[val_idx]
                
            svm = SVM(C=C, learning_rate=0.1, max_iter=100, batch_size=64)
            svm.fit(X_fold_train, y_fold_train)
            
            # Evaluate using F1 score
            predictions = svm.predict(X_fold_val)
            y_fold_val_orig = y_train[val_idx]
            _, _, _, f1 = get_eval_statistics(y_fold_val_orig, predictions)
            fold_f1s.append(f1)
            
        avg_f1 = np.mean(fold_f1s)
        results[C] = avg_f1
        print(f"Avg F1 Score: {avg_f1:.4f}")

    best_C = max(results, key=results.get)
    print(f"Best C selected: {best_C} with Average F1 Score: {results[best_C]:.4f}")

    # Train final model
    print(f"\nTraining final model with C={best_C}")
    final_model = SVM(C=best_C, learning_rate=0.1, max_iter=100)
    final_model.fit(X_train_tfidf, y_train_svm)
    
    # Evaluate on Test Set
    test_predictions = final_model.predict(X_test_tfidf)

    # Get evaluation statistics
    precision, recall, accuracy, f1_score = get_eval_statistics(y_test, test_predictions)
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    
if __name__ == "__main__":
    main()
