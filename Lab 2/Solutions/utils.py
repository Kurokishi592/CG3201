import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import auc
import matplotlib.pyplot as plt

def prepare_train_test():
    """
    Read data and split into 80% training and 20% testing with seed 42.
    """
    data = pd.read_csv("emails.csv")
    X = data.iloc[:, 1:-1]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    vocab = X.columns
    return X_train.to_numpy(), X_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy(), vocab

def get_eval_statistics(y_true, y_pred):
    """
    Compute precision, recall and accuracy, f1-score from true and predicted labels.
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, accuracy, f1_score

def compute_auc(recall, precision):
    """
    Compute area under precision-recall curve using sklearn auc
    """
    return auc(recall, precision)

def precision_recall_curve(y_true, y_scores):
    """
    Compute precision-recall pairs for different thresholds
    """
    # Sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_scores)[::-1]
    y_scores = y_scores[desc_score_indices]
    y_true = y_true[desc_score_indices]
    
    # Generate distinct thresholds
    distinct_value_indices = np.where(np.diff(y_scores))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_scores.size - 1]
    
    # Calculate TPs and FPs at each threshold
    tps = np.cumsum(y_true)[threshold_idxs]
    fps = (1 + threshold_idxs) - tps
    
    total_positives = np.sum(y_true)
    
    # Calculate Precision and Recall
    precision = tps / (tps + fps)
    
    # Handle division by zero for recall if no positives
    if total_positives > 0:
        recall = tps / total_positives
    else:
        recall = np.zeros_like(tps, dtype=float)
    
    # Prepend point (Recall=0, Precision=1)
    precision = np.r_[1, precision]
    recall = np.r_[0, recall]
    
    return precision, recall

def plot_pr_curve(recall, precision, auc_score, title='Precision-Recall Curve'):
    """
    Plot Precision-Recall curve
    """
    plt.figure()
    plt.plot(recall, precision, label=f'PR curve (area = {auc_score:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower left")
    plt.show()
