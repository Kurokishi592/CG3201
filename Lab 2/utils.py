import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from pathlib import Path

def prepare_train_test():
    """
    Read data and split into 80% training and 20% testing with seed 42.
    """
    csv_path = Path(__file__).with_name("emails.csv")
    data = pd.read_csv(csv_path)
    X = data.iloc[:, 1:-1]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    vocab = X.columns
    return X_train.to_numpy(), X_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy(), vocab

def get_eval_statistics(y_true, y_pred):
    """
    Compute precision, recall and accuracy, f1-score from true and predicted labels.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    accuracy = float(np.mean(y_true == y_pred))

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return accuracy, precision, recall, f1

def compute_auc(recall, precision):
    """
    Compute area under precision-recall curve using sklearn auc
    """
    return auc(recall, precision)

def precision_recall_curve(y_true, y_scores):
    """
    Compute precision-recall pairs for different thresholds
    """
    y_true = np.asarray(y_true).astype(int)
    y_scores = np.asarray(y_scores).astype(float)

    order = np.argsort(-y_scores)
    y_scores_sorted = y_scores[order]
    y_true_sorted = y_true[order]

    total_pos = int(np.sum(y_true_sorted == 1))
    if total_pos == 0:
        precision = np.array([1.0])
        recall = np.array([0.0])
        thresholds = np.array([])
        return precision, recall, thresholds

    tp = np.cumsum(y_true_sorted == 1)
    fp = np.cumsum(y_true_sorted == 0)

    change_idx = np.where(np.diff(y_scores_sorted))[0]
    threshold_idx = np.r_[change_idx, len(y_scores_sorted) - 1]

    tp = tp[threshold_idx]
    fp = fp[threshold_idx]
    thresholds = y_scores_sorted[threshold_idx]

    precision = tp / (tp + fp)
    recall = tp / total_pos

    precision = np.r_[1.0, precision]
    recall = np.r_[0.0, recall]

    return precision, recall, thresholds

def plot_pr_curve(recall, precision, auc_score, title='Precision-Recall Curve'):
    """
    Plot Precision-Recall curve
    """
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"AUC={auc_score:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()