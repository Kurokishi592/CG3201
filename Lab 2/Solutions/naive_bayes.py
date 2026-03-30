import numpy as np 
from utils import prepare_train_test, precision_recall_curve, plot_pr_curve, compute_auc, get_eval_statistics

class NaiveBayes:
    def __init__(self):
        self.prior_spam = None
        self.log_prob_spam = None
        self.log_prob_ham = None

    def fit(self, X_train, y_train):
        """
        Train the Naive Bayes model using numpy arrays.
        X_train: (n_samples, n_features)
        y_train: (n_samples,)
        """
        n_samples, vocab_size = X_train.shape
        self.prior_spam = np.mean(y_train)
        
        # Separate spam and ham
        spam_mask = (y_train == 1)
        ham_mask = (y_train == 0)
        
        # Count occurrences of each word in spam and ham
        spam_word_counts = np.sum(X_train[spam_mask], axis=0)
        ham_word_counts = np.sum(X_train[ham_mask], axis=0)
        
        # Total words in each class
        total_spam_words = np.sum(spam_word_counts)
        total_ham_words = np.sum(ham_word_counts)
        
        # Laplacian smoothing        
        self.log_prob_spam = np.log((spam_word_counts + 1) / (total_spam_words + vocab_size))
        self.log_prob_ham = np.log((ham_word_counts + 1) / (total_ham_words + vocab_size))

    def compute_scores(self, X_test):
        """
        Compute the log odds ratio (scores) for positive class (spam)
        """
        spam_scores = X_test @ self.log_prob_spam + np.log(self.prior_spam)
        ham_scores = X_test @ self.log_prob_ham + np.log(1 - self.prior_spam)
        
        return spam_scores - ham_scores

    def predict(self, X_test):
        """
        Use bayesian decision to predict.
        """
        scores = self.compute_scores(X_test)
        return (scores > 0).astype(int)

if __name__ == '__main__':
    X_train, X_test, y_train, y_test, vocab = prepare_train_test()
    
    nb = NaiveBayes()
    nb.fit(X_train, y_train)

    y_scores = nb.compute_scores(X_test)
    y_pred = nb.predict(X_test)
    # Get eval statistics
    precision, recall, accuracy, f1_score = get_eval_statistics(y_test, y_pred)
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1_score}")

    # Compute scores for PR curve
    precision, recall = precision_recall_curve(y_test, y_scores)
    pr_auc = compute_auc(recall, precision)
    print(f"PR AUC: {pr_auc}")

    # Plot PR curve
    plot_pr_curve(recall, precision, pr_auc, title='Precision-Recall Curve (Naive Bayes)')