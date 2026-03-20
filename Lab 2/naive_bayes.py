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
        """
        X_train = np.asarray(X_train, dtype=float)
        y_train = np.asarray(y_train, dtype=int)

        n_docs, vocab_size = X_train.shape
        spam_mask = y_train == 1
        ham_mask = ~spam_mask

        n_spam = int(np.sum(spam_mask))
        self.prior_spam = n_spam / n_docs

        spam_word_counts = X_train[spam_mask].sum(axis=0)
        ham_word_counts = X_train[ham_mask].sum(axis=0)

        total_spam_words = float(spam_word_counts.sum())
        total_ham_words = float(ham_word_counts.sum())

        # Laplace smoothing: +1 in numerator, +|V| in denominator
        prob_word_given_spam = (spam_word_counts + 1.0) / (total_spam_words + vocab_size)
        prob_word_given_ham = (ham_word_counts + 1.0) / (total_ham_words + vocab_size)

        self.log_prob_spam = np.log(prob_word_given_spam)
        self.log_prob_ham = np.log(prob_word_given_ham)

    def compute_scores(self, X_test):
        """
        Compute the difference between log(spam | words) and log(ham | words)
        """
        X_test = np.asarray(X_test, dtype=float)

        log_prior_spam = np.log(self.prior_spam)
        log_prior_ham = np.log(1.0 - self.prior_spam)

        log_spam = log_prior_spam + X_test @ self.log_prob_spam
        log_ham = log_prior_ham + X_test @ self.log_prob_ham

        return log_spam - log_ham

    def predict(self, X_test):
        """
        Use bayesian decision to predict.
        """
        scores = self.compute_scores(X_test)
        # Score >= 0 means spam is more likely than ham
        return (scores >= 0).astype(int)

if __name__ == '__main__':
    """
    Train Naive Bayes model and evaluate its performance.
    """
    X_train, X_test, y_train, y_test, vocab = prepare_train_test()
    model = NaiveBayes()
    model.fit(X_train, y_train)

    y_scores = model.compute_scores(X_test)
    y_pred = (y_scores >= 0).astype(int)

    accuracy, precision, recall, f1 = get_eval_statistics(y_test, y_pred)
    print("Naive Bayes (zero-one loss decision)")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    pr_precision, pr_recall, _ = precision_recall_curve(y_test, y_scores)
    auc_score = compute_auc(pr_recall, pr_precision)
    plot_pr_curve(pr_recall, pr_precision, auc_score, title="Naive Bayes Precision-Recall Curve")
    
    """
    at the start by letting threshold be very high, all are classified as negative
    decrease threshold means a goes to 0 and b goes to 0, both leftwards
    """