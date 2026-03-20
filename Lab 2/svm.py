import numpy as np
import pandas as pd
from utils import prepare_train_test, precision_recall_curve, plot_pr_curve, compute_auc, get_eval_statistics
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import argparse

class TFIDF:
    def __init__(self):
        self.idf = None

    def fit(self, X_train):
        """
        Find the IDF values for each word in the vocabulary.
        """
        X_train = np.asarray(X_train, dtype=float)
        n_docs = X_train.shape[0]

        # df(t): number of documents containing term t
        df = np.sum(X_train > 0, axis=0)
        # idf(t) = ln(1 + n/(1+df(t))) + 1
        self.idf = np.log(1.0 + n_docs / (1.0 + df)) + 1.0

    def transform(self, X):
        """
        Tranform each email features using IDF values.
        """
        if self.idf is None:
            raise ValueError("TFIDF must be fit before calling transform().")

        X = np.asarray(X, dtype=float)
        tfidf = X * self.idf

        row_sums = tfidf.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        tfidf = tfidf / row_sums

        return tfidf

class SVM:
    def __init__(
        self,
        C=1.0,
        learning_rate=1,
        max_iter=100,
        batch_size=64,
        tol=0.0001,
        decay_step=10,
        decay_gamma=0.7,
    ):
        self.C = C
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.tol = tol
        self.decay_step = int(decay_step)
        self.decay_gamma = float(decay_gamma)
        self.w = None
        self.b = None
        self.loss_history_ = None

    def _compute_objective(self, X, y_pm1):
        scores = X @ self.w + self.b
        hinge = np.maximum(0.0, 1.0 - y_pm1 * scores)
        return 0.5 * float(self.w @ self.w) + self.C * float(np.mean(hinge))

    def fit(self, X_train, y_train):
        """
        Train the SVM model using mini-batch gradient descent.
        """
        X_train = np.asarray(X_train, dtype=float)
        y_train = np.asarray(y_train, dtype=int)
        y = np.where(y_train == 1, 1.0, -1.0)

        n_samples, n_features = X_train.shape
        self.w = np.zeros(n_features, dtype=float)
        self.b = 0.0

        self.loss_history_ = []

        rng = np.random.default_rng(42)
        lr0 = float(self.learning_rate)
        prev_loss = np.inf

        # Step decay: decay every N epochs
        decay_step = max(self.decay_step, 1)
        decay_gamma = self.decay_gamma

        for epoch in range(self.max_iter):
            lr = lr0 * (decay_gamma ** (epoch // decay_step))
            perm = rng.permutation(n_samples)

            for start in range(0, n_samples, self.batch_size):
                idx = perm[start : start + self.batch_size]
                Xb = X_train[idx]
                yb = y[idx]
                m = float(len(idx))

                scores = Xb @ self.w + self.b
                margins = yb * scores
                violating = margins < 1.0

                if np.any(violating):
                    Xv = Xb[violating]
                    yv = yb[violating]
                    # Standard mini-batch gradient for objective:
                    #   0.5||w||^2 + C * mean_i hinge_i
                    w_grad = self.w - (self.C / m) * (Xv.T @ yv)
                    b_grad = -(self.C / m) * float(np.sum(yv))
                else:
                    w_grad = self.w
                    b_grad = 0.0

                self.w -= lr * w_grad
                self.b -= lr * b_grad

            loss = self._compute_objective(X_train, y)
            self.loss_history_.append(loss)

            if abs(prev_loss - loss) < self.tol:
                break
            prev_loss = loss
    
    def compute_scores(self, X):
        """
        Compute the scores for each email, 
        basically find the dot product of X and w and add b.
        """
        X = np.asarray(X, dtype=float)
        return X @ self.w + self.b

    def predict(self, X_test):
        """
        Predict the label for each email
        """
        scores = self.compute_scores(X_test)
        return (scores >= 0).astype(int)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plot-training-curves",
        action="store_true",
        help="Plot 4-fold training loss vs epoch curves for C in {1,100,10000,1000000}.",
    )
    parser.add_argument("--lr0", type=float, default=1e-3, help="Base learning rate (before decay).")
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        help="Step-decay multiplier applied every 10 epochs.",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs for training-curve plots.")
    args = parser.parse_args()

    X_train, X_test, y_train, y_test, vocab = prepare_train_test()

    if args.plot_training_curves:
        # 4-fold training curves for specified C values
        C_values = [1, 100, 10000, 1000000]
        kf = KFold(n_splits=4, shuffle=True, random_state=42)

        for C in C_values:
            plt.figure(figsize=(7, 5))
            for fold, (train_idx, _) in enumerate(kf.split(X_train), start=1):
                X_tr_raw = X_train[train_idx]
                y_tr = y_train[train_idx]

                tfidf = TFIDF()
                tfidf.fit(X_tr_raw)
                X_tr = tfidf.transform(X_tr_raw)

                # Use a consistent SGD SVM method + step learning-rate decay.
                # If curves explode for huge C, lower --lr0.
                model = SVM(C=float(C), learning_rate=float(args.lr0), max_iter=int(args.epochs), batch_size=64, tol=0.0)
                model.decay_step = 10
                model.decay_gamma = float(args.gamma)
                model.fit(X_tr, y_tr)

                epochs = np.arange(1, len(model.loss_history_) + 1)
                plt.plot(epochs, model.loss_history_, label=f"Fold {fold}")

            plt.xlabel("Epoch")
            plt.ylabel("Training loss")
            plt.title(f"SVM training curve (4 folds) - C={C} (LR decays every 10 epochs)")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.show()

    else:
        """Train SVM model, select best C using cross validation, then evaluate on test set."""
        # Cross validation to select C (4-fold)
        C_values = [0.01, 0.1, 1.0, 10.0, 100.0]
        kf = KFold(n_splits=4, shuffle=True, random_state=42)

        best_C = None
        best_f1 = -1.0

        for C in C_values:
            fold_f1 = []
            for train_idx, val_idx in kf.split(X_train):
                X_tr_raw = X_train[train_idx]
                y_tr = y_train[train_idx]
                X_val_raw = X_train[val_idx]
                y_val = y_train[val_idx]

                tfidf_fold = TFIDF()
                tfidf_fold.fit(X_tr_raw)
                X_tr = tfidf_fold.transform(X_tr_raw)
                X_val = tfidf_fold.transform(X_val_raw)

                model = SVM(C=C, learning_rate=0.001, max_iter=300, batch_size=64, tol=1e-4, decay_step=10, decay_gamma=0.7)
                model.fit(X_tr, y_tr)
                y_val_pred = model.predict(X_val)
                _, _, _, f1 = get_eval_statistics(y_val, y_val_pred)
                fold_f1.append(f1)

            avg_f1 = float(np.mean(fold_f1))
            print(f"C={C:<6} avg_f1={avg_f1:.4f}")
            if avg_f1 > best_f1:
                best_f1 = avg_f1
                best_C = C

        print(f"Best C from 4-fold CV: {best_C} (avg F1={best_f1:.4f})")

        tfidf_final = TFIDF()
        tfidf_final.fit(X_train)
        X_train_tfidf = tfidf_final.transform(X_train)
        X_test_tfidf = tfidf_final.transform(X_test)

        final_model = SVM(C=best_C, learning_rate=0.001, max_iter=300, batch_size=64, tol=1e-4, decay_step=10, decay_gamma=0.7)
        final_model.fit(X_train_tfidf, y_train)
        y_pred = final_model.predict(X_test_tfidf)

        accuracy, precision, recall, f1 = get_eval_statistics(y_test, y_pred)
        print("SVM (TF-IDF features)")
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1-score : {f1:.4f}")
        
        "start by 0.1, then every 10 epoch multiply learning rate by 0.5"