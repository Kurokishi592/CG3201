import numpy as np
from ucimlrepo import fetch_ucirepo
from matplotlib import pyplot as plt

np.random.seed(0)

''' 
performs SGD. One SGD step picks one training sample and compute margin, before updating w,b
'''
class SVM:
    def __init__(self, C=1.0, learning_rate=0.0001, max_iter=1000):
        self.w = None
        self.b = None
        self.C = float(C)
        self.learning_rate = float(learning_rate)
        self.max_iter = int(max_iter)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).flatten() # to avoid 1 element arrays. keep scalar

        n_samples, n_features = X.shape
        # initialise to 0 as the starting point of SGD
        self.w = np.zeros(n_features, dtype=float)
        self.b = 0.0

        for _ in range(self.max_iter):
            # if iterate 0..N-1, updates become biased by dataset ordering
            # shuffling each epoch gives a closer approx to sampling a random mini batch to converge more reliably
            indices = np.random.permutation(n_samples) 
            for i in indices:
                xi = X[i]
                yi = y[i]

                margin = yi * (np.dot(self.w, xi) + self.b)
                if margin >= 1:
                    self.w -= self.learning_rate * self.w
                else:
                    self.w -= self.learning_rate * (self.w - self.C * yi * xi)
                    self.b += self.learning_rate * self.C * yi

        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.w + self.b

    def predict(self, X):
        scores = self.decision_function(X)
        ypred = np.where(scores >= 0, 1, -1)
        return ypred

class MultiClassSVM:
    def __init__(self, C=1.0, learning_rate=0.0001, max_iter=1000):
        self.labels = None
        self.classifiers = None
        self.C = float(C)
        self.learning_rate = float(learning_rate)
        self.max_iter = int(max_iter)
        
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).flatten()

        self.labels = np.unique(y)
        self.classifiers = []

        for label in self.labels:
            y_binary = np.where(y == label, 1, -1)
            clf = SVM(C=self.C, learning_rate=self.learning_rate, max_iter=self.max_iter)
            clf.fit(X, y_binary)
            self.classifiers.append(clf)

        return self
    
    def predict(self, X):
        X = np.asarray(X, dtype=float)
        scores = np.column_stack([clf.decision_function(X) for clf in self.classifiers])
        best = np.argmax(scores, axis=1)
        return self.labels[best]
    
def load_data(binary=False):
    wine_quality = fetch_ucirepo(id=186)

    X = wine_quality.data.features.to_numpy()
    y = wine_quality.data.targets.to_numpy().flatten()

    if binary:
        y = np.where(y > 5, 1, -1)

    # Shuffle
    indices = np.random.permutation(X.shape[0])
    X = X[indices]
    y = y[indices]

    # Split train and test set
    Xtrain = X[:4800]   # dimension: 4800 x 11
    Xtest = X[4800:]
    ytrain = y[:4800]   # dimension: 4800 x 1
    ytest = y[4800:]

    # Normalization
    mu, sigma = Xtrain.mean(axis=0), Xtrain.std(axis=0)
    Xtrain = (Xtrain - mu) / sigma
    Xtest = (Xtest - mu) / sigma

    return Xtrain, ytrain, Xtest, ytest

def plot(Xtrain, ytrain, Xtest, ytest):
    Cs = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    learning_rate = 0.0001
    max_iter = 1000

    train_acc = []
    test_acc = []
    margins = []

    for C in Cs:
        svm = SVM(C=C, learning_rate=learning_rate, max_iter=max_iter)
        svm.fit(Xtrain, ytrain)
        ypred_train = svm.predict(Xtrain)
        ypred_test = svm.predict(Xtest)

        train_acc.append(np.mean(ypred_train == ytrain))
        test_acc.append(np.mean(ypred_test == ytest))
        margins.append(2.0 / (np.linalg.norm(svm.w) + 1e-12))

    plt.figure()
    plt.plot(Cs, train_acc, marker='o', label='Train accuracy')
    plt.plot(Cs, test_acc, marker='o', label='Test accuracy')
    plt.xscale('log')
    plt.xlabel('C (log scale)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs C')
    plt.legend()

    plt.figure()
    plt.plot(Cs, margins, marker='o')
    plt.xscale('log')
    plt.xlabel('C (log scale)')
    plt.ylabel('Margin (2/||w||)')
    plt.title('Margin vs C')

    plt.show()

def confusionMatrix(yactual, ypredict):
    yactual = np.asarray(yactual).flatten()
    ypredict = np.asarray(ypredict).flatten()

    labels = np.unique(np.concatenate([yactual, ypredict]))
    label_to_index = {label: i for i, label in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)

    for ya, yp in zip(yactual, ypredict):
        cm[label_to_index[ya], label_to_index[yp]] += 1

    return labels, cm

def __main__():
    # SVM for binary classification
    print("Binary classification")
    Xtrain, ytrain, Xtest, ytest = load_data(binary=True)

    svm = SVM(C=1.0, learning_rate=0.0001, max_iter=1000)
    svm.fit(Xtrain, ytrain)
    ypred_train = svm.predict(Xtrain)
    ypred_test = svm.predict(Xtest)
    
    print("Accuracy on training set: ")
    print(np.mean(ypred_train == ytrain))
    print("Accuracy on test set: ")
    print(np.mean(ypred_test == ytest))
    
    plot(Xtrain, ytrain, Xtest, ytest)

    # SVM for multi-class classification
    print()
    print("Multi-class classification")
    Xtrain, ytrain, Xtest, ytest = load_data(binary=False)

    msvm = MultiClassSVM(C=1.0, learning_rate=0.0001, max_iter=1000)
    msvm.fit(Xtrain, ytrain)
    ypred_train = msvm.predict(Xtrain)
    ypred_test = msvm.predict(Xtest)

    print("Accuracy on training set: ")
    print(np.mean(ypred_train == ytrain))
    print("Accuracy on test set: ")
    print(np.mean(ypred_test == ytest))

    print("Confusion matrix:")
    labels, cm = confusionMatrix(ytest, ypred_test)
    print("Labels:", labels)
    print(cm)
    
    
__main__()

