import numpy as np
from ucimlrepo import fetch_ucirepo
from matplotlib import pyplot as plt

np.random.seed(0)

class SVM:
    def __init__(self, C=1.0, learning_rate=0.0001, max_iter=1000, tol=0.0001):
        self.C = C
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.w = None
        self.b = None

    def fit(self, X, y):
        
        self.w = np.zeros(X.shape[1])
        self.b = 0
        
        for it in range(self.max_iter):

            # Shuffle
            indices = np.random.permutation(X.shape[0])
            X = X[indices]
            y = y[indices]
            
            prev_w = self.w.copy()
            prev_b = self.b
            
            for i in range(len(y)):
                if y[i] * (self.w.T @ X[i] + self.b) >= 1:
                    grad_w = self.w
                    grad_b = 0
                else:
                    grad_w = self.w - self.C * y[i] * X[i]
                    grad_b = - self.C * y[i]

                self.w = self.w - self.learning_rate * grad_w
                self.b = self.b - self.learning_rate * grad_b

            if np.linalg.norm(self.w - prev_w) < self.tol and abs(self.b - prev_b) < self.tol:
                break
                
    def predict(self, X):
        return np.sign(X @ self.w + self.b)

class MultiClassSVM:
    def __init__(self, C=1.0, learning_rate=0.0001, max_iter=1000, tol=0.0001):
        self.C = C
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.labels = None
        self.classifiers = None
        
    def fit(self, X, y):
        self.labels = np.unique(y)
        self.classifiers = [SVM(C=self.C, learning_rate=self.learning_rate, max_iter=self.max_iter, tol=self.tol) for _ in range(len(self.labels))]

        for i in range(len(self.labels)):
            ytrain = np.where(y == self.labels[i], 1, -1)
            self.classifiers[i].fit(X, ytrain)

    def predict(self, X):
        scores = np.zeros((len(X), len(self.labels)))
        for i in range(len(self.labels)):
            scores[:, i] = X @ self.classifiers[i].w + self.classifiers[i].b
        return self.labels[np.argmax(scores, axis = 1)]
    
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
    Cvals = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    score_train = []
    score_test = []
    margin = []
    for C in Cvals:
        classifier = SVM(C=C)
        classifier.fit(Xtrain, ytrain)
        
        ypred = classifier.predict(Xtrain)
        score_train.append(np.mean(ypred == ytrain) * 100)
        
        ypred = classifier.predict(Xtest)
        score_test.append(np.mean(ypred == ytest) * 100)

        margin.append(2.0 / np.linalg.norm(classifier.w))

    plt.plot(Cvals, score_train, label='Training accuracy')
    plt.plot(Cvals, score_test, label = 'Test accuracy')
    plt.xlabel('C')
    plt.ylabel('Accuracy')
    plt.title('Accuracy v.s. regularization parameter')
    plt.legend()
    plt.xscale('log')
    plt.tight_layout()
    plt.show()

    plt.plot(Cvals, margin)
    plt.xlabel('C')
    plt.ylabel('Margin (2/||w||)')
    plt.title('Margin v.s. regularization parameter')
    plt.tight_layout()
    plt.xscale('log')
    plt.show()

def confusionMatrix(ytrue, ypred):
    classes = np.unique(ytrue)
    nClasses = len(classes)
    mat = np.zeros((nClasses, nClasses), dtype=int)
    
    for i in range(nClasses):
        for j in range(nClasses):
            mat[i, j] = np.sum((ytrue == classes[i]) & (ypred == classes[j]))
            
    return mat

def __main__():
    
    # SVM for binary classification
    Xtrain, ytrain, Xtest, ytest = load_data(binary=True)

    print("Binary classification")
    classifier = SVM()
    
    classifier.fit(Xtrain, ytrain)
    
    ypred = classifier.predict(Xtrain)
    print("Training accuracy: ", np.mean(ypred == ytrain))
    ypred = classifier.predict(Xtest)
    print("Test accuracy: ", np.mean(ypred == ytest))

    plot(Xtrain, ytrain, Xtest, ytest)

    # SVM for multi-class classification
    print()
    print("Multi-class classification")
    
    Xtrain, ytrain, Xtest, ytest = load_data(binary=False)
    classifier = MultiClassSVM()

    classifier.fit(Xtrain, ytrain)

    ypred = classifier.predict(Xtrain)
    print("Training accuracy: ", np.mean(ypred == ytrain))
    ypred = classifier.predict(Xtest)
    print("Test accuracy: ", np.mean(ypred == ytest))

    print("Confusion matrix:")
    print(confusionMatrix(ytest, ypred))
    
__main__()
