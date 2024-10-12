import numpy as np
import matplotlib.pyplot as plt
class Perceptron:
    """Perceptron classifier.
        Parameters
        ------------
        eta : float
          Learning rate (between 0.0 and 1.0)
        n_iter : int
          Passes over the training dataset.
        random_state : int
          Random number generator seed for random weight
          initialization.
        Attributes
        -----------
        w_ : 1d-array
          Weights after fitting.
        b_ : Scalar
          Bias unit after fitting.
        errors_ : list
          Number of misclassifications (updates) in each epoch.
    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Fit training data.
            Parameters
            ----------
            X : {array-like}, shape = [n_examples, n_features]
              Training vectors, where n_examples is the number of
              examples and n_features is the number of features.
            y : array-like, shape = [n_examples]
              Target values.
            Returns
            -------
            self : object
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        # self.w_ = np.zeros(X.shape[1])
        self.b_ = np.float_(0.)
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_ += update * xi
                self.b_ += update
                errors += int(update != 0.0)
            self.errors_.append(errors)

        return self


    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, 0)


# As an optional exercise after reading this chapter, you can change
# self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1]) to self.w_ = np.zeros(X.shape[1])
# and run the perceptron train- ing code presented in the next section with different values for eta.
# You will observe that the decision boundary does not change.


# # Create a simple dataset for training
# X = np.array([[1, 1], [2, 1], [2, 2], [3, 3]])
# y = np.array([0, 0, 1, 1])
#
# # Try different values of the learning rate (eta)
# etas = [0.001, 0.01, 0.1]
#
# # Plot decision boundaries for different eta values
# plt.figure(figsize=(10, 7))
#
# for i, eta in enumerate(etas):
#     ppn = Perceptron(eta=eta, n_iter=10)
#     ppn.fit(X, y)
#
#     # Plot the training data
#     plt.subplot(1, len(etas), i + 1)
#     plt.scatter(X[:, 0], X[:, 1], c=y, marker='o', label='Training points')
#
#     # Plot decision boundary (weight vector perpendicular to boundary)
#     x1 = np.linspace(0, 4, 100)
#     x2 = -(ppn.w_[0] * x1 + ppn.b_) / ppn.w_[1]
#     plt.plot(x1, x2, label=f'eta={eta}', color='r')
#
#     plt.title(f'Learning rate: {eta}')
#     plt.xlabel('Feature 1')
#     plt.ylabel('Feature 2')
#
# plt.tight_layout()
# plt.show()


# Training a perceptron model on the Iris dataset
import os
import pandas as pd

s = 'https://archive.ics.uci.edu/ml/'\
'machine-learning-databases/iris/iris.data'
print(f'from url: {s}')

df = pd.read_csv(s, header=None, encoding='utf-8')
# print(df.tail())

# select setosa and versicolor
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', 0, 1)
#extract sepal length and petal length
X = df.iloc[0:100, [0,2]].values

#plot data
plt.scatter(X[:50, 0], X[:50, 1], color='red',
            marker='o', label='Setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue',
            marker='s', label='Versicolor')
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')
plt.show()


# Now, it’s time to train our perceptron algorithm on the Iris data subset that we just
# extracted. Also, we will plot the misclassification error for each epoch to check whether the
# algorithm converged and found a decision boundary that separates the two Iris flower classes:

ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X,y)
plt.plot(range(1, len(ppn.errors_)+1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.show()









