#------------------A Tour of Machine Learning Classifiers Using Scikit-Learn--------------

# First steps with scikit-learn – training a perceptron
from sklearn import datasets
import numpy as np

# We will assign the petal length and petal width of the 150 flower examples to the feature matrix, X,
# and the corresponding class labels of the flower species to the vector array, y:
iris = datasets.load_iris()
X = iris.data[:, [2,3]]
y = iris.target
# print("class labels:", np.unique(y))

# To evaluate how well a trained model performs on unseen data, we will further split the
# dataset into separate training and test datasets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

# We can use NumPy’s bincount function, which counts the number of occurrences
# of each value in an array, to verify that this is indeed the case:
# print('Labels counts in y:', np.bincount(y))
# print('Labels count in y_train:', np.bincount(y_train))
# print('Labels count in y_test:', np.bincount(y_test))


# Many machine learning and optimization algorithms also require feature scaling for optimal performance,
# as we saw in the gradient descent example in Chapter 2. Here, we will standardize the features using the
# StandardScaler class from scikit-learn’s preprocessing module:
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# Having standardized the training data, we can now train a perceptron model. Most algorithms in scikit-learn
# already support multiclass classification by default via the one-versus-rest (OvR) method, which allows us to
# feed the three flower classes to the perceptron all at once. The code is as follows:
from sklearn.linear_model import Perceptron
ppn = Perceptron(eta0=0.1,random_state=1)
ppn.fit(X_train_std, y_train)

y_pred = ppn.predict(X_test_std)
# print('Misclassified examples: %d' % (y_test != y_pred).sum())

from sklearn.metrics import accuracy_score
# print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
# print('Accuracy: %.3f' % ppn.score(X_test_std, y_test))

# Finally, we can use our plot_decision_regions function from Chapter 2 to plot the decision regions
# of our newly trained perceptron model and visualize how well it separates the different flower examples
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    #setup marker generator and color map
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    #plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    #plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=f'Class {cl}',
                    edgecolors='black')
        # highlight test examples
        if test_idx:
            #plot all examples
            X_test, y_test = X[test_idx, :], y[test_idx]
            plt.scatter(X_test[:, 0], X_test[:, 1],
                        c='none', edgecolors='black', alpha=1.0,
                        linewidths=1, marker='o',
                        s=100, label='Test set')

# With the slight modification that we made to the plot_decision_regions function, we can now
# specify the indices of the examples that we want to mark on the resulting plots. The code is as follows:
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
# plot_decision_regions(X=X_combined_std, y=y_combined, classifier=ppn, test_idx=range(105, 150))
# plt.xlabel('Petal length [standardized]')
# plt.ylabel('Petal width [standardized]')
# plt.legend(loc='upper left')
# plt.tight_layout()
# plt.show()


# Now, let’s simply plot the sigmoid function for some values in the range –7 to 7 to see how it looks:
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))
# z = np.arange(-7, 7, 0.1)
# sigma_z = sigmoid(z)
# plt.plot(z, sigma_z)
# plt.axvline(0.0, color='k')
# plt.ylim(-0.1, 1.1)
# plt.xlabel('z')
# plt.ylabel('$\sigma (z)$')
# # y axis ticks and gridline
# plt.yticks([0.0, 0.5, 1.0])
# ax = plt.gca()
# ax.yaxis.grid(True)
# plt.tight_layout()
# plt.show()


# Let's write a short code snippet to create a plot that illustrates the loss of classifying
# a single training example for different values of σ(z):
def loss_1(z):
    return - np.log(sigmoid(z))

def loss_0(z):
    return - np.log(1 - sigmoid(z))
z = np.arange(-10, 10, 0.1)
sigma_z = sigmoid(z)
c1 = [loss_1(x) for x in z]
# plt.plot(sigma_z, c1, label='L(w,b) if y=1')
c0 = [loss_0(x) for x in z]
# plt.plot(sigma_z, c0, linestyle='--', label='L(w,b) if y=0')
# plt.ylim(0.0, 5.1)
# plt.xlim([0, 1])
# plt.xlabel('$\sigma(z)$')
# plt.ylabel('L(w,b)')
# plt.legend(loc='best')
# plt.tight_layout()
# plt.show()

class LogisticRegressionGD:
    """Gradient descent-based logistic regression classifier.
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
          Weights after training.
        b_ : Scalar
          Bias unit after fitting.
        losses_ : list
          Mean squared error loss function values in each epoch.
    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """ Fit training data.
            Parameters
            ----------
            X : {array-like}, shape = [n_examples, n_features]
              Training vectors, where n_examples is the
              number of examples and n_features is the
              number of features.
            y : array-like, shape = [n_examples]
              Target values.
            Returns
            -------
            self : Instance of LogisticRegressionGD
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01,size=X.shape[1])
        self.b_ = np.float_(0.)
        self.losses_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
            self.b_ += self.eta * 2.0 * errors.mean()
            loss = (-y.dot(np.log(output)) - ((1-y).dot(np.log(1-output))) / X.shape[0])
            self.losses_.append(loss)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_) + self.b_

    def activation(self, z):
        """Compute logistic sigmoid activation"""
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)


# So, let’s consider only setosa and versicolor flowers (classes 0 and 1) and check that
# our implemen- tation of logistic regression works:
X_train_01_subset = X_train_std[(y_train == 0) | (y_train == 1)]
y_train_01_subset = y_train[(y_train == 0) | (y_train == 1)]
lrgd = LogisticRegressionGD(eta=0.3, n_iter=1000, random_state=1)
lrgd.fit(X_train_01_subset, y_train_01_subset)
# plot_decision_regions(X_train_01_subset, y_train_01_subset, classifier=lrgd)
# plt.xlabel('Petal length [standardized]')
# plt.ylabel('Petal width [standardized]')
# plt.legend(loc='upper left')
# plt.tight_layout()
# plt.show()

# Training a logistic regression model with scikit-learn
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=100.0, solver='lbfgs', multi_class='ovr')
lr.fit(X_train_std, y_train)
# plot_decision_regions(X_combined_std, y_combined, classifier=lr, test_idx=range(105, 150))
# plt.xlabel('Petal length [standardized]')
# plt.ylabel('Petal width [standardized]')
# plt.legend(loc='upper left')
# plt.tight_layout()
# plt.show()


# we can predict the probabilities of the first three examples in the test dataset as follows:
# print(lr.predict_proba(X_test_std[:3, :]))

# we can get the predicted class labels by identifying the largest column in each row, for ex- ample, using NumPy’s argmax function:
# print(lr.predict_proba(X_test_std[:3, :]).argmax(axis=1))

# print(lr.predict(X_test_std[0, :].reshape(1, -1)))


# fitting 10 logistic regression models with different values for the inverse-regularization parameter, C
# weights, params = [], []
# for c in np.arange(-5, 5):
#     lr = LogisticRegression(C=10.**c,multi_class='ovr')
#     lr.fit(X_train_std, y_train)
#     weights.append(lr.coef_[1])
#     params.append(10.**c)
# weights = np.array(weights)
# plt.plot(params, weights[:, 0], label='Petal length')
# plt.plot(params, weights[:, 1], linestyle='--', label='Petal width')
# plt.ylabel('Weight coefficient')
# plt.xlabel('C')
# plt.legend(loc='upper left')
# plt.xscale('log')
# plt.show()


# Now that we have learned the basic concepts behind a linear SVM, let’s train
# an SVM model to classify the different flowers in our Iris dataset:
from sklearn.svm import SVC

svm = SVC(kernel='linear', C=1.0, random_state=1)
svm.fit(X_train_std, y_train)
# plot_decision_regions(X_combined_std, y_combined, classifier=svm, test_idx=range(105, 150))
# plt.xlabel('Petal length [standardized]')
# plt.ylabel('Petal width [standardized]')
# plt.legend(loc='upper left')
# plt.tight_layout()
# plt.show()


# We could initialize the SGD version of the perceptron (loss='perceptron'), logistic regression (loss='log'),
# and an SVM with default parameters (loss='hinge'), as follows:
from sklearn.linear_model import SGDClassifier
ppn = SGDClassifier(loss='perceptron')
lr = SGDClassifier(loss='log')
svm = SGDClassifier(loss='hinge')


# Using the following code, we will create a simple dataset that has the form of an XOR gate
# using the logical_or function from NumPy, where 100 examples will be assigned the class label 1,
# and 100 examples will be assigned the class label -1:
np.random.seed(1)
X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:, 0] > 0,
                       X_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, 0)
# plt.scatter(X_xor[y_xor==1, 0],
#             X_xor[y_xor==1, 1],
#             c='royalblue', marker='s', label='Class 1')
# plt.scatter(X_xor[y_xor==0,0],
#             X_xor[y_xor==0,1],
#             c='tomato', marker='o', label='Class 0')
# plt.xlim([-3, 3])
# plt.ylim([-3, 3])
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.legend(loc='best')
# plt.tight_layout()
# plt.show()


# Here, we simply use the SVC class from scikit-learn that we imported
# earlier and replace the kernel='linear' parameter with kernel='rbf':

svm = SVC(kernel='rbf', random_state=1, gamma=0.10, C=10.0)
svm.fit(X_xor, y_xor)
# plot_decision_regions(X_xor, y_xor, classifier=svm)
# plt.legend(loc='upper left')
# plt.tight_layout()
# plt.show()

#to get better understanding of y, lets apply an RBF kernel SVM to our Iris flower dataset:
svm = SVC(kernel='rbf', random_state=1, gamma=0.2, C=1.0)
svm.fit(X_train_std, y_train)
# plot_decision_regions(X_combined_std, y_combined, classifier=svm, test_idx=range(105, 150))
# plt.xlabel('Petal length [standardized]')
# plt.ylabel('Petal width [standardized]')
# plt.legend(loc='upper left')
# plt.tight_layout()
# plt.show()


#now lets increase the value of y and observe the effect on the decision bondary
svm = SVC(kernel='rbf', random_state=1, gamma=100.0, C=1.0)
svm.fit(X_train_std, y_train)
# plot_decision_regions(X_combined_std, y_combined, classifier=svm, test_idx=range(105, 150))
# plt.xlabel('Petal length [standardized]')
# plt.ylabel('Petal width [standardized]')
# plt.legend(loc='upper left')
# plt.tight_layout()
# plt.show()

# To provide a visual intuition, let us visualize the entropy values
# for different class distributions via the following code:
def entropy(p):
    return - p * np.log2(p) - (1 - p) * np.log2((1-p))
x = np.arange(0.0, 1.0, 0.01)
ent = [entropy(p) if p!=0 else None for p in x]
# plt.ylabel('Entropy')
# plt.xlabel('Class-membership probability p(i=1)')
# plt.plot(x, ent)
# plt.show()

# For a more visual comparison of the three different impurity criteria that we discussed previously,
# let’s plot the impurity indices for the probability range [0, 1] for class 1. Note that we will also
# add a scaled version of the entropy (entropy / 2) to observe that the Gini impurity is an intermediate
# measure between entropy and the classification error. The code is as follows:
def gini(p):
    return p*(1 - p) + (1 - p)*(1 - (1-p))
def entropy(p):
    return - p*np.log2(p) - (1 - p)*np.log2((1-p))
def error(p):
    return 1 - np.max([p, 1-p])
x = np.arange(0.0, 1.0, 0.01)
ent = [entropy(p) if p!=0 else None for p in x]
sc_ent = [e*0.5 if e else None for e in ent]
err = [error(i) for i in x]
# fig = plt.figure()
# ax = plt.subplot(111)
# for i, lab, ls, c, in zip([ent, sc_ent, gini(x), err],
#                           ['Entropy', 'Entropy (scaled)', 'Gini impurity', 'Misclassification error'],
#                           ['-', '-', '--', '-.'], ['black', 'lightgray', 'red', 'green', 'cyan']):
#     line = ax.plot(x, i, label=lab, linestyle=ls, lw=2, color=c)
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=5,
#           fancybox=True, shadow=False)
# ax.axhline(y=0.5, linewidth=1, color='k', linestyle='--')
# ax.axhline(y=1.0, linewidth=1, color='k', linestyle='--')
# plt.ylim([0, 1.1])
# plt.xlabel('p(i=1)')
# plt.ylabel('impurity index')
# plt.show()

# Building a decision tree
from sklearn.tree import DecisionTreeClassifier
tree_model = DecisionTreeClassifier(criterion='gini',max_depth=4, random_state=1)
tree_model.fit(X_train, y_train)
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
# plot_decision_regions(X_combined, y_combined, classifier=tree_model, test_idx=range(105, 150))
# plt.xlabel('Petal length [cm]')
# plt.ylabel('Petal width [cm]')
# plt.legend(loc='upper left')
# plt.tight_layout()
# plt.show()

# A nice feature in scikit-learn is that it allows us to readily visualize the decision tree model after train-
# ing via the following code:
from sklearn import tree
feature_names = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width']
# tree.plot_tree(tree_model, feature_names=feature_names, filled=True)
# plt.show()


# Conveniently, we don’t have to construct the random forest classifier from
# individual decision trees by ourselves because there is already an implementation in scikit-learn that we can use:
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=25,
                                random_state=1,
                                n_jobs=2)
forest.fit(X_train, y_train)
# plot_decision_regions(X_combined,y_combined,classifier=forest,test_idx=range(105, 150))
# plt.xlabel('Petal length [cm]')
# plt.ylabel('Petal width [cm]')
# plt.legend(loc='upper left')
# plt.tight_layout()
# plt.show()

# By executing the following code, we will now implement a KNN model in scikit-learn using a Euclid- ean distance metric:
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
knn.fit(X_train_std, y_train)
# plot_decision_regions(X_combined_std, y_combined, classifier=knn, test_idx=range(105, 150))
# plt.xlabel('Petal length [standardized]')
# plt.ylabel('Petal width [standardized]')
# plt.legend(loc='upper left')
# plt.tight_layout()
# plt.show()













