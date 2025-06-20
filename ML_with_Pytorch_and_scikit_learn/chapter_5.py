#--------------Compressing Data via Dimensionality Reduction---------------
# Extracting the principal components step by step
# First, we will start by loading the Wine dataset that we worked with in Chapter 4
import pandas as pd
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',
                      header=None)

# Next, we will process the Wine data into separate training and test datasets—using 70 percent and 30
# percent of the data, respectively—and standardize it to unit variance:
from sklearn.model_selection import train_test_split
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

# standardize the features
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# we will use the linalg.eig function from NumPy to obtain the eigenpairs of the Wine covariance matrix:
import numpy as np
cov_mat = np.cov(X_train_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
# print('\nEigenvalues \n', eigen_vecs)
# Using the numpy.cov function, we computed the covariance matrix of the standardized training data set.
# Using the linalg.eig function, we performed the eigendecomposition, which yielded a vector (eigen_vals)
# consisting of 13 eigenvalues and the corresponding eigenvectors stored as columns
# in a 13×13-dimensional matrix (eigen_vecs).

# Using the NumPy cumsum function, we can then calculate the cumulative sum of explained variances,
# which we will then plot via Matplotlib’s step function:
tot = sum(eigen_vals)
var_exp = [(i/tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
import matplotlib.pyplot as plt
# plt.bar(range(1, 14), var_exp, align='center', label='Individual explained variance')
# plt.step(range(1, 14), cum_var_exp, where='mid', label='Cumulative explained variance')
# plt.ylabel('Explained variance ratio')
# plt.xlabel('Principal component index')
# plt.legend(loc='best')
# plt.tight_layout()
# plt.show()

# -Feature transformation
#We start by sorting the eigenpairs by decreasing order of the eigenvalues:
# Make a list of (eigenvalue, eigenvector) tuples
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i])
               for i in range(len(eigen_vals))]
# Sort the (eigenvalue, eigenvector) tuples from high to low
eigen_pairs.sort(key=lambda k: k[0], reverse=True)

# Next, we collect the two eigenvectors that correspond to the two largest eigenvalues,
# to capture about 60 percent of the variance in this dataset
w = np.hstack((eigen_pairs[0][1][:, np.newaxis],
               eigen_pairs[1][1][:, np.newaxis]))
# print('Matrix W:\n', w)

# Using the projection matrix, we can now transform an example, x (represented as a 13-dimensional row vector),
# onto the PCA subspace (the principal components one and two) obtaining x′, now a two-dimensional
# example vector consisting of two new features:
# print(X_train_std[0].dot(w))
#or
X_train_pca = X_train_std.dot(w)

# Lastly, let’s visualize the transformed Wine training dataset, now stored as an 124×2-dimensional
# matrix, in a two-dimensional scatterplot:
colors = ['r', 'b', 'g']
markers = ['o', 's', '^']
# for l, c, m in zip(np.unique(y_train), colors, markers):
#     plt.scatter(X_train_pca[y_train == l, 0],
#                 X_train_pca[y_train == l, 1],
#                 c=c, label=f'Class {l}', marker=m)
# #
# plt.xlabel('PC 1')
# plt.ylabel('PC 2')
# plt.legend(loc='lower left')
# plt.tight_layout()
# plt.show()


# Principal component analysis in scikit-learn
from matplotlib.colors import ListedColormap
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    # setup marker generator and color map
    markers=('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    #plot decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.3,cmap=cmap)
    plt.xlim(xx1.min(), xx2.max())
    plt.ylim(xx2.min(), xx2.max())

    #plot class labels
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl, 0],
                    y=X[y==cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=f'Class {cl}',
                    edgecolors='black')

from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
# initializing the PCA transformer and logistic regression estimator:
pca = PCA(n_components=2)
lr = LogisticRegression(multi_class='ovr',
                        random_state=1, solver='lbfgs')
# dimensionality reduction:
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
# fitting the logistic regression model on the reduced dataset:
lr.fit(X_train_pca, y_train)
# plot_decision_regions(X_train_pca, y_train, classifier=lr)
# plt.xlabel('PC 1')
# plt.ylabel('PC 2')
# plt.legend(loc='lower left')
# plt.tight_layout()
# plt.show()

# let’s plot the decision regions of the logistic regression on the transformed test dataset to see
# if it can separate the classes well:
# plot_decision_regions(X_test_pca, y_test, classifier=lr)
# plt.xlabel('PC 1')
# plt.ylabel('PC 2')
# plt.legend(loc='lower left')
# plt.tight_layout()
# plt.show()

# If we are interested in the explained variance ratios of the different principal components,
# we can simply initialize the PCA class with the n_components parameter set to None, so all principal
# components are kept and the explained variance ratio can then be accessed via the explained_variance_ratio_ attribute:
pca = PCA(n_components=None)
X_train_pca = pca.fit_transform(X_train_std)
# print(pca.explained_variance_ratio_)

# -Assessing feature contributions
# First, we compute the 13×13-dimensional loadings matrix by multiplying the eigenvectors
# by the square root of the eigenvalues:
loadings = eigen_vecs * np.sqrt(eigen_vals)
# Then, we plot the loadings for the first principal component, loadings[:, 0], which is the first col-
# umn in this matrix:
# fig,ax = plt.subplots()
# ax.bar(range(13), loadings[:, 0], align='center')
# ax.set_ylabel('loading for PC 1')
# ax.set_xticks(range(13))
# ax.set_xticklabels(df_wine.columns[1:], rotation=90)
# plt.ylim([-1, 1])
# plt.tight_layout()
# plt.show()

# We can obtain the loadings from a fitted scikit-learn PCA object in a similar manner, where pca.
# components_ represents the eigenvectors and pca.explained_variance_ represents the eigenvalues:
sklearn_loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

# To compare the scikit-learn PCA loadings with those we created previously, let us create a similar bar plot:
# fig, ax = plt.subplots()
# ax.bar(range(13), sklearn_loadings[:, 0], align='center')
# ax.set_ylabel('Loading for PC 1')
# ax.set_xticks(range(13))
# ax.set_xticklabels(df_wine.columns[1:], rotation=90)
# plt.ylim([-1, 1])
# plt.tight_layout()
# plt.show()


# mean vectors can be computed by the following code, where we compute one mean vector for each of the three labels
np.set_printoptions(precision=4)
mean_vecs = []
for label in range(1, 4):
    mean_vecs.append(np.mean(X_train_std[y_train==label], axis=0))
#     print(f'MV {label}: {mean_vecs[label - 1]}\n')

d = 13 #number of features
S_W = np.zeros((d,d))
for label, mv in zip(range(1,4), mean_vecs):
    class_scatter = np.zeros((d,d))
    for row in X_train_std[y_train == label]:
        row, mv = row.reshape(d, 1), mv.reshape(d, 1)
        class_scatter += (row - mv).dot((row - mv).T)
    S_W += class_scatter
# print('Whithin-class scatter matrix: '
#       f'{S_W.shape[0]}x{S_W.shape[1]}')
# print('Class label distribution:', np.bincount(y_train)[1:])

# The code for computing the scaled within-class scatter matrix is as follows:
d = 13 #number of features
S_W = np.zeros((d, d))
for label, mv in zip(range(1, 4), mean_vecs):
    class_scatter = np.cov(X_train_std[y_train==label].T)
    S_W += class_scatter

# print('Scaled within class scatter matrix: '
#       f'{S_W.shape[0]}x{S_W.shape[1]}')

# Here, m is the overall mean that is computed, including examples from all c classes:
mean_overall = np.mean(X_train_std, axis=0)
mean_overall = mean_overall.reshape(d, 1)

d = 13 #number of features
S_B = np.zeros((d, d))
for i, mean_vec in enumerate(mean_vecs):
    n = X_train_std[y_train == i + 1, :].shape[0]
    mean_vec = mean_vec.reshape(d, 1) # make column vector
    S_B += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)
# print('Between-class catter matrix: '
#       f'{S_B.shape[0]}x{S_B.shape[1]}')

# The remaining steps of the LDA are similar to the steps of the PCA. However, instead of performing
# the eigendecomposition on the covariance matrix, we solve the generalized eigenvalue problem of
# the matrix,S^-1 w * Sb
eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
# After we compute the eigenpairs, we can sort the eigenvalues in descending order:
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]
eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)
# print('Eigenvalues in descending order:\n')
# for eigen_vals in eigen_pairs:
#     print(eigen_vals[0])

# To measure how much of the class-discriminatory information is captured by the linear
# discriminants (eigenvectors), let’s plot the linear discriminants by decreasing eigenvalues
tot = sum(eigen_vals.real)
discr = [(i / tot) for i in sorted(eigen_vals.real, reverse=True)]
cum_discr = np.cumsum(discr)
# plt.bar(range(1, 14), discr, align='center', label='Individual discriminability')
# plt.step(range(1, 14), cum_discr, where='mid', label='Cumulative discriminability')
# plt.ylabel('"Discriminability" ratio')
# plt.xlabel('Linear Discriminants')
# plt.ylim([-0.1, 1.1])
# plt.legend(loc='best')
# plt.tight_layout()
# plt.show()

# Let’s now stack the two most discriminative eigenvector columns to create the transformation matrix, W:
w = np.hstack((eigen_pairs[0][1][:, np.newaxis].real, eigen_pairs[1][1][:, np.newaxis].real))
# print('Matrix W:\n', w)

# Using the transformation matrix W that we created in the previous subsection, we can now transform the training
# dataset by multiplying the matrices:
# X′ = XW
X_train_lda = X_train_std.dot(w)
colors = ['r', 'b', 'g']
markers = ['o', 's', '^']
# for l, c, m in zip(np.unique(y_train), colors, markers):
#     plt.scatter(X_train_lda[y_train == l, 0],
#                 X_train_lda[y_train == l, 1]* (-1),
#                 c=c, label=f'Class {l}', marker=m)
# plt.xlabel('LD 1')
# plt.ylabel('LD 2')
# plt.legend(loc='lower right')
# plt.tight_layout()
# plt.show()

# That step-by-step implementation was a good exercise to understand the inner workings of
# LDA and understand the differences between LDA and PCA. Now, let’s look at the LDA class implemented in scikit-learn:

# the following import statement is one line
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)
# Next, let’s see how the logistic regression classifier handles the lower-dimensional
# training dataset after the LDA transformation:
lr = LogisticRegression(multi_class='ovr', random_state=1, solver='lbfgs')
lr = lr.fit(X_train_lda,y_train)
# plot_decision_regions(X_train_lda, y_train, classifier=lr)
# plt.xlabel('LD 1')
# plt.ylabel('LD 2')
# plt.legend(loc='lower left')
# plt.tight_layout()
# plt.show()

# By lowering the regularization strength, we could probably shift the decision boundaries so that
# the logistic regression model classifies all examples in the training dataset correctly. However,
# and more importantly, let’s take a look at the results on the test dataset:
X_test_lda = lda.transform(X_test_std)
# plot_decision_regions(X_test_lda, y_test, classifier=lr)
# plt.xlabel('LD 1')
# plt.ylabel('LD 2')
# plt.legend(loc='lower left')
# plt.tight_layout()
# plt.show()



# The following code shows a quick demonstration of how t-SNE can be applied to a 64-dimensional dataset.
# First, we load the Digits dataset from scikit-learn, which consists of low-resolution handwrit- ten digits
# (the numbers 0-9):
from sklearn.datasets import load_digits
digits = load_digits()
fig, ax = plt.subplots(1,4)
# for i in range(4):
#     ax[i].imshow(digits.images[i], cmap='Greys')
# plt.show()
# print(digits.data.shape)

# Next, let us assign the features (pixels) to a new variable X_digits and the labels to another new
# variable y_digits:
y_digits = digits.target
X_digits = digits.data

# Then, we import the t-SNE class from scikit-learn and fit a new tsne object. Using fit_transform, w
# e perform the t-SNE fitting and data transformation in one step:
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, init='pca', random_state=123)
X_digits_tsne = tsne.fit_transform(X_digits)

# Finally, let us visualize the 2D t-SNE embeddings using the following code
import matplotlib.patheffects as PathEffects
def plot_projection(x, colors):
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    for i in range(10):
        plt.scatter(x[colors==i, 0],
                    x[colors==i, 1])

    for i in range(10):
        xtext, ytext =  np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground='w'),
            PathEffects.Normal()
        ])

plot_projection(X_digits_tsne, y_digits)
plt.show()












