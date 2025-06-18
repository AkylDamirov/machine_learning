# -------------------Building Good Training Datasets – Data Preprocessing-----------------
# Identifying missing values in tabular data
import matplotlib.pyplot as plt
# Before we discuss several techniques for dealing with missing values, let’s create a simple example
# DataFrame from a comma-separated values (CSV) file to get a better grasp of the problem:
import pandas as pd
from io import StringIO
csv_data = \
'''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0, 11.0, 12.0,'''
df = pd.read_csv(StringIO(csv_data))
# print(df)

# Using the sum method, we can then return the number of missing values per column as follows:
# print(df.isnull().sum())


# Eliminating training examples or features with missing values
# One of the easiest ways to deal with missing data is simply to remove the corresponding features
# (columns) or training examples (rows) from the dataset entirely; rows with missing values can easily
# be dropped via the dropna method:
# print(df.dropna(axis=0))

# Similarly, we can drop columns that have at least one NaN in any row by setting the axis argument to 1
# print(df.dropna(axis=1))

# The dropna method supports several additional parameters that can come in handy:
# only drop rows where all columns are NaN  (returns the whole array here since we don't
# have a row with all values NaN)
# print(df.dropna(how='all'))

# drop rows that have fewer than 4 real values
# print(df.dropna(thresh=4))

# only drop rows where NaN appear in specific columns (here: 'C')
# print(df.dropna(subset=['C']))

# -Imputing missing values
# One of the most common interpolation techniques is mean imputation, where we simply replace the missing
# value with the mean value of the entire feature column. A convenient way to achieve this is by using
# the SimpleImputer class from scikit-learn, as shown in the following code:
from sklearn.impute import SimpleImputer
import numpy as np
imr = SimpleImputer(missing_values=np.nan, strategy='mean')
imr.fit(df.values)
imputed_data = imr.transform(df.values)
# print(imputed_data)

# Alternatively, an even more convenient way to impute missing values is by using pandas’
# fillna method and providing an imputation method as an argument
# print(df.fillna(df.mean()))

# Categorical data encoding with pandas
import pandas as pd
df = pd.DataFrame([
    ['green', 'M', 10.1, 'class2'],
    ['red', 'L', 13.5, 'class1'],
    ['blue', 'XL', 15.3, 'class2']])
df.columns = ['color', 'size', 'price', 'classlabel']
# print(df)

# Mapping ordinal features
# To make sure that the learning algorithm interprets the ordinal features correctly,
# we need to convert the categorical string values into integers
size_mapping = {'XL':3,
                'L':2,
                'M':1}
df['size'] = df['size'].map(size_mapping)
# print(df)

# If we want to transform the integer values back to the original string representation
# at a later stage, we can simply define a reverse-mapping dictionary
inv_size_mapping = {v:k for k, v in size_mapping.items()}
# print(df['size'].map(inv_size_mapping))

# Encoding class labels
class_mapping = {label: idx for idx, label in enumerate(np.unique(df['classlabel']))}
# print(class_mapping)
# Next, we can use the mapping dictionary to transform the class labels into integers:
df['classlabel'] = df['classlabel'].map(class_mapping)
# print(df)

# We can reverse the key-value pairs in the mapping dictionary as follows to map the converted class
# labels back to the original string representation:
inv_class_mapping = {v: k for k,v in class_mapping.items()}
df['classlabel'] = df['classlabel'].map(inv_class_mapping)
# print(df)

# Alternatively, there is a convenient LabelEncoder class directly implemented in scikit-learn to achieve this:
from sklearn.preprocessing import LabelEncoder
class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel'].values)
# print(y)

# Note that the fit_transform method is just a shortcut for calling fit and transform separately,
# and we can use the inverse_transform method to transform the integer class labels back into their original
# string representation:
# print(class_le.inverse_transform(y))

# Performing one-hot encoding on nominal features
# We could use a similar approach to transform the nominal color column of our dataset, as follows:
X = df[['color', 'size', 'price']].values
color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:, 0])
# print(X)

# A common workaround for this problem is to use a technique called one-hot encoding. The idea behind
# this approach is to create a new dummy feature for each unique value in the nominal feature column.
# Here, we would convert the color feature into three new features: blue, green, and red. Binary values
# can then be used to indicate the particular color of an example; for example, a blue example can be encoded
# as blue=1, green=0, red=0. To perform this transformation, we can use the OneHotEncoder that is implemented
# in scikit-learn’s preprocessing module:
from sklearn.preprocessing import OneHotEncoder
X = df[['color', 'size', 'price']].values
color_ohe = OneHotEncoder()
# print(color_ohe.fit_transform(X[:, 0].reshape(-1, 1)).toarray())

# Note that we applied the OneHotEncoder to only a single column, (X[:, 0].reshape(-1, 1)),
# to avoid modifying the other two columns in the array as well. If we want to selectively transform
# columns in a multi-feature array, we can use the ColumnTransformer, which accepts a list of
# (name, transformer, column(s)) tuples as follows:
from sklearn.compose import ColumnTransformer
X = df[['color', 'size', 'price']].values
c_transf = ColumnTransformer([
    ('onehot', OneHotEncoder(),[0]),
    ('nothing', 'passthrough', [1,2])
])
# print(c_transf.fit_transform(X).astype(float))
# c_transf.fit_transform(X).astype(float)

# Applied to a DataFrame, the get_dummies method will only convert string columns and leave all other columns unchanged:
# print(pd.get_dummies(df[['price', 'color', 'size']]).astype(int))

# If we use the get_dummies function, we can drop the first column by passing a True
# argument to the drop_first parameter, as shown in the following code example:
# print(pd.get_dummies(df[['price', 'color', 'size']], drop_first=True).astype(int))

# In order to drop a redundant column via the OneHotEncoder, we need to set drop='first' and set
# categories='auto' as follows:
color_ohe = OneHotEncoder(categories='auto', drop='first')
c_transf = ColumnTransformer([
    ('onehot', color_ohe, [0]),
    ('nothing', 'passthrough', [1,2])
])
# print(c_transf.fit_transform(X).astype(float))

# Partitioning a dataset into separate training and test datasets
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',
                      header=None)
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium',
    'Total phenols', 'Flavanoids',
    'Nonflavanoid phenols', 'Proanthocyanins','Color intensity',
    'Hue', 'OD280/OD315 of diluted wines', 'Proline']

# print('Class labels', np.unique(df_wine['Class label']))
# print(df_wine.head())


# A convenient way to randomly partition this dataset into separate test and training datasets
# is to use the train_test_split function from scikit-learn’s model_selection submodule:
from sklearn.model_selection import train_test_split
X,y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)



# The min-max scaling procedure is implemented in scikit-learn and can be used as follows:
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.transform(X_test)

# You can perform the standardization and normalization shown in the table manually by executing the
# following code examples:
ex = np.array([0, 1, 2, 3, 4, 5])
# print('standardized:', (ex - ex.mean()) / ex.std())
# print('normalized:', (ex - ex.min()) / (ex.max() - ex.min()))

# Similar to the MinMaxScaler class, scikit-learn also implements a class for standardization:
from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

# For regularized models in scikit-learn that support L1 regularization, we can simply set the penalty
# parameter to 'l1' to obtain a sparse solution:
from sklearn.linear_model import LogisticRegression
LogisticRegression(penalty='l1', solver='liblinear', multi_class='ovr')
# Note that we also need to select a different optimization algorithm (for example, solver='liblinear'),
# since 'lbfgs' currently does not support L1-regularized loss optimization. Applied to the standardized Wine data,
# the L1 regularized logistic regression would yield the following sparse solution:
lr = LogisticRegression(penalty='l1', C=1.0, solver='liblinear', multi_class='ovr')
#note that C=1.0 is the default. You can increase or decrease it to make the regularization effect
#stronger or weaker, respectively.
lr.fit(X_train_std, y_train)
# print('Training accuracy:', lr.score(X_train_std, y_train))
# print('Test accuracy:', lr.score(X_test_std, y_test))

# In the last example on regularization in this chapter, we will vary the regularization strength and
# plot the regularization path—the weight coefficients of the different features for different regularization strengths:
# fig = plt.figure()
# ax = plt.subplot(111)
# colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black',
#           'pink', 'lightgreen', 'lightblue', 'gray', 'indigo', 'orange']
#
# weights, params = [], []
# for c in np.arange(-4., 6.):
#     lr = LogisticRegression(penalty='l1', C=10.0**c,
#                             solver='liblinear',
#                             multi_class='ovr', random_state=0)
#     lr.fit(X_train_std, y_train)
#     weights.append(lr.coef_[1])
#     params.append(10**c)
#
# weights = np.array(weights)
# for column, color in zip(range(weights.shape[1]), colors):
#     plt.plot(params, weights[:, column],
#              label=df_wine.columns[column+1],
#              color=color)
# plt.axhline(0, color='black', linestyle='--', linewidth=3)
# plt.xlim([10**(-5), 10**5])
# plt.ylabel('Weight coefficient')
# plt.xlabel('C (inverse regularization strength)')
# plt.xscale('log')
# plt.legend(loc='upper left')
# ax.legend(loc='upper center', bbox_to_anchor=(1.38, 1.03),
#           ncol=1, fancybox=True)
# plt.show()


# Sequential feature selection algorithms
from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
class SBS:
    def __init__(self, estimator, k_features,
                 scoring=accuracy_score,
                 test_size=0.25, random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state
    def fit(self, X, y):
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=self.test_size,
                             random_state=self.random_state)
        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, y_train,
                                 X_test, y_test, self.indices_)
        self.scores_ = [score]
        while dim > self.k_features:
            scores = []
            subsets = []
            for p in combinations(self.indices_, r=dim - 1):
                score = self._calc_score(X_train, y_train, X_test, y_test, p)
                scores.append(score)
                subsets.append(p)

            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1

            self.scores_.append(scores[best])

        self.k_score_ = self.scores_[-1]
        return self

    def transform(self, X):
        return X[:, self.indices_]

    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)

        return score

# Now, let’s see our SBS implementation in action using the KNN classifier from scikit-learn:
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
sbs = SBS(knn, k_features=1)
sbs.fit(X_train_std, y_train)

k_feat = [len(k) for k in sbs.subsets_]
# plt.plot(k_feat, sbs.scores_, marker='o')
# plt.ylim([0.7, 1.02])
# plt.ylabel('Accuracy')
# plt.xlabel('Number of features')
# plt.grid()
# plt.tight_layout()
# plt.show()

# To satisfy our own curiosity, let’s see what the smallest feature subset (k=3), which yielded such a good
# performance on the validation dataset, looks like:
k3 = list(sbs.subsets_[10])
# print(df_wine.columns[1:][k3])

# Next, let’s evaluate the performance of the KNN classifier on the original test dataset:
knn.fit(X_train_std, y_train)
# print('Training accuracy:', knn.score(X_train_std, y_train))
# print('Test accuracy:',  knn.score(X_test_std, y_test))

# Now, let’s use the selected three-feature subset and see how well KNN performs:
knn.fit(X_train_std[:, k3], y_train)
# print('Training accuracy:', knn.score(X_train_std[:, k3], y_train))
# print('Training accuracy:', knn.score(X_test_std[:, k3], y_test))


# Assessing feature importance with random forests
from sklearn.ensemble import RandomForestClassifier
feat_labels = df_wine.columns[1:]
forest = RandomForestClassifier(n_estimators=500, random_state=1)
forest.fit(X_train, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
# for f in range(X_train.shape[1]):
#     print("%2d) %-*s %f" % (f+1, 30, feat_labels[indices[f]], importances[indices[f]]))
# plt.title('Feature importance')
# plt.bar(range(X_train.shape[1]), importances[indices], align='center')
# plt.xticks(range(X_train.shape[1]), feat_labels[indices], rotation=90)
# plt.xlim([-1, X_train.shape[1]])
# plt.tight_layout()
# plt.show()

# For example, we could set the threshold to 0.1 to reduce the dataset to the five most
# important features using the following code:
from sklearn.feature_selection import SelectFromModel
sfm = SelectFromModel(forest, threshold=0.1, prefit=True)
X_selected = sfm.transform(X_train)
# print('Number of features that meet this threshold',
#       'criterion', X_selected.shape[1])
# for f in range(X_selected.shape[1]):
#     print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))
















