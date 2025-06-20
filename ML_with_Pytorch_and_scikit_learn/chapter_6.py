# ----------------Learning Best Practices for Model Evaluation and Hyperparameter Tuning--------------------
# Loading the Breast Cancer Wisconsin dataset
import pandas as pd
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data',
                 header=None)

# Next, we will assign the 30 features to a NumPy array, X. Using a LabelEncoder object, we will
# transform the class labels from their original string representation ('M' and 'B') into integers:
from sklearn.preprocessing import LabelEncoder
X = df.iloc[:, 2:].values
y = df.iloc[:, 1].values
le = LabelEncoder()
y = le.fit_transform(y)
# print(le.classes_)

# After encoding the class labels (diagnosis) in an array, y, the malignant tumors are now rep- resented
# as class 1, and the benign tumors are represented as class 0, respectively. We can double-check this mapping
# by calling the transform method of the fitted LabelEncoder on two dummy class labels:
# print(le.transform(['M', 'B']))

# Before we construct our first model pipeline in the following subsection, let’s divide the dataset into a separate
# training dataset (80 percent of the data) and a separate test dataset (20 percent of the data):
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=1)

# Combining transformers and estimators in a pipeline

# Instead of going through the model fitting and data transformation steps for the training and test datasets separately,
# we can chain the StandardScaler, PCA, and LogisticRegression objects in a pipeline:
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

pipe_lr = make_pipeline(StandardScaler(), PCA(n_components=2), LogisticRegression())
pipe_lr.fit(X_train, y_train)
y_pred = pipe_lr.predict(X_test)
test_acc = pipe_lr.score(X_test, y_test)
# print(f'Test accuracy: {test_acc:3f}')

# In stratified cross-validation, the class label proportions are preserved in each fold to ensure that each
# fold is representative of the class proportions in the training dataset, which we will illustrate by using
# the StratifiedKFold iterator in scikit-learn:
import numpy as np
from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=10).split(X_train ,y_train)
scores = []
for k, (train, test) in enumerate(kfold):
    pipe_lr.fit(X_train[train], y_train[train])
    score = pipe_lr.score(X_train[test], y_train[test])
    scores.append(score)
    # print(f'Fold: {k+1:02d}, '
    #       f'Class distr.: {np.bincount(y_train[train])}, '
    #       f'Acc.: {score:.3f}')

mean_acc = np.mean(scores)
std_acc = np.std(scores)
# print(f'\nCV accuracy: {mean_acc:.3f} +/- {std_acc:.3f}')

# Although the previous code example was useful to illustrate how k-fold cross-validation works,
# scikit- learn also implements a k-fold cross-validation scorer, which allows us to evaluate our model using
# stratified k-fold cross-validation less verbosely:
from sklearn.model_selection import cross_val_score
scores = cross_val_score(estimator=pipe_lr, X=X_train, y=y_train, cv=10, n_jobs=1)
# print(f'CV accuracy scores: {scores}')
# print(f'CV accuracy: {np.mean(scores):.3f}'
#       f'+/- {np.std(scores):.3f}')


# but let’s first see how we can use the learning curve function from scikit-learn to evaluate the model:
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

pipe_lr = make_pipeline(StandardScaler(), LogisticRegression(penalty='l2', max_iter=10000))
train_sizes, train_scores, test_scores = learning_curve(estimator=pipe_lr, X=X_train, y=y_train,
                                                        train_sizes=np.linspace(0.1, 1.0, 10),
                                                        cv=10, n_jobs=1)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
# plt.plot(train_sizes, train_mean, color='blue', marker='o',
#          markersize=5, label='Training accuracy')
#
# plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std,
#                  alpha=0.15, color='blue')
#
# plt.plot(train_sizes, test_mean, color='green', linestyle='--',
#          marker='s', markersize=5, label='Validation accuracy')
#
# plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
# plt.grid()
# plt.xlabel('Number of training examples')
# plt.ylabel('Accuracy')
# plt.legend(loc='lower right')
# plt.ylim([0.8, 1.03])
# plt.show()

# Let’s go ahead and see how we create validation curves via scikit-learn:
from sklearn.model_selection import validation_curve
param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
train_scores, test_scores = validation_curve(estimator=pipe_lr, X=X_train, y=y_train,
                                             param_name='logisticregression__C',
                                             param_range=param_range,
                                             cv=10)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
# plt.plot(param_range, train_mean, color='blue',
#          marker='o', markersize=5, label='Training accuracy')
# plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
#
# plt.plot(param_range, test_mean, color='green', linestyle='--',
#          marker='s', markersize=5, label='Validation accuracy')
#
# plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
# plt.grid()
# plt.xscale('log')
# plt.legend(loc='lower right')
# plt.xlabel('Parameter C')
# plt.ylabel('Accuracy')
# plt.ylim([0.8, 1.0])
# plt.show()

# Tuning hyperparameters via grid search
# The grid search approach is quite simple: it’s a brute-force exhaustive search paradigm where we specify a
# list of values for different hyperparameters, and the computer evaluates the model performance for each combination
# to obtain the optimal combination of values from this set:
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

pipe_svc = make_pipeline(StandardScaler(), SVC(random_state=1))
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = [{'svc__C':param_range, 'svc__kernel': ['linear']},
              {'svc__C':param_range, 'svc__gamma': param_range, 'svc__kernel':['rbf']}]
gs = GridSearchCV(estimator=pipe_svc, param_grid=param_grid, scoring='accuracy',
                  cv=10, refit=True, n_jobs=-1)
gs = gs.fit(X_train, y_train)
# print(gs.best_score_)
# print(gs.best_params_)

# Finally, we use the independent test dataset to estimate the performance of the best-selected model,
# which is available via the best_estimator_ attribute of the GridSearchCV object:
clf = gs.best_estimator_
clf.fit(X_train, y_train)
# print(f'Test accuracy: {clf.score(X_test, y_test):.3f}')


# Let’s now see the RandomizedSearchCV in action and tune an SVM as we did with GridSearchCV in the previous section:
from sklearn.model_selection import RandomizedSearchCV
pipe_svc = make_pipeline(StandardScaler(), SVC(random_state=1))
param_grid = [{'svc__C':param_range, 'svc__kernel': ['linear']},
              {'svc__C':param_range, 'svc__gamma': param_range, 'svc__kernel':['rbf']}]
rs = RandomizedSearchCV(estimator=pipe_svc, param_distributions=param_grid,
                        scoring='accuracy', refit=True,
                        n_iter=20, cv=10, random_state=1, n_jobs=-1)
rs = rs.fit(X_train, y_train)
# print(rs.best_score_)
# print(rs.best_params_)


# Taking the idea of randomized search one step further, scikit-learn implements a successive halving variant,
# HalvingRandomSearchCV, that makes finding suitable hyperparameter configurations more efficient. Successive
# halving, given a large set of candidate configurations, successively throws out unpromising hyperparameter
# configurations until only one configuration remains. We can summarize the procedure via the following steps:
from sklearn.experimental import enable_halving_search_cv

# After enabling the experimental support, we can use randomized search with successive halving as shown in the following:
from sklearn.model_selection import HalvingRandomSearchCV
hs = HalvingRandomSearchCV(pipe_svc, param_distributions=param_grid, n_candidates='exhaust',
                           resource='n_samples', factor=1.5, random_state=1, n_jobs=-1)

# We can then carry out the search similar to RandomizedSearchCV:
hs = hs.fit(X_train, y_train)
# print(hs.best_score_)
# print(hs.best_params_)
clf = hs.best_estimator_
# print(f'Test accuracy: {hs.score(X_test, y_test):.3f}')

# In scikit-learn, we can perform nested cross-validation with grid search as follows:
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = [{'svc__C':param_range, 'svc__kernel': ['linear']},
              {'svc__C':param_range, 'svc__gamma': param_range, 'svc__kernel':['rbf']}]
gs = GridSearchCV(estimator=pipe_svc, param_grid=param_grid, scoring='accuracy', cv=2)
scores = cross_val_score(gs, X_train, y_train, scoring='accuracy', cv=5)
# print(f'CV accuracy: {np.mean(scores):.3f}'
#       f'+/- {np.std(scores):.3f}')

# For example, we can use the nested cross-validation approach to compare an SVM model to a simple
# decision tree classifier; for simplicity, we will only tune its depth parameter:
from sklearn.tree import DecisionTreeClassifier
gs = GridSearchCV(estimator=DecisionTreeClassifier(random_state=0),
                  param_grid=[{'max_depth':[1,2,3,4,5,6,7, None]}],
                  scoring='accuracy', cv=2)
scores = cross_val_score(gs, X_train, y_train, scoring='accuracy', cv=5)
# print(f'CV accuracy: {np.mean(scores):.3f}'
#       f'+/- {np.std(scores):.3f}')



# scikit-learn provides a convenient confusion_matrix function that we can use, as follows:
from sklearn.metrics import confusion_matrix
pipe_svc.fit(X_train, y_train)
y_pred = pipe_svc.predict(X_test)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
# print(confmat)

# We can map this information onto the confusion matrix illustration in Figure 6.9 using Matplotlib’s matshow function:
fig, ax = plt.subplots(figsize=(2.5, 2.5))
plt.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

# ax.xaxis.set_ticks_position('bottom')
# plt.xlabel('Predicted label')
# plt.ylabel('True label')
# plt.show()













