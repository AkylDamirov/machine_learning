#building a good training datasets - data preprocessing
import numpy as np
import pandas as pd
from io import StringIO
csv_data = \
"""A,B,C,D
1.0, 2.0, 3.0, 4.0
5.0, 6.0,,8.0
10.0, 11.0, 12.0,"""
df = pd.read_csv(StringIO(csv_data))
# print(df)
# print(df.isnull().sum())
# print(df.values)
# print(df.dropna(axis=0))
# print(df.dropna(axis=1))

# only drop rows where all columns are NaN
# (returns the whole array here since we don't >>> # have a row with all values NaN)
# print(df.dropna(how='all'))

# drop rows that have fewer than 4 real values
# print(df.dropna(thresh=4))

# only drop rows where NaN appear in specific columns (here: 'C')
# print(df.dropna(subset=['C']))

# from sklearn.impute import SimpleImputer
# import numpy as np
#
# imr = SimpleImputer(missing_values=np.nan, strategy='mean')
# imr = imr.fit(df.values)
# imputed_data = imr.transform(df.values)
# print(imputed_data)

# print(df.fillna(df.mean()))

import pandas as ps

# df = pd.DataFrame([
#     ['green', 'M', 10.1, 'class2'],
#     ['red', 'L', 13.5, 'class1'],
#     ['blue', 'XL', 15.3, 'class2']
# ])
# df.columns = ['color', 'size', 'price', 'classlabel']
# print(df)

# size_mapping = {'XL':3, 'L':2, 'M':1}
# df['size'] = df['size'].map(size_mapping)
# inv_size_mapping = {v: k for k,v in size_mapping.items()}
# df['size'] = df['size'].map(inv_size_mapping)
# print(df)
# class_mapping = {label: idx for idx, label in enumerate(np.unique(df['classlabel']))}
# print(class_mapping)
# df['classlabel'] = df['classlabel'].map(class_mapping)
# print(df)

# inv_class_mapping = {v:k for k,v in class_mapping.items()}
# df['classlabel'] = df['classlabel'].map(inv_class_mapping)
# print(df)

# from sklearn.preprocessing import LabelEncoder
# class_le = LabelEncoder()
# y = class_le.fit_transform(df['classlabel'].values)
# print(y)
# print(class_le.inverse_transform(y))

# Performing one-hot encoding on nominal features
# X = df[['color', 'size', 'price']].values
# color_le = LabelEncoder()
# X[:, 0] = color_le.fit_transform(X[:, 0])
# print(X)

# from sklearn.preprocessing import OneHotEncoder
# X = df[['color', 'size', 'price']].values
# color_ohe = OneHotEncoder()
# color_ohe.fit_transform(X[:, 0].reshape(-1, 1)).toarray()
# print(color_ohe)

# from sklearn.compose import ColumnTransformer
# x = df[['color', 'size', 'price']].values
# c_transf = ColumnTransformer([
#     ('onehot', OneHotEncoder(), [0]),
#     ('nothing', 'passthrough', [1,2])
# ])
# c_transf.fit_transform(X).astype(float)
# print(pd.get_dummies(df[['price', 'color', 'size']]))
# print(pd.get_dummies(df[['price', 'color', 'size']], drop_first=True))
# color_ohe = OneHotEncoder(categories='auto', drop='first')
# c_transf = ColumnTransformer([
#     ('onehot', color_ohe, [0]),
#     ('nothing', 'passthrough', [1,2])
# ])
# print(c_transf.fit_transform(X).astype(float))

# df = pd.DataFrame([['green', 'M', 10.1, 'class2'],
#                    ['red', 'L', 13.5, 'class1'],
#                    ['blue', 'XL', 15.3, 'class2']])
#
# df.columns = ['color', 'size', 'price', 'classlabel']
# print(df)
# df['x > M'] = df['size'].apply(lambda x: 1 if x in {'L', 'Xl'} else 0)
# df['x > L'] = df['size'].apply(lambda x: 1 if x=='XL' else 0)
# del df['size']
# print(df)


df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',
                 header=None)
df_wine.columns = ['Class label', 'Alcohol',
                    'Malic acid', 'Ash',
                    'Alcalinity of ash', 'Magnesium',
                    'Total phenols', 'Flavanoids',
                    'Nonflavanoid phenols',
                    'Proanthocyanins',
                    'Color intensity', 'Hue',
                    'OD280/OD315 of diluted wines',
                    'Proline']

# print(f"Class label {np.unique(df_wine['Class label'])}")

from IPython.display import display
# df_wine.head()
from sklearn.model_selection import train_test_split
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=0,
                                                    stratify=y)

from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.transform(X_test)

ex = np.array([0, 1, 2, 3, 4, 5])
# print('standardized:', (ex - ex.mean()) / ex.std())
# print('normalized:', (ex - ex.min()) / (ex.max() - ex.min()))

from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)






