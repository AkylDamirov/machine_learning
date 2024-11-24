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

df = pd.DataFrame([
    ['green', 'M', 10.1, 'class2'],
    ['red', 'L', 13.5, 'class1'],
    ['blue', 'XL', 15.3, 'class2']
])
df.columns = ['color', 'size', 'price', 'classlabel']
# print(df)

size_mapping = {'XL':3, 'L':2, 'M':1}
df['size'] = df['size'].map(size_mapping)
# inv_size_mapping = {v: k for k,v in size_mapping.items()}
# df['size'] = df['size'].map(inv_size_mapping)
# print(df)
class_mapping = {label: idx for idx, label in enumerate(np.unique(df['classlabel']))}
# print(class_mapping)
df['classlabel'] = df['classlabel'].map(class_mapping)
# print(df)

inv_class_mapping = {v:k for k,v in class_mapping.items()}
df['classlabel'] = df['classlabel'].map(inv_class_mapping)
# print(df)

from sklearn.preprocessing import LabelEncoder
class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel'].values)
# print(y)
# print(class_le.inverse_transform(y))











