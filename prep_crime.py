
'''
preprocessing script is modified from 
https://github.com/vbordalo/Communities-Crime/blob/master/Crime_v1.ipynb
'''

import numpy as np
from sklearn.impute import SimpleImputer
from pandas import read_csv
from sklearn.model_selection import train_test_split

# read data from files
attrib = read_csv('data/attributes.csv', delim_whitespace=True)
df_data = read_csv('data/communities.data', names=attrib['attributes'])

print("Data format: ")
print(df_data.head())

# Remove non-predictive features: state, county, community, communityname, fold
df_data = df_data.drop(columns=['state','county',
                            'community','communityname',
                            'fold'], axis=1)
print("\nAfter removing non-predictive features, Data shape: ", df_data.shape)

# remove and fill out missing values
df_data = df_data.replace('?', np.nan)
feat_missing = df_data.columns[df_data.isnull().any()].tolist()

print("feature missing:", feat_missing)
print("%d features contains missing values "%len(feat_missing))

# OtherPerCap has only one missing value and will be filled 
# by a mean value using Imputer from sklearn.preprocessing. 
# The others features present many missing values and will be removed from the data set.
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(df_data[['OtherPerCap']])
df_data[['OtherPerCap']] = imputer.transform(df_data[['OtherPerCap']])

df_data = df_data.dropna(axis=1)
print("\nAfter removing/preproecssing missing value, data shape:", df_data.shape)

# define senstive attribute: rule 2 white>0.5
df_A = (df_data['racePctWhite']>= 0.80)

# drop sensitive-related attributes in X
df_data = df_data.drop(columns=['racepctblack',
                                'racePctWhite',
                                'racePctAsian',
                                'racePctHisp'], axis=1)

# get X, Y and A
print(df_data.head())
X = df_data.drop(columns=['ViolentCrimesPerPop'], axis=1).values # X = df_data.iloc[:,:-1].values
Y = df_data['ViolentCrimesPerPop'].values # Y = df_data.iloc[:, -1].values
A = df_A.values.astype(int)
# print(X[:5,-1], Y[:5])
print("X.shape, Y.shape, A.shape", X.shape, Y.shape, A.shape)
print("X.dtype, Y.dtype, A.dtype", X.dtype, Y.dtype, A.dtype)

# train/test split: 0.8/0.2 
X_train, X_test, Y_train, Y_test, A_train, A_test  = train_test_split(X, Y, A, 
                                                        test_size=0.2, 
                                                        random_state=0, 
                                                        stratify=A)


# print(X_train.shape, X_test.shape)
# print(Y_train.shape, Y_test.shape)
print(A_train.shape, A_test.shape)
print(A_train.sum(), A_test.sum())

f_out_np = 'data/communities_crime.npz'
np.savez(f_out_np, x_train=X_train, x_test=X_test,
            y_train=Y_train, y_test=Y_test,
            attr_train=A_train, attr_test=A_test)