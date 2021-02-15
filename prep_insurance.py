'''
Medical Cost Personal Datasets Preprocessing scripts
'''

import numpy as np 
import pandas as pd
import math

from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/insurance.csv')

# print data summary
print("Before pre-processing: ")
print(df.describe())

# feature engineering
# 1. region feature
onehotencoder = OneHotEncoder(categories='auto')
var1 = onehotencoder.fit_transform(df.region.values.reshape(-1,1)).toarray()
var1 = pd.DataFrame(var1)
var1.columns = ['region_1', 'region_2', 'region_3', 'region_4']
var1 = var1.iloc[:,0:3]
df = pd.concat([df, var1], axis=1)
# 2. smoker
onehotencoder = OneHotEncoder(categories='auto')
var2 = onehotencoder.fit_transform(df.smoker.values.reshape(-1,1)).toarray()
var2 = pd.DataFrame(var2)
var2.columns = ['smoker_1', 'smoker_2']
var2 = var2.iloc[:,0]
df = pd.concat([df, var2], axis=1)
# 3. sex
onehotencoder = OneHotEncoder(categories='auto')
var3 = onehotencoder.fit_transform(df.sex.values.reshape(-1,1)).toarray()
var3 = pd.DataFrame(var3)
var3.columns = ['sex_1', 'sex_2']
var3 = var3.iloc[:, 0]
df = pd.concat([df, var3], axis=1)

# # standarize numerical feature other than the target label
num_vars = df[["age", "children", "bmi"]].values
scaler = StandardScaler().fit(num_vars)
scaled_num_vars = scaler.transform(num_vars)
df[["age", "children", "bmi"]] = scaled_num_vars


# normalize the target variable -charges to [0, 1]
tar_var = df[["charges"]].values
min_max_scaler = MinMaxScaler(feature_range=(0, 1))
scaled_tar_var = min_max_scaler.fit_transform(tar_var)
df[["charges"]] = scaled_tar_var

df = df.drop(columns = ['region', 'sex', 'smoker'])
# print data after preprocessing
df = df[['age', "sex_1", 'bmi', 'children', 'region_1', 'region_2', 'region_3','smoker_1', 'charges']]
print("*"*50)
print("After preprocessing")
print(df.describe())

# create dataset 
X = df[['age', 'bmi', 'children', 'region_1', 'region_2', 'region_3', 'smoker_1']].values
Y = df['charges'].values
A = df["sex_1"].values.astype(int)

print("df.shape ", df.shape)
print("np.sum(A)", np.sum(A))
print("Before subsampling, X.shape, Y.shape, A.shape", X.shape, Y.shape, A.shape)


# subsample dataset
np.random.seed(3)
# find the index of A 
idx_0 = np.where(A == 0)[0]
idx_1 = np.where(A == 1)[0]

X_0, X_1 = X[idx_0], X[idx_1]
Y_0, Y_1 = Y[idx_0], Y[idx_1]
A_0, A_1 = A[idx_0], A[idx_1]

# subsampling for group 0
samp_prop = 0.05
selected_idx = np.random.choice(len(A_0), size=int(len(A_0)*samp_prop), replace=False)
X_0, Y_0, A_0 = X_0[selected_idx], Y_0[selected_idx], A_0[selected_idx]

# subsampling for group 1
samp_prop = 0.5
selected_idx = np.random.choice(len(A_1), size=int(len(A_1)*samp_prop), replace=False)
X_1, Y_1, A_1 = X_1[selected_idx], Y_1[selected_idx], A_1[selected_idx]

# concatenate the data back and suffle
X = np.concatenate((X_0, X_1), axis=0)
Y = np.concatenate((Y_0, Y_1), axis=0)
A = np.concatenate((A_0, A_1), axis=0)

print("# of example in group 0: ", len(A_0))
print("# of example in group 1: ", len(A_1))

print("After subsampling, X.shape, Y.shape, A.shape", X.shape, Y.shape, A.shape)


# train-test split
X_train, X_test, Y_train, Y_test, A_train, A_test = train_test_split(X, Y, A, 
                                                                test_size=0.3, 
                                                                random_state=0, 
                                                                stratify=A)

# linear regression 
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, Y_train)
lr_pred = reg.predict(X_test)
print("MSE: ", mean_squared_error(y_pred=lr_pred, y_true=Y_test))
print("RMSE: ", mean_squared_error(y_pred=lr_pred, y_true=Y_test, squared=False))
print("R^2: ", r2_score(y_true=Y_test, y_pred=lr_pred))    

from fairlearn.metrics import group_summary
print("Under Bounded Group Loss constraint, MSE summary: {}".format(group_summary(mean_squared_error, Y_test, lr_pred, sensitive_features=A_test)))
results = group_summary(mean_squared_error, Y_test, lr_pred, sensitive_features=A_test)
cls_error = results['overall']
error_0 = results['by_group'][0]
error_1 = results['by_group'][1]
print("err_gap: ",  np.abs(error_0-error_1))
             
# save to file
f_out_np = 'data/insurance.npz'
np.savez(f_out_np, x_train=X_train, x_test=X_test,
            y_train=Y_train, y_test=Y_test,
            attr_train=A_train, attr_test=A_test)