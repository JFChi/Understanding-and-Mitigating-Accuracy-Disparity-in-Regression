import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

'''
The bar passage study was initiated in 1991 by Law School Admission
Council national longitudinal. The dataset contains records for law students who took the bar exam.
The binary outcome indicates whether the student passed the bar exam or not. The features include
variables such as cluster, lsat score, undergraduate GPA, zfyGPA, zGPA, full-time status, family
income, age and also sensitive variables such as race and gender. The variable cluster is the result of a
clustering of similar law schools (which is done apriori), and is used to adjust for the effect of type of
law school. zGPA is the z-scores of the students overall GPA and zfyGPA is the first year GPA relative
to students at the same law school. 
'''

def clean_dataset(dataset, attributes, centered):
    df = pd.read_csv(dataset)
    sens_df = pd.read_csv(attributes)

    ## Get and remove label Y
    y_col = [str(c) for c in sens_df.columns if sens_df[c][0] == 2]
    print('label feature: {}'.format(y_col))
    if(len(y_col) > 1):
        raise ValueError('More than 1 label column used')
    if (len(y_col) < 1):
        raise ValueError('No label column used')

    y = df[y_col[0]]

    ## Do not use labels in rest of data
    X = df.loc[:, df.columns != y_col[0]]
    X = X.loc[:, X.columns != 'Unnamed: 0']
    ## Create X_prime, by getting protected attributes
    sens_cols = [str(c) for c in sens_df.columns if sens_df[c][0] == 1]
    print('sensitive features: {}'.format(sens_cols))
    sens_dict = {c: 1 if c in sens_cols else 0 for c in df.columns}
    X, sens_dict = one_hot_code(X, sens_dict)
    sens_names = [key for key in sens_dict.keys() if sens_dict[key] == 1]
    print('there are {} sensitive features including derivative features'.format(len(sens_names)))

    X_prime = X[sens_names]

    if(centered):
        X = center(X)

    # normalize y to [0, 1]
    y = ( y - np.min(y) ) / (np.max(y) - np.min(y) )

    return X, X_prime, y

def center(X):
    for col in X.columns:
        X.loc[:, col] = ( X.loc[:, col]-np.mean(X.loc[:, col]) ) / np.std(X.loc[:, col])
        # X.loc[:, col] =  X.loc[:, col]-np.mean(X.loc[:, col])
    return X

def one_hot_code(df1, sens_dict):
    cols = df1.columns
    for c in cols:
        if isinstance(df1[c][0], str):
            column = df1[c]
            df1 = df1.drop(c, 1)
            unique_values = list(set(column))
            n = len(unique_values)
            if n > 2:
                for i in range(n):
                    col_name = '{}.{}'.format(c, i)
                    col_i = [1 if el == unique_values[i] else 0 for el in column]
                    df1[col_name] = col_i
                    sens_dict[col_name] = sens_dict[c]
                del sens_dict[c]
            else:
                col_name = c
                col = [1 if el == unique_values[0] else 0 for el in column]
                df1[col_name] = col
    return df1, sens_dict

if __name__ == "__main__":
    # load data
    data_path = "data/lawschool.csv"
    centered = True
    lawschool_attributes = "data/lawschool_protected.csv"

    # save data summary to csv
    law_school_df = pd.read_csv(data_path)
    law_school_df.describe().to_csv("data/lawschool_summary.csv")


    df_X, df_A, df_Y = clean_dataset(data_path, lawschool_attributes, centered)
    
    # to numpy
    X = df_X.values
    Y = df_Y.values
    A = df_A.values.astype(int).squeeze()

    print("X.shape, Y.shape, A.shape", X.shape, Y.shape, A.shape)
    print(np.min(X), np.max(X))
    print(np.min(Y), np.max(Y))
    print(np.min(A), np.max(A))

    # train/test split: 0.8/0.2 
    X_train, X_test, Y_train, Y_test, A_train, A_test  = train_test_split(X, Y, A, 
                                                            test_size=0.2, 
                                                            random_state=0, 
                                                            stratify=A)

    print(A_train.shape, A_test.shape)
    print(A_train.sum(), A_test.sum())


    f_out_np = 'data/law_school.npz'
    np.savez(f_out_np, x_train=X_train, x_test=X_test,
                y_train=Y_train, y_test=Y_test,
                attr_train=A_train, attr_test=A_test)
