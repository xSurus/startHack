from tkinter import N
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

def one_hot(df: pd.DataFrame, catcol_name: str, cat_val: int):
    return np.array(df[catcol_name] == cat_val, dtype=int)

from dfcols import all_square_cols
def most_frequent(df: pd.DataFrame, col_name: str):
    matrix = df[all_square_cols(col_name)].to_numpy()
    return np.array(list(map(np.argmax, map(np.bincount, matrix))))

def mean(df: pd.DataFrame, col_name: str):
    matrix = df[all_square_cols(col_name)].to_numpy()
    return matrix.mean(axis=1)

def std(df: pd.DataFrame, col_name: str):
    matrix = df[all_square_cols(col_name)].to_numpy()
    return matrix.std(axis=1)

from sklearn.preprocessing import normalize as sknormalize
def normalize(df: pd.DataFrame, col_name: str):
    return sknormalize(df[[col_name]].to_numpy(), axis=0)

from sklearn.svm import SVC
from sklearn.model_selection import KFold
def test_feats_svc(X, y, k=3):
    svc = SVC(random_state=42)
    kf = KFold(n_splits=k)
    f1mean = 0
    i = 0
    for train_ind, test_ind in kf.split(X):
        X_train, X_test = X[train_ind], X[test_ind]
        y_train, y_test = y[train_ind], y[test_ind]

        svc.fit(X_train, y_train)
        y_pred = svc.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        f1mean += f1

        i+= 1

    f1mean /= k
    return f1mean

