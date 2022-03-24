import numpy as np
import pandas as pd

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