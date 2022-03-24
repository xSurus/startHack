import pandas as pd
import numpy as np
import dfcols
continuous = ['sdoif', 'elevation', 'procurv', 'placurv', 'lsfactor', 'slope', 'twi', 'aspect']
categorical = ['geology']

def originalXy(as_numpy=False):
    train = pd.read_csv('data/Train.csv')
    original_columns = list(train.columns)[1:-1]
    X = train[original_columns]
    y = train["Label"].to_numpy()

    if as_numpy:
        X = X.to_numpy()
    return X, y

from sklearn.preprocessing import normalize

def baselineXy(as_numpy=False):
    X, y = originalXy()
    original_columns = list(X.columns)
    def most_freq(df, col):
        matrix = df[dfcols.all_square_cols(col)].to_numpy()
        return np.array(list(map(np.argmax, map(np.bincount, matrix))))
    
    for cat in categorical:
        # get most freq category
        X[cat] = most_freq(X, cat)
        # onehots
        for cat_val in X[cat].unique():
            onehot_name = f"{cat}_{cat_val}"
            X[onehot_name] = np.array(X[cat] == cat_val, dtype=int)

        X.drop([cat], inplace=True, axis=1)

    def mean(df: pd.DataFrame, col_name: str):
        matrix = df[dfcols.all_square_cols(col_name)].to_numpy()
        return matrix.mean(axis=1)

    fns = [(np.array, 'id'), (np.sqrt, 'sqrt'), (np.square, 'sq')]

    mean_transf = {
        'slope': [0],
        'elevation': [1],
        'lsfactor': [0],
        'placurv': [0],
        'twi': [0, 1],
        'aspect': [0],
        'sdoif': [0, 2],
        'procurv': [0],
    }

    for col in continuous:
        m = mean(X, col)
        m -= m.min()
        for fn, name in [fns[i] for i in mean_transf[col]]:
            X[f"{col}_mean_{name}"] = fn(m)

    X.drop(original_columns, inplace=True, axis=1)

    if as_numpy:
        X = X.to_numpy()

    return X, y