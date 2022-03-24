import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

pd.set_option('display.max_columns', None)
df = pd.read_csv('../Data/Train.csv')
df_train = df.drop(labels=["Sample_ID","Label"], axis=1)[df.columns.drop(list(df.filter(regex='geology')))]
y_train = df[["Label"]]
df_train_t8 = df_train.head(800)
y_train_t8 = y_train.head(800)
Xtr = df_train_t8.to_numpy()
ytr = y_train_t8.to_numpy()

df_train_l2 = df_train.tail(200)
y_train_l2 = df_train.tail(200)
Xte = df_train_l2.to_numpy()
yte = y_train_l2.to_numpy()

#clf_LR = LogisticRegression().fit(Xtr,ytr)
#y_pred = clf_LR.predict(Xte)
#f1_LR = f1_score(y_pred, yte)
#print(f1_LR)
print(Xtr.shape())