# Import libraries
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
pd.set_option('display.max_columns', None)
import warnings
warnings.filterwarnings('ignore')

# Read files to pandas dataframes
train = pd.read_csv('../data/Train.csv')
test = pd.read_csv('../data/Test.csv')
sample_submission = pd.read_csv('../data/SampleSubmission.csv')

main_cols = train.columns.difference(['Sample_ID', 'Label'])
X = train[main_cols]
y = train.Label

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=2022)
estimators = [
    ('mlp', MLPClassifier(alpha=0.5)),
    ('rf', RandomForestClassifier(n_estimators=500, random_state=42)),
    ('svr', make_pipeline(StandardScaler(), SVC(gamma='auto', random_state=42)))
]
# Train model
model = StackingClassifier(
    estimators=estimators, final_estimator=LogisticRegression()
)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Check the auc score of the model
print(f'RandomForest F1 score on the X_test is: {f1_score(y_test, y_pred)}\n')
