# data_preprocessing.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_data():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data'
    data = pd.read_csv(url, header=None)
    return data

def preprocess_data(data):
    # Assign column names
    data.columns = ['Feature_' + str(i) for i in range(1, 61)] + ['Class']

    # Convert target variable to binary (M:1, R:0)
    data['Class'] = data['Class'].map({'M': 1, 'R': 0})

    # Apply log transformation
    X = np.log(data.iloc[:, :-1] + 1)
    y = data['Class']
    return X, y

def get_train_test_split(X, y, train_size=158, random_state=None):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test
