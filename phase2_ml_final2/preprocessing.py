import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


def load_and_preprocess():

    X_train_df = pd.read_csv("X_train.csv")
    X_test_df  = pd.read_csv("X_test.csv")

    y_train = pd.read_csv("y_train.csv").squeeze()
    y_test  = pd.read_csv("y_test.csv").squeeze()

    categorical_cols = ["gender", "age-band", "imd-band", "high-education", "region", "disability"]
    numerical_cols   = ["prev-attempts", "studied-credits", "avg-score"]

    # One-hot encode categorical columns
    X_train_df = pd.get_dummies(X_train_df, columns=categorical_cols)
    X_test_df  = pd.get_dummies(X_test_df,  columns=categorical_cols)

    # Align columns in case some categories only appear in one split
    X_train_df, X_test_df = X_train_df.align(X_test_df, join="left", axis=1, fill_value=0)

    # Scale numerical columns
    scaler = StandardScaler()
    X_train_df[numerical_cols] = scaler.fit_transform(X_train_df[numerical_cols])
    X_test_df[numerical_cols]  = scaler.transform(X_test_df[numerical_cols])

    X_train = X_train_df.to_numpy()
    X_test  = X_test_df.to_numpy()

    # Balance the training set to fix class imbalance
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    return X_train, X_test, y_train, y_test, skf, scaler, None