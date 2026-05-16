


import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_and_preprocess():
    # 1. Load the pre-split CSVs
    X_train_df = pd.read_csv("phase2_ml_final2/X_train.csv")
    X_test_df  = pd.read_csv("phase2_ml_final2/X_test.csv")
    y_train    = pd.read_csv("phase2_ml_final2/y_train.csv").squeeze()   # Series
    y_test     = pd.read_csv("phase2_ml_final2/y_test.csv").squeeze()    # Series

    print(f"X_train: {X_train_df.shape}, X_test: {X_test_df.shape}")

    # 2. Convert to numpy arrays 
    # The KNN CV loop uses integer indexing (X_train[train_index]),
    # which requires a numpy array, not a DataFrame.
    X_train = X_train_df.to_numpy()
    X_test  = X_test_df.to_numpy()

    # 3. Re-fit scaler on X_train (for reference / future inverse-transform)
    NUMERICAL_IDX = [
        list(X_train_df.columns).index(c)
        for c in ["prev-attempts", "studied-credits", "avg-score"]
    ]
    scaler = StandardScaler()
    scaler.fit(X_train[:, NUMERICAL_IDX])   # already scaled in CSV; fit for ref

    # 4. Re-fit label encoders (for reference / inverse-transform)
    df_raw = pd.read_csv("phase2_ml_final2/WithdrawlStudents.csv")
    df_raw["imd-band"] = df_raw["imd-band"].fillna("Unknown")

    CATEGORICAL_COLS = [
        "gender", "age-band", "imd-band",
        "high-education", "region", "disability",
    ]
    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        le.fit(df_raw[col].astype(str))
        encoders[col] = le

    # 5. Stratified K-Fold splitter 
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    return X_train, X_test, y_train, y_test, skf, scaler, encoders