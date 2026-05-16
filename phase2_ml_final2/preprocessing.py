import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_and_preprocess():

    # ── Load dataset splits ─────────────────────────────────────────────
    X_train_df = pd.read_csv("phase2_ml_final2/X_train.csv")
    X_test_df  = pd.read_csv("phase2_ml_final2/X_test.csv")
    y_train    = pd.read_csv("phase2_ml_final2/y_train.csv").squeeze()
    y_test     = pd.read_csv("phase2_ml_final2/y_test.csv").squeeze()

    print(f"X_train: {X_train_df.shape}, X_test: {X_test_df.shape}")

    # ── Convert to numpy ────────────────────────────────────────────────
    X_train = X_train_df.to_numpy()
    X_test  = X_test_df.to_numpy()

    # ── Fit scaler (IMPORTANT: do NOT assume already scaled) ────────────
    numerical_cols = ["prev-attempts", "studied-credits", "avg-score"]

    numerical_idx = [X_train_df.columns.get_loc(c) for c in numerical_cols]

    scaler = StandardScaler()
    X_train[:, numerical_idx] = scaler.fit_transform(X_train[:, numerical_idx])
    X_test[:, numerical_idx]  = scaler.transform(X_test[:, numerical_idx])

    # ── Label encoders (optional reference only) ────────────────────────
    df_raw = pd.read_csv("phase2_ml_final2/WithdrawlStudents.csv")
    df_raw["imd-band"] = df_raw["imd-band"].fillna("Unknown")

    categorical_cols = [
        "gender", "age-band", "imd-band",
        "high-education", "region", "disability",
    ]

    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        le.fit(df_raw[col].astype(str))
        encoders[col] = le

    # ── CV splitter ─────────────────────────────────────────────────────
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    return X_train, X_test, y_train, y_test, skf, scaler, encoders