"""
model_lr.py
-----------
Train and evaluate a Logistic Regression classifier.
CV logic:
  - Tries C in [0.01, 0.1, 1, 10, 100]
  - Selects the best C by mean F1 over a Stratified 5-Fold CV on the training set
  - Re-trains on the full training set with the best C
  - Evaluates on the held-out test set
Depends on preprocessing.py (and the CSV files it reads) being in the same directory.
"""
from preprocessing import load_and_preprocess
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix,
)
import numpy as np

# ── Load data ─────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test, skf, scaler, encoders = load_and_preprocess()

C_values   = [0.01, 0.1, 1, 10, 100]
best_C     = 0
best_score = 0

# ── Cross-validation ──────────────────────────────────────────────────────────
print("\n===== CROSS VALIDATION =====\n")

for C in C_values:
    scores = []

    for train_index, val_index in skf.split(X_train, y_train):
        # X_train is a numpy array → integer indexing works directly
        X_tr  = X_train[train_index]
        X_val = X_train[val_index]

        # y_train is a pandas Series → use .iloc for positional indexing
        y_tr  = y_train.iloc[train_index]
        y_val = y_train.iloc[val_index]

        lr = LogisticRegression(C=C, max_iter=1000, random_state=42)
        lr.fit(X_tr, y_tr)

        pred = lr.predict(X_val)
        scores.append(f1_score(y_val, pred))

    avg = np.mean(scores)
    print(f"C = {C} | F1 = {avg:.4f}")

    if avg > best_score:
        best_score = avg
        best_C     = C

print("\nBEST C:", best_C)
print("BEST F1 (CV):", round(best_score, 4))

# ── Final evaluation on test set ──────────────────────────────────────────────
print("\n===== FINAL TEST =====\n")

lr = LogisticRegression(C=best_C, max_iter=1000, random_state=42)
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

print("Accuracy: ", round(accuracy_score(y_test, y_pred),  4))
print("Precision:", round(precision_score(y_test, y_pred), 4))
print("Recall:   ", round(recall_score(y_test, y_pred),    4))
print("F1:       ", round(f1_score(y_test, y_pred),        4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))