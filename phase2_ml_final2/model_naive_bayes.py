from preprocessing import load_and_preprocess
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np


# ── Load data ─────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test, skf, scaler, encoders = load_and_preprocess()

# ── Hyperparameters ───────────────────────────────────────────────────
smoothing_values = [1e-12, 1e-11, 1e-10, 1e-9, 1e-8]

best_smoothing = None
best_score = -1

print("\n===== CROSS VALIDATION =====\n")

# ── Cross-validation ──────────────────────────────────────────────────
for smoothing in smoothing_values:
    fold_scores = []

    for train_idx, val_idx in skf.split(X_train, y_train):

        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model = GaussianNB(var_smoothing=smoothing)
        model.fit(X_tr, y_tr)

        preds = model.predict(X_val)
        fold_scores.append(f1_score(y_val, preds))

    avg_score = np.mean(fold_scores)

    print(f"var_smoothing={smoothing} | F1={avg_score:.4f}")

    if avg_score > best_score:
        best_score = avg_score
        best_smoothing = smoothing


print("\nBEST var_smoothing:", best_smoothing)
print("BEST CV F1:", round(best_score, 4))

# ── Final model ───────────────────────────────────────────────────────
print("\n===== FINAL TEST =====\n")

final_model = GaussianNB(var_smoothing=best_smoothing)
final_model.fit(X_train, y_train)

y_pred = final_model.predict(X_test)

print("Accuracy: ", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred), 4))
print("Recall:   ", round(recall_score(y_test, y_pred), 4))
print("F1:       ", round(f1_score(y_test, y_pred), 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))