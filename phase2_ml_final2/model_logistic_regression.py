from preprocessing import load_and_preprocess
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
import numpy as np

from shared_utils import (
    plot_roc_curve,
    plot_metrics_bar,
    plot_confusion_matrix,
    plot_logistic_coefficients
)

# =========================
# LOAD DATA
# =========================
X_train, X_test, y_train, y_test, skf, scaler, encoders = load_and_preprocess()

# =========================
# PARAM SEARCH
# =========================
C_values = [0.01, 0.1, 1, 10, 100]

best_C = None
best_score = -1


def evaluate_model(C):

    scores = []

    for train_index, val_index in skf.split(X_train, y_train):

        X_tr = X_train[train_index]
        X_val = X_train[val_index]

        y_tr = y_train.iloc[train_index]
        y_val = y_train.iloc[val_index]

        model = LogisticRegression(C=C, max_iter=1000, random_state=42)
        model.fit(X_tr, y_tr)

        pred = model.predict(X_val)
        scores.append(f1_score(y_val, pred))

    return np.mean(scores)


# =========================
# GRID SEARCH
# =========================
print("\n===== GRID SEARCH =====\n")

for C in C_values:
    score = evaluate_model(C)
    print(f"C={C} | CV F1={score:.4f}")

    if score > best_score:
        best_score = score
        best_C = C


# =========================
# RANDOM SEARCH
# =========================
print("\n===== RANDOM SEARCH =====\n")

random_results = []

for i in range(10):
    C = 10 ** np.random.uniform(-2, 2)

    score = evaluate_model(C)

    print(f"Random C={C:.4f} | CV F1={score:.4f}")

    random_results.append((C, score))

best_random = max(random_results, key=lambda x: x[1])


# =========================
# FINAL SELECTION
# =========================
if best_random[1] > best_score:
    best_C = best_random[0]
    method = "Random Search"
else:
    method = "Grid Search"

print("\n===== FINAL SELECTION =====")
print("Method:", method)
print("Best C:", best_C)

# =========================
# FINAL MODEL
# =========================
final_model = LogisticRegression(C=best_C, max_iter=1000, random_state=42)
final_model.fit(X_train, y_train)

y_pred = final_model.predict(X_test)
y_prob = final_model.predict_proba(X_test)[:, 1]

# =========================
# FINAL METRICS
# =========================
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n===== FINAL TEST =====\n")
print("Accuracy: ", round(acc, 4))
print("Precision:", round(prec, 4))
print("Recall:   ", round(rec, 4))
print("F1:       ", round(f1, 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# =========================
# 🔥 VISUALS (FORCED 4 OUTPUTS)
# =========================

# 1) ROC CURVE
plot_roc_curve(
    "Logistic Regression",
    y_test,
    y_prob,
    "logreg_roc.png"
)

# 2) METRICS BAR
plot_metrics_bar(
    "Logistic Regression",
    {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1
    },
    "logreg_metrics.png"
)

# 3) CONFUSION MATRIX
plot_confusion_matrix(
    "Logistic Regression",
    y_test,
    y_pred,
    "logreg_confusion.png"
)

# 4) FEATURE IMPORTANCE (FORCED SAFE VERSION)
try:
    feature_names = (
        X_train.columns
        if hasattr(X_train, "columns")
        else [f"f{i}" for i in range(X_train.shape[1])]
    )

    plot_logistic_coefficients(
        final_model,
        feature_names,
        "logreg_features.png"
    )

except Exception as e:
    print("Feature importance failed:", e)