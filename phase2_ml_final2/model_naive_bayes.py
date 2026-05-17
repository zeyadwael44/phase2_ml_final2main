from preprocessing import load_and_preprocess
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score
)
import numpy as np

# =========================
# LOAD DATA
# =========================
X_train, X_test, y_train, y_test, skf, scaler, encoders = load_and_preprocess()

# =========================
# GRID SEARCH
# =========================
smoothing_values = [1e-12, 1e-11, 1e-10, 1e-9, 1e-8]

best_smoothing = None
best_score = -1


def evaluate_model(smoothing):

    fold_scores = []

    for train_idx, val_idx in skf.split(X_train, y_train):

        X_tr = X_train[train_idx]
        X_val = X_train[val_idx]

        y_tr = y_train.iloc[train_idx]
        y_val = y_train.iloc[val_idx]

        model = GaussianNB(var_smoothing=smoothing)
        model.fit(X_tr, y_tr)

        preds = model.predict(X_val)
        fold_scores.append(f1_score(y_val, preds))

    return np.mean(fold_scores)


print("\n===== GRID SEARCH =====\n")

for s in smoothing_values:
    score = evaluate_model(s)

    print(f"Grid | var_smoothing={s} | CV F1={score:.4f}")

    if score > best_score:
        best_score = score
        best_smoothing = s


print("\nBest Grid Search Parameter:")
print("var_smoothing =", best_smoothing)
print("Best CV F1 =", best_score)


# =========================
# RANDOM SEARCH
# =========================
print("\n===== RANDOM SEARCH =====\n")

random_scores = []

for i in range(10):

    s = 10 ** np.random.uniform(-12, -7)

    score = evaluate_model(s)

    print(f"Random | var_smoothing={s:.12f} | CV F1={score:.4f}")

    random_scores.append((s, score))


best_random = max(random_scores, key=lambda x: x[1])

print("\nBest Random Search Parameter:")
print("var_smoothing =", best_random[0])
print("Best CV F1 =", best_random[1])


# =========================
# SELECT BEST MODEL
# =========================
if best_random[1] > best_score:
    best_smoothing = best_random[0]
    best_method = "Random Search"
else:
    best_method = "Grid Search"


print("\n===== FINAL BEST PARAMETER =====")
print("Selected Method:", best_method)
print("Best var_smoothing:", best_smoothing)


# =========================
# FINAL MODEL
# =========================
final_model = GaussianNB(var_smoothing=best_smoothing)
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
auc = roc_auc_score(y_test, y_prob)

print("\n===== FINAL TEST RESULTS =====\n")

print("Accuracy: ", round(acc, 4))
print("Precision:", round(prec, 4))
print("Recall:   ", round(rec, 4))
print("F1:       ", round(f1, 4))
print("ROC-AUC:  ", round(auc, 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# =========================
# VISUALIZATIONS (FIXED & CLEAN)
# =========================
from shared_utils import (
    plot_roc_curve,
    plot_metrics_bar,
    plot_feature_distribution,
    plot_confusion_matrix
)

# ROC Curve
plot_roc_curve("GaussianNB", y_test, y_prob, "nb_roc_curve.png")

# Metrics Bar
plot_metrics_bar(
    "GaussianNB",
    {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "ROC-AUC": auc
    },
    "nb_metrics.png"
)

# Feature distribution (NO .values because you use numpy)
plot_feature_distribution(
    X_train,
    y_train,
    feature_index=0,
    filename="nb_feature_0.png"
)

# Confusion Matrix
plot_confusion_matrix(
    "GaussianNB",
    y_test,
    y_pred,
    "nb_confusion_matrix.png"
)