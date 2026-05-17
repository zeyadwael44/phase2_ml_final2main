<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
=======
>>>>>>> 2e5f513 (resolve)
from preprocessing import load_and_preprocess
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
import numpy as np
import random

# IMPORTANT: avoid import issues
import shared_utils as su


# =========================
# LOAD DATA
# =========================
X_train, X_test, y_train, y_test, skf, scaler, encoders = load_and_preprocess()

feature_names = (
    X_train.columns if hasattr(X_train, "columns")
    else np.arange(X_train.shape[1])
)

# =========================
# GRID SEARCH
# =========================
print("\n===== GRID SEARCH =====\n")

n_estimators_values = [50, 100, 200]
max_depth_values = [None, 10, 20]
min_samples_split_values = [2, 5]

best_grid_score = 0
best_grid_params = None

for n_estimators in n_estimators_values:
    for max_depth in max_depth_values:
        for min_samples_split in min_samples_split_values:

            scores = []

            for train_idx, val_idx in skf.split(X_train, y_train):

                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1
                )

                model.fit(X_tr, y_tr)
                preds = model.predict(X_val)
                scores.append(f1_score(y_val, preds))

            avg = np.mean(scores)

            print(f"Grid | n_estimators={n_estimators}, max_depth={max_depth}, min_samples_split={min_samples_split} | F1={avg:.4f}")

            if avg > best_grid_score:
                best_grid_score = avg
                best_grid_params = (n_estimators, max_depth, min_samples_split)

best_grid_n, best_grid_d, best_grid_s = best_grid_params

print("\nBEST GRID:", best_grid_params, "F1:", best_grid_score)


# =========================
# RANDOM SEARCH
# =========================
print("\n===== RANDOM SEARCH =====\n")

best_random_score = 0
best_random_params = None

for i in range(10):

    n_estimators = random.choice([50, 100, 150, 200, 300])
    max_depth = random.choice([None, 5, 10, 15, 20, 30])
    min_samples_split = random.randint(2, 10)

    scores = []

    for train_idx, val_idx in skf.split(X_train, y_train):

        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        scores.append(f1_score(y_val, preds))

    avg = np.mean(scores)

    print(f"Random | n_estimators={n_estimators}, max_depth={max_depth}, min_samples_split={min_samples_split} | F1={avg:.4f}")

    if avg > best_random_score:
        best_random_score = avg
        best_random_params = (n_estimators, max_depth, min_samples_split)

best_r_n, best_r_d, best_r_s = best_random_params

print("\nBEST RANDOM:", best_random_params, "F1:", best_random_score)


# =========================
# FINAL MODEL - RANDOM
# =========================
print("\n===== FINAL RANDOM MODEL =====\n")

random_model = RandomForestClassifier(
    n_estimators=best_r_n,
    max_depth=best_r_d,
    min_samples_split=best_r_s,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

random_model.fit(X_train, y_train)
random_pred = random_model.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, random_pred), 4))
print("Precision:", round(precision_score(y_test, random_pred, zero_division=0), 4))
print("Recall:", round(recall_score(y_test, random_pred, zero_division=0), 4))
print("F1:", round(f1_score(y_test, random_pred, zero_division=0), 4))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, random_pred))


# =========================
# RANDOM VISUALS
# =========================
su.plot_confusion_matrix(
    "RandomForest (Random)",
    y_test,
    random_pred,
    "rf_random_confusion.png"
)

su.plot_rf_feature_importance(
    random_model,
    feature_names,
    "rf_random_feature_importance.png"
)


# =========================
# FINAL MODEL - GRID
# =========================
print("\n===== FINAL GRID MODEL =====\n")

grid_model = RandomForestClassifier(
    n_estimators=best_grid_n,
    max_depth=best_grid_d,
    min_samples_split=best_grid_s,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

grid_model.fit(X_train, y_train)
grid_pred = grid_model.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, grid_pred), 4))
print("Precision:", round(precision_score(y_test, grid_pred, zero_division=0), 4))
print("Recall:", round(recall_score(y_test, grid_pred, zero_division=0), 4))
print("F1:", round(f1_score(y_test, grid_pred, zero_division=0), 4))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, grid_pred))


# =========================
# GRID VISUALS
# =========================
su.plot_confusion_matrix(
    "RandomForest (Grid)",
    y_test,
    grid_pred,
    "rf_grid_confusion.png"
)

su.plot_rf_feature_importance(
    grid_model,
    feature_names,
    "rf_grid_feature_importance.png"
)