from preprocessing import load_and_preprocess
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix,
)
import numpy as np
import random

# Load data
X_train, X_test, y_train, y_test, skf, scaler, encoders = load_and_preprocess()

# GRID SEARCH
n_estimators_values = [50, 100, 200]
max_depth_values    = [None, 10, 20]
min_samples_split_values = [2, 5]

best_grid_n_estimators    = None
best_grid_max_depth       = None
best_grid_min_samples_split = None
best_grid_score           = 0

print("\n===== GRID SEARCH =====\n")

for n_estimators in n_estimators_values:
    for max_depth in max_depth_values:
        for min_samples_split in min_samples_split_values:

            scores = []

            for train_index, val_index in skf.split(X_train, y_train):

                X_tr  = X_train[train_index]
                X_val = X_train[val_index]
                y_tr  = y_train.iloc[train_index]
                y_val = y_train.iloc[val_index]

                rf = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                )
                rf.fit(X_tr, y_tr)
                pred = rf.predict(X_val)
                scores.append(f1_score(y_val, pred))

            avg = np.mean(scores)
            print(f"n_estimators={n_estimators}, max_depth={max_depth}, min_samples_split={min_samples_split} | F1={avg:.4f}")

            if avg > best_grid_score:
                best_grid_score             = avg
                best_grid_n_estimators      = n_estimators
                best_grid_max_depth         = max_depth
                best_grid_min_samples_split = min_samples_split

print("\nBEST GRID RESULT")
print("n_estimators =", best_grid_n_estimators)
print("max_depth =", best_grid_max_depth)
print("min_samples_split =", best_grid_min_samples_split)
print("F1 =", round(best_grid_score, 4))

# RANDOM SEARCH
print("\n===== RANDOM SEARCH =====\n")

best_random_n_estimators      = None
best_random_max_depth         = None
best_random_min_samples_split = None
best_random_score             = 0

for i in range(10):
    n_estimators      = random.choice([50, 100, 150, 200, 300])
    max_depth         = random.choice([None, 5, 10, 15, 20, 30])
    min_samples_split = random.randint(2, 10)

    scores = []

    for train_index, val_index in skf.split(X_train, y_train):

        X_tr  = X_train[train_index]
        X_val = X_train[val_index]
        y_tr  = y_train.iloc[train_index]
        y_val = y_train.iloc[val_index]

        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X_tr, y_tr)
        pred = rf.predict(X_val)
        scores.append(f1_score(y_val, pred))

    avg = np.mean(scores)
    print(f"Random n_estimators={n_estimators}, max_depth={max_depth}, min_samples_split={min_samples_split} | F1={avg:.4f}")

    if avg > best_random_score:
        best_random_score             = avg
        best_random_n_estimators      = n_estimators
        best_random_max_depth         = max_depth
        best_random_min_samples_split = min_samples_split

print("\nBEST RANDOM RESULT")
print("n_estimators =", best_random_n_estimators)
print("max_depth =", best_random_max_depth)
print("min_samples_split =", best_random_min_samples_split)
print("F1 =", round(best_random_score, 4))

# FINAL RESULTS SECTION
print("\n============================================================\n")

# BEST RANDOM MODEL
print("===== BEST RANDOM MODEL (FINAL TEST) =====\n")

random_model = RandomForestClassifier(
    n_estimators=best_random_n_estimators,
    max_depth=best_random_max_depth,
    min_samples_split=best_random_min_samples_split,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
random_model.fit(X_train, y_train)
random_pred = random_model.predict(X_test)

print("n_estimators:", best_random_n_estimators)
print("max_depth:", best_random_max_depth)
print("min_samples_split:", best_random_min_samples_split)
print("Accuracy:", round(accuracy_score(y_test, random_pred), 4))
print("Precision:", round(precision_score(y_test, random_pred, zero_division=0), 4))
print("Recall:", round(recall_score(y_test, random_pred, zero_division=0), 4))
print("F1:", round(f1_score(y_test, random_pred, zero_division=0), 4))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, random_pred))

print("\n============================================================\n")

# BEST GRID MODEL
print("===== BEST GRID MODEL (FINAL TEST) =====\n")

grid_model = RandomForestClassifier(
    n_estimators=best_grid_n_estimators,
    max_depth=best_grid_max_depth,
    min_samples_split=best_grid_min_samples_split,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
grid_model.fit(X_train, y_train)
grid_pred = grid_model.predict(X_test)

print("n_estimators:", best_grid_n_estimators)
print("max_depth:", best_grid_max_depth)
print("min_samples_split:", best_grid_min_samples_split)
print("Accuracy:", round(accuracy_score(y_test, grid_pred), 4))
print("Precision:", round(precision_score(y_test, grid_pred, zero_division=0), 4))
print("Recall:", round(recall_score(y_test, grid_pred, zero_division=0), 4))
print("F1:", round(f1_score(y_test, grid_pred, zero_division=0), 4))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, grid_pred))