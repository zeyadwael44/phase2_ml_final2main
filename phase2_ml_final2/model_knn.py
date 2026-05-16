from preprocessing import load_and_preprocess
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix,
)
import numpy as np
import random
# Load data
X_train, X_test, y_train, y_test, skf, scaler, encoders = load_and_preprocess()

# GRID SEARCH
k_values = [1, 3, 5, 7, 9, 11]

best_grid_k = 0
best_grid_weights = "uniform"
best_grid_score = 0

print("\n===== GRID SEARCH =====\n")

for k in k_values:
    for weights in ["uniform", "distance"]:

        scores = []

        for train_index, val_index in skf.split(X_train, y_train):

            X_tr  = X_train[train_index]
            X_val = X_train[val_index]

            y_tr  = y_train[train_index]
            y_val = y_train[val_index]

            knn = KNeighborsClassifier(n_neighbors=k, weights=weights)
            knn.fit(X_tr, y_tr)

            pred = knn.predict(X_val)
            scores.append(f1_score(y_val, pred))

        avg = np.mean(scores)

        print(f"K={k}, weights={weights} | F1={avg:.4f}")

        if avg > best_grid_score:
            best_grid_score = avg
            best_grid_k = k
            best_grid_weights = weights

print("\nBEST GRID RESULT")
print("K =", best_grid_k)
print("Weights =", best_grid_weights)
print("F1 =", round(best_grid_score, 4))


# RANDOM SEARCH
print("\n===== RANDOM SEARCH =====\n")

best_random_k = None
best_random_weights = None
best_random_score = 0

for i in range(10):

    k = random.randint(1, 20)
    weights = random.choice(["uniform", "distance"])

    scores = []

    for train_index, val_index in skf.split(X_train, y_train):

        X_tr  = X_train[train_index]
        X_val = X_train[val_index]

        y_tr  = y_train[train_index]
        y_val = y_train[val_index]

        knn = KNeighborsClassifier(n_neighbors=k, weights=weights)
        knn.fit(X_tr, y_tr)

        pred = knn.predict(X_val)
        scores.append(f1_score(y_val, pred))

    avg = np.mean(scores)

    print(f"Random K={k}, weights={weights} | F1={avg:.4f}")

    if avg > best_random_score:
        best_random_score = avg
        best_random_k = k
        best_random_weights = weights

print("\nBEST RANDOM RESULT")
print("K =", best_random_k)
print("Weights =", best_random_weights)
print("F1 =", round(best_random_score, 4))


# FINAL RESULTS SECTION

print("\n============================================================\n")

# BEST RANDOM MODEL
print("===== BEST RANDOM MODEL (FINAL TEST) =====\n")

random_model = KNeighborsClassifier(
    n_neighbors=best_random_k,
    weights=best_random_weights
)

random_model.fit(X_train, y_train)
random_pred = random_model.predict(X_test)

print("K:", best_random_k)
print("Weights:", best_random_weights)

print("Accuracy:", round(accuracy_score(y_test, random_pred), 4))
print("Precision:", round(precision_score(y_test, random_pred), 4))
print("Recall:", round(recall_score(y_test, random_pred), 4))
print("F1:", round(f1_score(y_test, random_pred), 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, random_pred))


print("\n============================================================\n")

# BEST GRID MODEL
print("===== BEST GRID MODEL (FINAL TEST) =====\n")

grid_model = KNeighborsClassifier(
    n_neighbors=best_grid_k,
    weights=best_grid_weights
)

grid_model.fit(X_train, y_train)
grid_pred = grid_model.predict(X_test)

print("K:", best_grid_k)
print("Weights:", best_grid_weights)

print("Accuracy:", round(accuracy_score(y_test, grid_pred), 4))
print("Precision:", round(precision_score(y_test, grid_pred), 4))
print("Recall:", round(recall_score(y_test, grid_pred), 4))
print("F1:", round(f1_score(y_test, grid_pred), 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, grid_pred))

