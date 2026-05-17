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
k_values = [3, 5, 7, 9, 11]

best_grid_k       = 0
best_grid_weights = "uniform"
best_grid_metric  = "euclidean"
best_grid_score   = 0

print("\n===== GRID SEARCH =====\n")

for k in k_values:
    for weights in ["uniform", "distance"]:
        for metric in ["euclidean", "manhattan", "minkowski"]:

            scores = []

            for train_index, val_index in skf.split(X_train, y_train):

                X_tr  = X_train[train_index]
                X_val = X_train[val_index]

                y_tr  = y_train[train_index]
                y_val = y_train[val_index]

                knn = KNeighborsClassifier(n_neighbors=k, weights=weights, metric=metric)
                knn.fit(X_tr, y_tr)

                pred = knn.predict(X_val)
                scores.append(f1_score(y_val, pred))

            avg = np.mean(scores)

            print(f"K={k}, weights={weights}, metric={metric} | F1={avg:.4f}")

            if avg > best_grid_score:
                best_grid_score   = avg
                best_grid_k       = k
                best_grid_weights = weights
                best_grid_metric  = metric

print("\nBEST GRID RESULT")
print("K =", best_grid_k)
print("Weights =", best_grid_weights)
print("Metric =", best_grid_metric)
print("F1 =", round(best_grid_score, 4))


# RANDOM SEARCH
print("\n===== RANDOM SEARCH =====\n")

best_random_k       = None
best_random_weights = None
best_random_metric  = None
best_random_score   = 0

for i in range(1):

    k       = random.randint(1, 30)
    weights = random.choice(["uniform", "distance"])
    metric  = random.choice(["euclidean", "manhattan", "minkowski"])

    scores = []

    for train_index, val_index in skf.split(X_train, y_train):

        X_tr  = X_train[train_index]
        X_val = X_train[val_index]

        y_tr  = y_train[train_index]
        y_val = y_train[val_index]

        knn = KNeighborsClassifier(n_neighbors=k, weights=weights, metric=metric)
        knn.fit(X_tr, y_tr)

        pred = knn.predict(X_val)
        scores.append(f1_score(y_val, pred))

    avg = np.mean(scores)

    print(f"Random K={k}, weights={weights}, metric={metric} | F1={avg:.4f}")

    if avg > best_random_score:
        best_random_score   = avg
        best_random_k       = k
        best_random_weights = weights
        best_random_metric  = metric

print("\nBEST RANDOM RESULT")
print("K =", best_random_k)
print("Weights =", best_random_weights)
print("Metric =", best_random_metric)
print("F1 =", round(best_random_score, 4))


# FINAL RESULTS SECTION

print("\n============================================================\n")

# BEST RANDOM MODEL
print("===== BEST RANDOM MODEL (FINAL TEST) =====\n")

random_model = KNeighborsClassifier(
    n_neighbors=best_random_k,
    weights=best_random_weights,
    metric=best_random_metric
)

random_model.fit(X_train, y_train)
random_pred = random_model.predict(X_test)

print("K:", best_random_k)
print("Weights:", best_random_weights)
print("Metric:", best_random_metric)

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
    weights=best_grid_weights,
    metric=best_grid_metric
)

grid_model.fit(X_train, y_train)
grid_pred = grid_model.predict(X_test)

print("K:", best_grid_k)
print("Weights:", best_grid_weights)
print("Metric:", best_grid_metric)

print("Accuracy:", round(accuracy_score(y_test, grid_pred), 4))
print("Precision:", round(precision_score(y_test, grid_pred), 4))
print("Recall:", round(recall_score(y_test, grid_pred), 4))
print("F1:", round(f1_score(y_test, grid_pred), 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, grid_pred))


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def plot_decision_boundary(model, X, y, title):

    # Reduce to 2D for visualization
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    # Train a new model on the 2D data just for visualization
    visual_model = KNeighborsClassifier(n_neighbors=model.n_neighbors, weights=model.weights, metric=model.metric)
    visual_model.fit(X_2d, y)

    # Create a mesh grid to cover the 2D space
    step = 0.2
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, step),
        np.arange(y_min, y_max, step)
    )

    # Predict every point in the mesh
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = visual_model.predict(mesh_points)
    Z = Z.reshape(xx.shape)

    # Plot the decision regions and data points
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap="coolwarm", edgecolor="k", s=40)

    plt.title(title)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.tight_layout()
    plt.show()


plot_decision_boundary(grid_model, X_train, y_train, "Grid Search KNN Decision Boundary")
plot_decision_boundary(random_model, X_train, y_train, "Random Search KNN Decision Boundary")