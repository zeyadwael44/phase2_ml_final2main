from preprocessing import load_and_preprocess
from shared_utils import plot_svm_confusion_matrix, plot_svm_decision_boundary
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix,
)
from scipy.stats import loguniform
import numpy as np
import random

# Load data
X_train, X_test, y_train, y_test, skf, scaler, encoders = load_and_preprocess()

# Stratified sample (SVM is slow on 26k+ rows)
rng         = np.random.default_rng(42)
sample_size = 3000

y_arr  = y_train.to_numpy()
class0 = np.where(y_arr == 0)[0]
class1 = np.where(y_arr == 1)[0]
n0     = int(sample_size * len(class0) / len(y_arr))
n1     = sample_size - n0
idx    = np.sort(np.concatenate([
    rng.choice(class0, size=n0, replace=False),
    rng.choice(class1, size=n1, replace=False),
]))

X_sample = X_train[idx]
y_sample = y_train.iloc[idx].reset_index(drop=True)
print(f"Sampled {sample_size} rows (class 0: {n0}, class 1: {n1})")

skf_sample = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# GRID SEARCH
C_values      = [0.1, 1, 10]
kernel_values = ["linear", "rbf"]
gamma_values  = ["scale"]

best_grid_C      = None
best_grid_kernel = None
best_grid_gamma  = None
best_grid_score  = 0

print("\n===== GRID SEARCH =====\n")

for kernel in kernel_values:
    for C in C_values:
        for gamma in gamma_values:
            scores = []
            for train_index, val_index in skf_sample.split(X_sample, y_sample):
                X_tr  = X_sample[train_index]
                X_val = X_sample[val_index]
                y_tr  = y_sample.iloc[train_index]
                y_val = y_sample.iloc[val_index]

                svm = SVC(C=C, kernel=kernel, gamma=gamma,
                          class_weight="balanced", random_state=42, cache_size=1000)
                svm.fit(X_tr, y_tr)
                pred = svm.predict(X_val)
                scores.append(f1_score(y_val, pred))

            avg = np.mean(scores)
            print(f"kernel={kernel}, C={C}, gamma={gamma} | F1={avg:.4f}")

            if avg > best_grid_score:
                best_grid_score  = avg
                best_grid_C      = C
                best_grid_kernel = kernel
                best_grid_gamma  = gamma

print("\nBEST GRID RESULT")
print("C =", best_grid_C)
print("Kernel =", best_grid_kernel)
print("Gamma =", best_grid_gamma)
print("F1 =", round(best_grid_score, 4))

# RANDOM SEARCH
print("\n===== RANDOM SEARCH =====\n")

best_random_C      = None
best_random_kernel = None
best_random_gamma  = None
best_random_score  = 0

C_dist     = loguniform(0.01, 100)
gamma_opts = ["scale", 0.1]

for i in range(8):
    C      = round(float(C_dist.rvs(random_state=i)), 4)
    kernel = random.choice(["linear", "rbf"])
    gamma  = random.choice(gamma_opts)

    scores = []
    for train_index, val_index in skf_sample.split(X_sample, y_sample):
        X_tr  = X_sample[train_index]
        X_val = X_sample[val_index]
        y_tr  = y_sample.iloc[train_index]
        y_val = y_sample.iloc[val_index]

        svm = SVC(C=C, kernel=kernel, gamma=gamma,
                  class_weight="balanced", random_state=42, cache_size=1000)
        svm.fit(X_tr, y_tr)
        pred = svm.predict(X_val)
        scores.append(f1_score(y_val, pred))

    avg = np.mean(scores)
    print(f"Random kernel={kernel}, C={C}, gamma={gamma} | F1={avg:.4f}")

    if avg > best_random_score:
        best_random_score  = avg
        best_random_C      = C
        best_random_kernel = kernel
        best_random_gamma  = gamma

print("\nBEST RANDOM RESULT")
print("C =", best_random_C)
print("Kernel =", best_random_kernel)
print("Gamma =", best_random_gamma)
print("F1 =", round(best_random_score, 4))

# FINAL RESULTS SECTION
print("\n============================================================\n")

# BEST RANDOM MODEL
print("===== BEST RANDOM MODEL (FINAL TEST) =====\n")

random_model = SVC(
    C=best_random_C, kernel=best_random_kernel, gamma=best_random_gamma,
    class_weight="balanced", random_state=42, cache_size=1000,
)
random_model.fit(X_sample, y_sample)
random_pred = random_model.predict(X_test)

print("C:", best_random_C)
print("Kernel:", best_random_kernel)
print("Gamma:", best_random_gamma)
print("Accuracy:", round(accuracy_score(y_test, random_pred), 4))
print("Precision:", round(precision_score(y_test, random_pred, zero_division=0), 4))
print("Recall:", round(recall_score(y_test, random_pred, zero_division=0), 4))
print("F1:", round(f1_score(y_test, random_pred, zero_division=0), 4))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, random_pred))

plot_svm_confusion_matrix(
    "SVM Random Search Best",
    y_test, random_pred,
    "figure_svm_cm_random.png",
)

plot_svm_decision_boundary(
    "SVM Random Search Best",
    random_model, X_sample, y_sample, X_test, y_test,
    "figure_svm_boundary_random.png",
)

print("\n============================================================\n")

# BEST GRID MODEL
print("===== BEST GRID MODEL (FINAL TEST) =====\n")

grid_model = SVC(
    C=best_grid_C, kernel=best_grid_kernel, gamma=best_grid_gamma,
    class_weight="balanced", random_state=42, cache_size=1000,
)
grid_model.fit(X_sample, y_sample)
grid_pred = grid_model.predict(X_test)

print("C:", best_grid_C)
print("Kernel:", best_grid_kernel)
print("Gamma:", best_grid_gamma)
print("Accuracy:", round(accuracy_score(y_test, grid_pred), 4))
print("Precision:", round(precision_score(y_test, grid_pred, zero_division=0), 4))
print("Recall:", round(recall_score(y_test, grid_pred, zero_division=0), 4))
print("F1:", round(f1_score(y_test, grid_pred, zero_division=0), 4))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, grid_pred))

plot_svm_confusion_matrix(
    "SVM Grid Search Best",
    y_test, grid_pred,
    "figure_svm_cm_grid.png",
)

plot_svm_decision_boundary(
    "SVM Grid Search Best",
    grid_model, X_sample, y_sample, X_test, y_test,
    "figure_svm_boundary_grid.png",
)