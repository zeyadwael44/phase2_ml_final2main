from preprocessing import load_and_preprocess
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np



X_train, X_test, y_train, y_test, skf, scaler, encoders = load_and_preprocess()




grid_values = [1e-12, 1e-11, 1e-10, 1e-9, 1e-8]

random_values = np.logspace(-13, -7, 15)

best_smoothing = None
best_score = -1



def evaluate_model(smoothing):
    fold_scores = []

    for train_idx, val_idx in skf.split(X_train, y_train):

        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model = GaussianNB(var_smoothing=smoothing)
        model.fit(X_tr, y_tr)

        preds = model.predict(X_val)
        fold_scores.append(f1_score(y_val, preds))

    return np.mean(fold_scores)



print("\n===== GRID SEARCH =====\n")

for s in grid_values:
    score = evaluate_model(s)
    print(f"Grid | var_smoothing={s} | F1={score:.4f}")

    if score > best_score:
        best_score = score
        best_smoothing = s



print("\n===== RANDOM SEARCH =====\n")

random_samples = np.random.choice(random_values, size=5, replace=False)

for s in random_samples:
    score = evaluate_model(s)
    print(f"Random | var_smoothing={s:.2e} | F1={score:.4f}")

    if score > best_score:
        best_score = score
        best_smoothing = s



print("\n===== BEST RESULT =====\n")
print("Best var_smoothing:", best_smoothing)
print("Best CV F1:", round(best_score, 4))



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