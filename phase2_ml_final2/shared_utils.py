
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
import warnings
warnings.filterwarnings('ignore')

# Plot style 
plt.rcParams.update({
    'figure.facecolor': '#0f1117', 'axes.facecolor': '#1a1d2e',
    'axes.edgecolor':   '#3a3d5c', 'axes.labelcolor': '#c8cce8',
    'xtick.color':      '#8b8fad', 'ytick.color':     '#8b8fad',
    'text.color':       '#c8cce8', 'grid.color':      '#2a2d45',
    'grid.alpha': 0.5,             'font.family':     'DejaVu Sans',
})
COLORS = ['#4f9de8', '#e85f6b', '#50c878', '#f5a623', '#a855f7']


# Data loading 
def load_data():
    
    X_train = pd.read_csv('X_train.csv')
    X_test  = pd.read_csv('X_test.csv')
    y_train = pd.read_csv('y_train.csv').squeeze()
    y_test  = pd.read_csv('y_test.csv').squeeze()
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


#  Cross-validation helper 
def cross_validate_model(name, model, X_train, y_train, n_splits=5):
    
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    scoring = {
        'accuracy' : 'accuracy',
        'precision': 'precision',
        'recall'   : 'recall',
        'f1'       : 'f1',
        'roc_auc'  : 'roc_auc',
    }

    cv_results = cross_validate(
        model, X_train, y_train,
        cv=cv, scoring=scoring,
        return_train_score=False
    )

    print(f"\n{'─'*55}")
    print(f"Stratified {n_splits}-Fold Cross-Validation  —  {name}")
    print(f"{'─'*55}")

    summary = {}
    for metric, key in [
        ('Accuracy',  'test_accuracy'),
        ('Precision', 'test_precision'),
        ('Recall',    'test_recall'),
        ('F1-Score',  'test_f1'),
        ('ROC-AUC',   'test_roc_auc'),
    ]:
        scores = cv_results[key]
        mean, std = scores.mean(), scores.std()
        fold_str = '  '.join([f'{s:.4f}' for s in scores])
        print(f"  {metric:<12} folds : [{fold_str}]")
        print(f"  {metric:<12} mean  : {mean:.4f} +/- {std:.4f}")
        summary[metric] = (mean, std)

    print(f"{'─'*55}")
    return summary


# Test-set evaluation helper 
def evaluate_model(name, model, X_test, y_test):
    
    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_pred_prob)
    cm   = confusion_matrix(y_test, y_pred)

    print(f"\nTest-Set Results  —  {name}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {auc:.4f}")

    tn, fp, fn, tp = cm.ravel()
    print(f"\n  Confusion Matrix:")
    print(f"    True  Negatives (TN): {tn:,}  — correctly predicted Stay")
    print(f"    False Positives (FP): {fp:,}  — predicted Withdraw, actually Stayed")
    print(f"    False Negatives (FN): {fn:,}  — predicted Stay, actually Withdrew  ← costly")
    print(f"    True  Positives (TP): {tp:,}  — correctly predicted Withdraw")

    return {
        'Accuracy':    acc,
        'Precision':   prec,
        'Recall':      rec,
        'F1-Score':    f1,
        'ROC-AUC':     auc,
        'CM':          cm,
        'y_pred_prob': y_pred_prob,
    }


# NEW: Confusion matrix visualisation 
def plot_confusion_matrix(name, results, filename):
    
    cm    = results['CM']
    total = cm.sum()

    # Per-cell text: count + role + % of total
    labels = [
        [f"{cm[0,0]}\n(TN)\n{cm[0,0]/total*100:.1f}%",
         f"{cm[0,1]}\n(FP)\n{cm[0,1]/total*100:.1f}%"],
        [f"{cm[1,0]}\n(FN)\n{cm[1,0]/total*100:.1f}%",
         f"{cm[1,1]}\n(TP)\n{cm[1,1]/total*100:.1f}%"],
    ]

    # Blue = correct, Red = error
    cell_colors = [
        ['#185FA5', "#F5EFED"],   # row 0: TN=blue, FP=red
        ["#DAD2CF", '#185FA5'],   # row 1: FN=red,  TP=blue
    ]

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor('#0f1117')

    for i in range(2):
        for j in range(2):
            ax.add_patch(plt.Rectangle(
                (j - 0.5, i - 0.5), 1, 1,
                color=cell_colors[i][j], alpha=0.75, zorder=0
            ))
            ax.text(
                j, i, labels[i][j],
                ha='center', va='center',
                fontsize=13, fontweight='bold', color='white', zorder=1
            )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted: Stay (0)', 'Predicted: Withdraw (1)'], fontsize=10)
    ax.set_yticklabels(['Actual: Stay (0)', 'Actual: Withdraw (1)'], fontsize=10)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_title(f'Confusion Matrix — {name}',
                 fontsize=13, fontweight='bold', color='#e8eaf6', pad=14)
    ax.set_xlabel('Predicted Label', fontsize=11, labelpad=8)
    ax.set_ylabel('Actual Label',    fontsize=11, labelpad=8)

    legend_handles = [
        mpatches.Patch(color='#185FA5', alpha=0.75, label='Correct prediction (TN / TP)'),
        mpatches.Patch(color='#993C1D', alpha=0.75, label='Incorrect prediction (FP / FN)'),
    ]
    ax.legend(handles=legend_handles, loc='upper center',
              bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=9, framealpha=0)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='#0f1117')
    plt.close()
    print(f"  Saved: {filename}")