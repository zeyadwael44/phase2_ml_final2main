import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve,
    ConfusionMatrixDisplay,
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


# ── SVM: Lab-7-style confusion matrix ────────────────────────────────────────
def plot_svm_confusion_matrix(name, y_test, y_pred, filename):
    """
    Render a ConfusionMatrixDisplay (Lab-7 style) for an SVM result and
    save it to `filename`.

    Parameters
    ----------
    name     : str   – model label shown in the title (e.g. "SVM Grid Best")
    y_test   : array-like – true labels
    y_pred   : array-like – predicted labels
    filename : str   – path to save the PNG

    Usage in model_svm.py
    ----------------------
        from shared_utils import plot_svm_confusion_matrix
        plot_svm_confusion_matrix(
            "SVM Grid Best",
            y_test, grid_pred,
            "figure_svm_cm_grid.png",
        )
    """
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor('#0f1117')
    ax.set_facecolor('#1a1d2e')

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=['Stay (0)', 'Withdraw (1)'],
    )
    disp.plot(
        cmap='Blues',
        colorbar=False,
        ax=ax,
        values_format='d',
    )

    # Style the text inside cells so it reads on the dark background
    for text in disp.text_.ravel():
        text.set_fontsize(14)
        text.set_fontweight('bold')

    # Axis cosmetics to match the project's dark theme
    ax.set_title(f'Confusion Matrix — {name}',
                 fontsize=13, fontweight='bold', color='#e8eaf6', pad=12)
    ax.set_xlabel('Predicted Label', fontsize=11, color='#c8cce8', labelpad=8)
    ax.set_ylabel('Actual Label',    fontsize=11, color='#c8cce8', labelpad=8)
    ax.tick_params(colors='#8b8fad')
    for spine in ax.spines.values():
        spine.set_edgecolor('#3a3d5c')

    # Annotate each cell with its TN / FP / FN / TP role
    roles = [['TN', 'FP'], ['FN', 'TP']]
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            pct = cm[i, j] / total * 100
            ax.text(
                j, i + 0.35,
                f'({roles[i][j]})  {pct:.1f}%',
                ha='center', va='center',
                fontsize=9, color='#8b8fad',
            )

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='#0f1117')
    plt.close()
    print(f"  Saved: {filename}")


# ── SVM: Lab-7-style decision boundary (PCA → 2D) ────────────────────────────
def plot_svm_decision_boundary(name, model, X_train, y_train, X_test, y_test, filename):
    """
    Project high-dimensional SVM data down to 2 PCA components and draw the
    decision boundary exactly like Lab 7's plot_2d_svm:
      - filled contour background (coolwarm)
      - dashed margin lines at decision_function = ±1
      - solid decision boundary at decision_function = 0
      - scatter of train points (faded) + test points (solid)
      - support vectors highlighted in gold

    Because the SVM was trained in the original feature space, we:
      1. Fit PCA on X_train
      2. Project both splits to 2D for plotting
      3. Re-train a *display-only* SVC in that 2D space so we can call
         decision_function on the mesh — this model is never used for
         evaluation, only for drawing the boundary.

    Parameters
    ----------
    name     : str   – label shown in the title (e.g. "SVM Grid Best")
    model    : fitted SVC – used to copy C / kernel / gamma / class_weight
    X_train  : ndarray – training features (original space)
    y_train  : Series  – training labels
    X_test   : ndarray – test features (original space)
    y_test   : Series  – test labels
    filename : str   – path to save the PNG

    Usage in model_svm.py
    ----------------------
        from shared_utils import plot_svm_decision_boundary
        plot_svm_decision_boundary(
            "SVM Grid Best",
            grid_model, X_sample, y_sample, X_test, y_test,
            "figure_svm_boundary_grid.png",
        )
    """
    from sklearn.decomposition import PCA
    from sklearn.svm import SVC as _SVC

    # ── 1. Reduce to 2D ──────────────────────────────────────────────────────
    pca = PCA(n_components=2, random_state=42)
    X_tr_2d   = pca.fit_transform(X_train)
    X_te_2d   = pca.transform(X_test)
    y_tr      = np.array(y_train)
    y_te      = np.array(y_test)

    # ── 2. Re-train a display-only SVC in 2D ─────────────────────────────────
    params = model.get_params()
    display_svc = _SVC(
        C            = params['C'],
        kernel       = params['kernel'],
        gamma        = params['gamma'],
        class_weight = params['class_weight'],
        random_state = 42,
        cache_size   = 500,
    )
    display_svc.fit(X_tr_2d, y_tr)

    # ── 3. Build mesh ─────────────────────────────────────────────────────────
    pad = 0.6
    x_min, x_max = X_tr_2d[:, 0].min() - pad, X_tr_2d[:, 0].max() + pad
    y_min, y_max = X_tr_2d[:, 1].min() - pad, X_tr_2d[:, 1].max() + pad
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 400),
        np.linspace(y_min, y_max, 400),
    )
    Z = display_svc.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # ── 4. Plot ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#0f1117')
    fig.suptitle(
        f'SVM Decision Boundary (PCA 2D) — {name}',
        fontsize=14, fontweight='bold', color='#e8eaf6', y=1.01,
    )

    subsets = [
        (X_tr_2d, y_tr, 'Training data',  0.35),
        (X_te_2d, y_te, 'Test data',       0.85),
    ]

    for ax, (X_2d, y_lbl, subset_label, alpha_pts) in zip(axes, subsets):
        ax.set_facecolor('#1a1d2e')

        # Filled background
        ax.contourf(xx, yy, Z, levels=40, cmap='coolwarm', alpha=0.18)

        # Margin lines (±1) and decision boundary (0) — Lab-7 style
        ax.contour(
            xx, yy, Z,
            levels=[-1, 0, 1],
            colors=['#6ab0f5', '#ffffff', '#f5826a'],
            linestyles=['--', '-', '--'],
            linewidths=[1.2, 2.0, 1.2],
        )

        # Data points
        ax.scatter(
            X_2d[:, 0], X_2d[:, 1],
            c=y_lbl, cmap='coolwarm',
            edgecolor='k', linewidths=0.4,
            s=30, alpha=alpha_pts,
        )

        # Support vectors (from the 2D display model)
        sv = display_svc.support_vectors_
        ax.scatter(
            sv[:, 0], sv[:, 1],
            s=120, facecolors='none',
            edgecolors='goldenrod', linewidths=1.6,
            label='Support vectors',
        )

        # Legend for classes + support vectors
        import matplotlib.patches as mpatches
        from matplotlib.lines import Line2D
        handles = [
            mpatches.Patch(color='#4f9de8', label='Stay (0)'),
            mpatches.Patch(color='#e85f6b', label='Withdraw (1)'),
            Line2D([0], [0], marker='o', color='none',
                   markerfacecolor='none', markeredgecolor='goldenrod',
                   markersize=9, markeredgewidth=1.6, label='Support vectors'),
        ]
        ax.legend(handles=handles, fontsize=8, loc='upper right', framealpha=0.2)

        ax.set_title(subset_label, fontsize=11, color='#c8cce8', pad=6)
        ax.set_xlabel('PCA Component 1', fontsize=9, color='#8b8fad')
        ax.set_ylabel('PCA Component 2', fontsize=9, color='#8b8fad')
        ax.tick_params(colors='#8b8fad', labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#3a3d5c')

    # Explained variance annotation
    ev = pca.explained_variance_ratio_
    fig.text(
        0.5, -0.03,
        f'PCA explains {ev[0]*100:.1f}% + {ev[1]*100:.1f}% = {sum(ev)*100:.1f}% of variance',
        ha='center', fontsize=9, color='#8b8fad',
    )

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='#0f1117')
    plt.close()
    print(f"  Saved: {filename}")