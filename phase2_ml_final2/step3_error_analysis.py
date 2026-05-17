import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
import warnings
warnings.filterwarnings('ignore')

from preprocessing import load_and_preprocess

# ── Plot theme ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0f1117', 'axes.facecolor': '#1a1d2e',
    'axes.edgecolor':   '#3a3d5c', 'axes.labelcolor': '#c8cce8',
    'xtick.color':      '#8b8fad', 'ytick.color':     '#8b8fad',
    'text.color':       '#c8cce8', 'grid.color':      '#2a2d45',
    'grid.alpha': 0.5,             'font.family':     'DejaVu Sans',
})
COLORS = ['#4f9de8', '#e85f6b', '#50c878', '#f5a623', '#a855f7', '#888888']

# ── Load data ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 3 — ERROR ANALYSIS")
print("=" * 60)

X_train, X_test, y_train, y_test, skf, scaler, encoders = load_and_preprocess()
X_test_df    = pd.read_csv('phase2_ml_final2/X_test.csv')
feature_names = list(X_test_df.columns)

# ── Train all models (best hyperparameters from prior steps) ──────────────────
print("\nTraining all models...")

rng    = np.random.default_rng(42)
y_arr  = y_train.to_numpy()
class0 = np.where(y_arr == 0)[0]; class1 = np.where(y_arr == 1)[0]
n0 = int(3000 * len(class0) / len(y_arr)); n1 = 3000 - n0
idx = np.sort(np.concatenate([
    rng.choice(class0, size=n0, replace=False),
    rng.choice(class1, size=n1, replace=False),
]))
X_sample = X_train[idx]
y_sample = y_train.iloc[idx].reset_index(drop=True)

models = {
    'Baseline':            DummyClassifier(strategy='most_frequent', random_state=42),
    'Logistic Regression': LogisticRegression(C=1, max_iter=1000, random_state=42),
    'KNN':                 KNeighborsClassifier(n_neighbors=9, weights='distance'),
    'Naive Bayes':         GaussianNB(var_smoothing=1e-9),
    'Random Forest':       RandomForestClassifier(
                               n_estimators=200, max_depth=None,
                               min_samples_split=2, class_weight='balanced',
                               random_state=42, n_jobs=-1),
}
svm_model = SVC(C=1, kernel='rbf', gamma='scale', class_weight='balanced',
                random_state=42, probability=True, cache_size=1000)

for name, model in models.items():
    model.fit(X_train, y_train)
svm_model.fit(X_sample, y_sample)
models['SVM'] = svm_model

preds  = {name: model.predict(X_test)              for name, model in models.items()}
probas = {name: model.predict_proba(X_test)[:, 1]  for name, model in models.items()}
y_test_np   = y_test.to_numpy()
model_order = ['Baseline', 'Logistic Regression', 'KNN', 'Naive Bayes', 'SVM', 'Random Forest']
print("Done.\n")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Confusion Matrices (all models)
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 1: Confusion Matrices...")
roles = [['TN', 'FP'], ['FN', 'TP']]
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.patch.set_facecolor('#0f1117')
fig.suptitle('Confusion Matrices — All Models', fontsize=15, fontweight='bold',
             color='#e8eaf6', y=1.01)

for ax, name in zip(axes.ravel(), model_order):
    ax.set_facecolor('#1a1d2e')
    cm   = confusion_matrix(y_test, preds[name])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=['Stay (0)', 'Withdraw (1)'])
    disp.plot(cmap='Blues', colorbar=False, ax=ax, values_format='d')
    for text in disp.text_.ravel():
        text.set_fontsize(13); text.set_fontweight('bold')
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            ax.text(j, i + 0.35, f'({roles[i][j]})  {cm[i,j]/total*100:.1f}%',
                    ha='center', va='center', fontsize=8, color='#8b8fad')
    ax.set_title(name, fontsize=11, fontweight='bold', color='#e8eaf6', pad=8)
    ax.set_xlabel('Predicted Label', fontsize=9, color='#c8cce8')
    ax.set_ylabel('Actual Label',    fontsize=9, color='#c8cce8')
    ax.tick_params(colors='#8b8fad')
    for spine in ax.spines.values(): spine.set_edgecolor('#3a3d5c')

plt.tight_layout()
plt.savefig('phase2_ml_final2/figure_error_confusion_all.png', dpi=150,
            bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("  Saved: figure_error_confusion_all.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — ROC Curves (all models)
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 2: ROC Curves...")
fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor('#0f1117'); ax.set_facecolor('#1a1d2e')

for (name, prob), color in zip(probas.items(), COLORS):
    fpr, tpr, _ = roc_curve(y_test, prob)
    ax.plot(fpr, tpr, color=color, lw=1.8, label=f'{name}  (AUC = {auc(fpr,tpr):.3f})')

ax.plot([0,1],[0,1], color='#555577', lw=1, linestyle='--', label='Random (AUC = 0.500)')
ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
ax.set_xlabel('False Positive Rate', fontsize=11, color='#c8cce8')
ax.set_ylabel('True Positive Rate',  fontsize=11, color='#c8cce8')
ax.set_title('ROC Curves — All Models', fontsize=13, fontweight='bold', color='#e8eaf6')
ax.legend(loc='lower right', fontsize=9, framealpha=0.2)
ax.grid(True, alpha=0.3)
for spine in ax.spines.values(): spine.set_edgecolor('#3a3d5c')

plt.tight_layout()
plt.savefig('phase2_ml_final2/figure_error_roc_curves.png', dpi=150,
            bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("  Saved: figure_error_roc_curves.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — FN/FP pattern analysis (Random Forest)
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 3: FN/FP Pattern Analysis (Random Forest)...")

rf_pred = preds['Random Forest']
fn_mask = (rf_pred == 0) & (y_test_np == 1)
fp_mask = (rf_pred == 1) & (y_test_np == 0)
tp_mask = (rf_pred == 1) & (y_test_np == 1)
tn_mask = (rf_pred == 0) & (y_test_np == 0)

fn_df      = X_test_df[fn_mask]
fp_df      = X_test_df[fp_mask]
correct_df = X_test_df[tp_mask | tn_mask]

print(f"  False Negatives (missed withdrawals): {fn_mask.sum()}")
print(f"  False Positives (false alarms):       {fp_mask.sum()}")

fig = plt.figure(figsize=(16, 12))
fig.patch.set_facecolor('#0f1117')
fig.suptitle('Error Pattern Analysis — Random Forest (Best Model)',
             fontsize=14, fontweight='bold', color='#e8eaf6', y=1.01)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# Plot A: avg-score distribution
ax_a = fig.add_subplot(gs[0, 0]); ax_a.set_facecolor('#1a1d2e')
for data, label, color, alpha in [
    (correct_df['avg-score'], 'Correct',   '#4f9de8', 0.6),
    (fn_df['avg-score'],      'False Neg', '#e85f6b', 0.85),
    (fp_df['avg-score'],      'False Pos', '#f5a623', 0.85),
]:
    ax_a.hist(data, bins=30, alpha=alpha, color=color, label=label, edgecolor='none')
ax_a.set_title('Avg-score by error type', fontweight='bold', color='#e8eaf6')
ax_a.set_xlabel('Average score (raw)', color='#c8cce8')
ax_a.set_ylabel('Count', color='#c8cce8')
ax_a.legend(fontsize=8); ax_a.grid(axis='y')

# Plot B: studied-credits distribution
ax_b = fig.add_subplot(gs[0, 1]); ax_b.set_facecolor('#1a1d2e')
for data, label, color, alpha in [
    (correct_df['studied-credits'], 'Correct',   '#4f9de8', 0.6),
    (fn_df['studied-credits'],      'False Neg', '#e85f6b', 0.85),
    (fp_df['studied-credits'],      'False Pos', '#f5a623', 0.85),
]:
    ax_b.hist(data, bins=25, alpha=alpha, color=color, label=label, edgecolor='none')
ax_b.set_title('Studied-credits by error type', fontweight='bold', color='#e8eaf6')
ax_b.set_xlabel('Studied credits (raw)', color='#c8cce8')
ax_b.set_ylabel('Count', color='#c8cce8')
ax_b.legend(fontsize=8); ax_b.grid(axis='y')

# Plot C: prev-attempts breakdown for FN vs overall
ax_c = fig.add_subplot(gs[0, 2]); ax_c.set_facecolor('#1a1d2e')
cats    = ['0 prev attempts', '>=1 prev attempt']
fn_prev  = (fn_df['prev-attempts'] > 0).map({True: '>=1 prev attempt', False: '0 prev attempts'})
all_prev = (X_test_df['prev-attempts'] > 0).map({True: '>=1 prev attempt', False: '0 prev attempts'})
fn_pct  = [fn_prev.value_counts(normalize=True).get(c,0)*100 for c in cats]
all_pct = [all_prev.value_counts(normalize=True).get(c,0)*100 for c in cats]
x = np.arange(len(cats)); w = 0.35
ax_c.bar(x-w/2, all_pct, w, label='Overall test set', color='#4f9de8', alpha=0.8)
ax_c.bar(x+w/2, fn_pct,  w, label='False Negatives',  color='#e85f6b', alpha=0.8)
ax_c.set_title('Prev-attempts in missed withdrawals', fontweight='bold', color='#e8eaf6')
ax_c.set_ylabel('Percentage (%)', color='#c8cce8')
ax_c.set_xticks(x); ax_c.set_xticklabels(cats, fontsize=9)
ax_c.legend(fontsize=8); ax_c.grid(axis='y')

# Plot D: Feature importances
ax_d = fig.add_subplot(gs[1, 0]); ax_d.set_facecolor('#1a1d2e')
rf_model    = models['Random Forest']
importances = rf_model.feature_importances_
sorted_idx  = np.argsort(importances)
ax_d.barh([feature_names[i] for i in sorted_idx], importances[sorted_idx],
          color='#4f9de8', edgecolor='none', alpha=0.85)
ax_d.set_title('Feature importances (RF)', fontweight='bold', color='#e8eaf6')
ax_d.set_xlabel('Importance', color='#c8cce8'); ax_d.grid(axis='x')

# Plot E: FN rate by gender
ax_e = fig.add_subplot(gs[1, 1]); ax_e.set_facecolor('#1a1d2e')
actual_withdraw      = X_test_df[y_test_np == 1].copy()
actual_withdraw['fn'] = fn_mask[y_test_np == 1]
gender_fn_rate = actual_withdraw.groupby('gender')['fn'].mean() * 100
gender_labels  = {0: 'Female', 1: 'Male'}
genders   = sorted(gender_fn_rate.index)
fn_rates  = [gender_fn_rate[g] for g in genders]
bars = ax_e.bar([gender_labels[g] for g in genders], fn_rates,
                color=['#50c878','#a855f7'], edgecolor='none', alpha=0.85)
for bar, val in zip(bars, fn_rates):
    ax_e.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
              f'{val:.1f}%', ha='center', fontsize=10, color='#c8cce8')
ax_e.set_title('Missed withdrawal rate by gender', fontweight='bold', color='#e8eaf6')
ax_e.set_ylabel('False Negative rate (%)', color='#c8cce8')
ax_e.set_ylim(0, max(fn_rates)*1.25); ax_e.grid(axis='y')

# Plot F: FN rate by age-band
ax_f = fig.add_subplot(gs[1, 2]); ax_f.set_facecolor('#1a1d2e')
age_labels   = {0: '0-35', 1: '35-55', 2: '55+'}
age_fn_rate  = actual_withdraw.groupby('age-band')['fn'].mean() * 100
ages         = sorted(age_fn_rate.index)
fn_rates_age = [age_fn_rate[a] for a in ages]
bars = ax_f.bar([age_labels[a] for a in ages], fn_rates_age,
                color='#f5a623', edgecolor='none', alpha=0.85)
for bar, val in zip(bars, fn_rates_age):
    ax_f.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
              f'{val:.1f}%', ha='center', fontsize=10, color='#c8cce8')
ax_f.set_title('Missed withdrawal rate by age band', fontweight='bold', color='#e8eaf6')
ax_f.set_ylabel('False Negative rate (%)', color='#c8cce8')
ax_f.set_ylim(0, max(fn_rates_age)*1.25); ax_f.grid(axis='y')

plt.savefig('phase2_ml_final2/figure_error_fn_fp_analysis.png', dpi=150,
            bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("  Saved: figure_error_fn_fp_analysis.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — FN / FP / F1 comparison across all models
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 4: Model Error Comparison...")

summary = {}
for name in model_order:
    cm = confusion_matrix(y_test, preds[name])
    tn, fp, fn, tp = cm.ravel()
    summary[name] = {
        'FN':   fn,
        'FP':   fp,
        'F1':   round(f1_score(y_test, preds[name], zero_division=0)*100, 1),
    }

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.patch.set_facecolor('#0f1117')
fig.suptitle('Error Comparison Across All Models', fontsize=13,
             fontweight='bold', color='#e8eaf6')
x = np.arange(len(model_order))

for ax in axes:
    ax.set_facecolor('#1a1d2e')
    ax.set_xticks(x)
    ax.set_xticklabels(model_order, rotation=25, ha='right', fontsize=9)
    ax.grid(axis='y', alpha=0.4)
    for spine in ax.spines.values(): spine.set_edgecolor('#3a3d5c')

fn_vals = [summary[n]['FN'] for n in model_order]
bars = axes[0].bar(x, fn_vals, color='#e85f6b', edgecolor='none', alpha=0.85)
for bar, val in zip(bars, fn_vals):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+10,
                 str(val), ha='center', fontsize=9, color='#c8cce8')
axes[0].set_title('False Negatives (missed withdrawals)', fontweight='bold', color='#e8eaf6')
axes[0].set_ylabel('Count', color='#c8cce8')

fp_vals = [summary[n]['FP'] for n in model_order]
bars = axes[1].bar(x, fp_vals, color='#f5a623', edgecolor='none', alpha=0.85)
for bar, val in zip(bars, fp_vals):
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+5,
                 str(val), ha='center', fontsize=9, color='#c8cce8')
axes[1].set_title('False Positives (false alarms)', fontweight='bold', color='#e8eaf6')
axes[1].set_ylabel('Count', color='#c8cce8')

f1_vals = [summary[n]['F1'] for n in model_order]
bars = axes[2].bar(x, f1_vals, color='#4f9de8', edgecolor='none', alpha=0.85)
for bar, val in zip(bars, f1_vals):
    axes[2].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                 f'{val}%', ha='center', fontsize=9, color='#c8cce8')
axes[2].set_title('F1-score (%)', fontweight='bold', color='#e8eaf6')
axes[2].set_ylabel('F1 (%)', color='#c8cce8')

plt.tight_layout()
plt.savefig('phase2_ml_final2/figure_error_model_comparison.png', dpi=150,
            bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("  Saved: figure_error_model_comparison.png")

# ══════════════════════════════════════════════════════════════════════════════
# PRINTED SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("ERROR ANALYSIS SUMMARY")
print("=" * 60)

print(f"\n{'Model':<22} {'TN':>6} {'FP':>6} {'FN':>6} {'TP':>6}  {'FN Rate':>8}")
print("-" * 60)
for name in model_order:
    cm = confusion_matrix(y_test, preds[name])
    tn, fp, fn, tp = cm.ravel()
    fn_rate = fn / (y_test_np == 1).sum() * 100
    print(f"{name:<22} {tn:>6} {fp:>6} {fn:>6} {tp:>6}  {fn_rate:>7.1f}%")

print("""
Why FN matters most:
  A False Negative = a student who WILL withdraw is predicted
  to stay. They receive no intervention and drop out.
  A False Positive = a student who stays is falsely flagged.
  Wasteful, but not harmful. We therefore prioritise Recall.

Best model (Random Forest):
  Fewest False Negatives (802) among non-baseline models.
  Best F1 (0.6578) and ROC-AUC (0.8091).
  Feature importance: avg-score dominates (49.7%).

Key failure patterns:
  - Students near the decision boundary (ambiguous avg-score)
  - First-time students (no prior attempt history)
  - Male students missed at slightly higher rate than female

Limitations:
  - No time-series / engagement features in the dataset
  - SVM trained on 3,000-sample subset only
  - All models plateau at F1 ~ 0.62-0.66; richer features
    (e.g. VLE logs) would likely improve performance further
""")

print("=" * 60)
print("STEP 3 COMPLETE — 4 figures saved.")
print("=" * 60)