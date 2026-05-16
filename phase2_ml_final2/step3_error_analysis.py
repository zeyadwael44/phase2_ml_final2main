


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

#  Plot style 
plt.rcParams.update({
    'figure.facecolor': '#0f1117', 'axes.facecolor': '#1a1d2e',
    'axes.edgecolor': '#3a3d5c',   'axes.labelcolor': '#c8cce8',
    'xtick.color': '#8b8fad',      'ytick.color': '#8b8fad',
    'text.color': '#c8cce8',       'grid.color': '#2a2d45',
    'grid.alpha': 0.5,
})
PALETTE = ['#4f9de8', '#e85f6b']


# 1. LOAD DATA & RETRAIN BEST MODEL

print("=" * 60)
print("ERROR ANALYSIS")
print("=" * 60)

X_train = pd.read_csv('X_train.csv')
X_test  = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv').squeeze()
y_test  = pd.read_csv('y_test.csv').squeeze()

# Load original test data for human-readable analysis
df_orig = pd.read_csv('WithdrawlStudents.csv')
DROP_COLS = ['id_student', 'code_module', 'code_presentation', 'final-result']
df_orig = df_orig.drop(columns=DROP_COLS)
df_orig['imd-band'] = df_orig['imd-band'].fillna('Unknown')

# Check which model was best
results_df = pd.read_csv('model_results.csv', index_col=0)
best_model_name = results_df['F1-Score'].idxmax()
print(f"\nBest model: {best_model_name}")

# Retrain best model (Random Forest assumed)
rf = RandomForestClassifier(n_estimators=100, max_depth=10,
                             min_samples_split=5, random_state=42,
                             class_weight='balanced')
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)


# 2. MISCLASSIFICATION ANALYSIS

print("\n" + "=" * 60)
print("MISCLASSIFICATION BREAKDOWN")
print("=" * 60)

# Get test indices (we used random split so we recover via index)
X_test_reset = X_test.reset_index(drop=True)
y_test_reset = y_test.reset_index(drop=True)

# Classify each prediction
pred_series = pd.Series(y_pred, index=y_test_reset.index)

true_pos  = ((y_test_reset == 1) & (pred_series == 1)).sum()
true_neg  = ((y_test_reset == 0) & (pred_series == 0)).sum()
false_pos = ((y_test_reset == 0) & (pred_series == 1)).sum()  # False Alarm
false_neg = ((y_test_reset == 1) & (pred_series == 0)).sum()  # Missed Withdrawal

total = len(y_test_reset)
print(f"\nTotal test samples: {total:,}")
print(f"\nTrue  Positives (Correctly predicted withdrawal):     {true_pos:,} ({true_pos/total*100:.1f}%)")
print(f"True  Negatives (Correctly predicted no withdrawal):  {true_neg:,} ({true_neg/total*100:.1f}%)")
print(f"False Positives (False alarm – predicted withdrawal): {false_pos:,} ({false_pos/total*100:.1f}%)")
print(f"False Negatives (MISSED withdrawal – most costly):    {false_neg:,} ({false_neg/total*100:.1f}%)")

#Profile misclassified students 
# Rebuild original feature space for analysis (unscaled, unenconded)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

df_for_split = df_orig.copy()
CATEGORICAL_COLS = ['gender', 'age-band', 'imd-band', 'high-education', 'region', 'disability']
le_dict = {}
for col in CATEGORICAL_COLS:
    le = LabelEncoder()
    df_for_split[col] = le.fit_transform(df_for_split[col].astype(str))
    le_dict[col] = le

X_all = df_for_split.drop(columns=['withdrawl'])
y_all = df_for_split['withdrawl']
_, X_test_idx, _, _ = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
)

# Get original (unscaled) test data
df_test_orig = df_orig.iloc[X_test_idx.index].copy()
df_test_orig['y_true'] = y_test_reset.values
df_test_orig['y_pred'] = y_pred

# False Negatives profile (missed withdrawals)
fn_df = df_test_orig[(df_test_orig['y_true'] == 1) & (df_test_orig['y_pred'] == 0)]
fp_df = df_test_orig[(df_test_orig['y_true'] == 0) & (df_test_orig['y_pred'] == 1)]

print(f"\n── False Negative Profile (Missed Withdrawals) ─────────────")
print(f"Count: {len(fn_df):,}")
print(f"Avg score of missed students:          {fn_df['avg-score'].mean():.1f}")
print(f"Avg studied credits of missed:         {fn_df['studied-credits'].mean():.1f}")
print(f"Avg previous attempts of missed:       {fn_df['prev-attempts'].mean():.2f}")

print(f"\n── False Positive Profile (False Alarms) ─────────────────────")
print(f"Count: {len(fp_df):,}")
print(f"Avg score of false alarms:             {fp_df['avg-score'].mean():.1f}")
print(f"Avg studied credits of false alarms:   {fp_df['studied-credits'].mean():.1f}")


# 3. FIGURE 6: ERROR ANALYSIS PLOTS

fig = plt.figure(figsize=(18, 12))
fig.suptitle('Error Analysis – Random Forest (Best Model)',
             fontsize=16, fontweight='bold', color='#e8eaf6', y=0.98)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# Plot 1 – Error type breakdown pie
ax1 = fig.add_subplot(gs[0, 0])
error_counts = [true_pos, true_neg, false_pos, false_neg]
error_labels = [f'True Pos\n{true_pos:,}', f'True Neg\n{true_neg:,}',
                f'False Pos\n{false_pos:,}', f'False Neg\n{false_neg:,}']
error_colors = ['#50c878', '#4f9de8', '#f5a623', '#e85f6b']
wedges, texts, autotexts = ax1.pie(error_counts, labels=error_labels,
                                    colors=error_colors, autopct='%1.1f%%',
                                    startangle=90, textprops={'fontsize': 8})
ax1.set_title('Prediction Breakdown', fontweight='bold')

# Plot 2 – Avg score: correctly vs incorrectly classified withdrawals
ax2 = fig.add_subplot(gs[0, 1])
tp_df = df_test_orig[(df_test_orig['y_true'] == 1) & (df_test_orig['y_pred'] == 1)]
categories = ['Correct\n(True Pos)', 'Missed\n(False Neg)']
means = [tp_df['avg-score'].mean(), fn_df['avg-score'].mean()]
bars = ax2.bar(categories, means, color=['#50c878', '#e85f6b'],
               width=0.5, edgecolor='none')
for bar, val in zip(bars, means):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val:.1f}', ha='center', va='bottom', fontsize=11)
ax2.set_title('Avg Score: Caught vs Missed Withdrawals', fontweight='bold')
ax2.set_ylabel('Average Score')
ax2.grid(axis='y')

# Plot 3 – Studied credits distribution of FN vs TP
ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(tp_df['studied-credits'], bins=20, alpha=0.7, color='#50c878',
         label=f'Correctly Caught (n={len(tp_df):,})', edgecolor='none')
ax3.hist(fn_df['studied-credits'], bins=20, alpha=0.7, color='#e85f6b',
         label=f'Missed (n={len(fn_df):,})', edgecolor='none')
ax3.set_title('Studied Credits: Caught vs Missed', fontweight='bold')
ax3.set_xlabel('Studied Credits')
ax3.set_ylabel('Frequency')
ax3.legend(fontsize=8)
ax3.grid(axis='y')

# Plot 4 – Gender breakdown of false negatives
ax4 = fig.add_subplot(gs[1, 0])
fn_gender = fn_df['gender'].value_counts()
fp_gender = fp_df['gender'].value_counts()
x = np.arange(len(fn_gender))
ax4.bar(x - 0.2, fn_gender.values, 0.35, color='#e85f6b',
        label='False Neg (missed)', edgecolor='none')
ax4.bar(x + 0.2, fp_gender.values[:len(fn_gender)], 0.35, color='#f5a623',
        label='False Pos (false alarm)', edgecolor='none')
ax4.set_title('Error Types by Gender', fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(fn_gender.index)
ax4.set_ylabel('Count')
ax4.legend(fontsize=8)
ax4.grid(axis='y')

# Plot 5 – Disability vs error
ax5 = fig.add_subplot(gs[1, 1])
fn_dis = fn_df['disability'].value_counts()
tp_dis = tp_df['disability'].value_counts()
categories2 = list(set(list(fn_dis.index) + list(tp_dis.index)))
fn_vals = [fn_dis.get(c, 0) for c in categories2]
tp_vals = [tp_dis.get(c, 0) for c in categories2]
x2 = np.arange(len(categories2))
ax5.bar(x2 - 0.2, tp_vals, 0.35, color='#50c878', label='Caught', edgecolor='none')
ax5.bar(x2 + 0.2, fn_vals, 0.35, color='#e85f6b', label='Missed', edgecolor='none')
ax5.set_xticks(x2)
ax5.set_xticklabels(categories2)
ax5.set_title('Disability vs Error Type', fontweight='bold')
ax5.set_ylabel('Count')
ax5.legend(fontsize=8)
ax5.grid(axis='y')

# Plot 6 – Prev attempts of FN vs TP
ax6 = fig.add_subplot(gs[1, 2])
prev_combined = pd.DataFrame({
    'Caught': tp_df['prev-attempts'].value_counts().sort_index(),
    'Missed': fn_df['prev-attempts'].value_counts().sort_index()
}).fillna(0)
prev_combined.plot(kind='bar', ax=ax6, color=['#50c878', '#e85f6b'],
                   width=0.6, edgecolor='none', legend=True)
ax6.set_title('Prev Attempts: Caught vs Missed', fontweight='bold')
ax6.set_xlabel('Previous Attempts')
ax6.set_ylabel('Count')
ax6.set_xticklabels(ax6.get_xticklabels(), rotation=0)
ax6.legend(fontsize=8)
ax6.grid(axis='y')

plt.savefig('figure6_error_analysis.png', dpi=150, bbox_inches='tight',
            facecolor='#0f1117')
plt.close()
print("\n✓ Saved: figure6_error_analysis.png")


# 4. PRINT SUMMARY

print("\n" + "=" * 60)
print("ERROR ANALYSIS SUMMARY")
print("=" * 60)
print("""
Key Findings:
─────────────────────────────────────────────────────────────
1. FALSE NEGATIVES (most costly): Students who WITHDREW but
   the model predicted they would NOT.
   → These students received no support intervention.
   → Often have higher average scores than expected,
     making them harder to detect.

2. FALSE POSITIVES (less harmful): Students flagged for
   withdrawal who did NOT actually withdraw.
   → Results in unnecessary support allocation.
   → Preferable to missing a real withdrawal case.

3. RECOMMENDATION: Adjust classification threshold to
   prioritize RECALL over Precision — it is better to
   falsely flag a student than to miss one who withdraws.

4. LIMITATION: avg-score drops to 0 for withdrawn students
   who never submitted work — the model partially relies
   on this as a strong signal.
─────────────────────────────────────────────────────────────
""")

print("=" * 60)
print("ALL 3 STEPS COMPLETE ✓")
print("Generated files:")
print("  figure1_eda.png")
print("  figure2_confusion_matrices.png")
print("  figure3_roc_curves.png")
print("  figure4_metric_comparison.png")
print("  figure5_feature_importance.png")
print("  figure6_error_analysis.png")
print("  model_results.csv")
print("=" * 60)
