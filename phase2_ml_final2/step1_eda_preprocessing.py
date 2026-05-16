
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

#  Consistent plot style 
plt.rcParams.update({
    'figure.facecolor': '#0f1117',
    'axes.facecolor':   '#1a1d2e',
    'axes.edgecolor':   '#3a3d5c',
    'axes.labelcolor':  '#c8cce8',
    'xtick.color':      '#8b8fad',
    'ytick.color':      '#8b8fad',
    'text.color':       '#c8cce8',
    'grid.color':       '#2a2d45',
    'grid.alpha':       0.5,
    'font.family':      'DejaVu Sans',
})
PALETTE = ['#4f9de8', '#e85f6b']


# 1. LOAD DATA

print("=" * 60)
print("LOADING DATASET")
print("=" * 60)

df = pd.read_csv('phase2_ml_final2/WithdrawlStudents.csv')
print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print("\nFirst 3 rows:")
print(df.head(3).to_string())
print("\nData Types:")
print(df.dtypes)


# 2. EXPLORATORY DATA ANALYSIS (EDA)

print("\n" + "=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)

#  Missing values 
print("\nMissing Values:")
missing = df.isnull().sum()
print(missing[missing > 0])

#  Class distribution ─
print("\nClass Distribution (withdrawl):")
counts = df['withdrawl'].value_counts()
print(f"  Not Withdrawn (0): {counts[0]:,} ({counts[0]/len(df)*100:.1f}%)")
print(f"  Withdrawn     (1): {counts[1]:,} ({counts[1]/len(df)*100:.1f}%)")

#  Basic stats 
print("\nNumerical Summary:")
print(df[['prev-attempts', 'studied-credits', 'avg-score']].describe().round(2).to_string())

# FIGURE 1: EDA Overview
fig = plt.figure(figsize=(18, 14))
fig.suptitle('Exploratory Data Analysis – Student Withdrawal Dataset',
             fontsize=16, fontweight='bold', color='#e8eaf6', y=0.98)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# Plot 1 – Class distribution
ax1 = fig.add_subplot(gs[0, 0])
bars = ax1.bar(['Not Withdrawn\n(0)', 'Withdrawn\n(1)'],
               [counts[0], counts[1]], color=PALETTE, width=0.5, edgecolor='none')
ax1.set_title('Class Distribution', fontweight='bold')
ax1.set_ylabel('Count')
for bar, val in zip(bars, [counts[0], counts[1]]):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
             f'{val:,}\n({val/len(df)*100:.1f}%)',
             ha='center', va='bottom', fontsize=9, color='#c8cce8')
ax1.set_ylim(0, counts[0] * 1.2)
ax1.grid(axis='y')

# Plot 2 – Gender vs withdrawal
ax2 = fig.add_subplot(gs[0, 1])
gender_ct = pd.crosstab(df['gender'], df['withdrawl'], normalize='index') * 100
gender_ct.plot(kind='bar', ax=ax2, color=PALETTE, width=0.6, edgecolor='none', legend=False)
ax2.set_title('Gender vs Withdrawal %', fontweight='bold')
ax2.set_ylabel('Percentage (%)')
ax2.set_xlabel('')
ax2.set_xticklabels(['Female', 'Male'], rotation=0)
ax2.legend(['Not Withdrawn', 'Withdrawn'], fontsize=8, loc='upper right')
ax2.grid(axis='y')

# Plot 3 – Avg score distribution
ax3 = fig.add_subplot(gs[0, 2])
for val, color, label in zip([0, 1], PALETTE, ['Not Withdrawn', 'Withdrawn']):
    ax3.hist(df[df['withdrawl'] == val]['avg-score'],
             bins=40, alpha=0.7, color=color, label=label, edgecolor='none')
ax3.set_title('Avg Score Distribution', fontweight='bold')
ax3.set_xlabel('Average Score')
ax3.set_ylabel('Frequency')
ax3.legend(fontsize=8)
ax3.grid(axis='y')

# Plot 4 – Age band vs withdrawal
ax4 = fig.add_subplot(gs[1, 0])
age_ct = pd.crosstab(df['age-band'], df['withdrawl'], normalize='index') * 100
age_order = ['0-35', '35-55', '55<=']
age_ct = age_ct.reindex([x for x in age_order if x in age_ct.index])
age_ct.plot(kind='bar', ax=ax4, color=PALETTE, width=0.6, edgecolor='none', legend=False)
ax4.set_title('Age Band vs Withdrawal %', fontweight='bold')
ax4.set_ylabel('Percentage (%)')
ax4.set_xlabel('')
ax4.set_xticklabels(ax4.get_xticklabels(), rotation=0)
ax4.legend(['Not Withdrawn', 'Withdrawn'], fontsize=8)
ax4.grid(axis='y')

# Plot 5 – Studied credits boxplot
ax5 = fig.add_subplot(gs[1, 1])
data_0 = df[df['withdrawl'] == 0]['studied-credits']
data_1 = df[df['withdrawl'] == 1]['studied-credits']
bp = ax5.boxplot([data_0, data_1], patch_artist=True,
                  medianprops=dict(color='white', linewidth=2))
for patch, color in zip(bp['boxes'], PALETTE):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
ax5.set_title('Studied Credits by Class', fontweight='bold')
ax5.set_xticklabels(['Not Withdrawn', 'Withdrawn'])
ax5.set_ylabel('Studied Credits')
ax5.grid(axis='y')

# Plot 6 – Prev attempts
ax6 = fig.add_subplot(gs[1, 2])
prev_ct = df.groupby(['prev-attempts', 'withdrawl']).size().unstack(fill_value=0)
prev_ct_pct = prev_ct.div(prev_ct.sum(axis=1), axis=0) * 100
prev_ct_pct.plot(kind='bar', ax=ax6, color=PALETTE, width=0.6, edgecolor='none', legend=False)
ax6.set_title('Prev Attempts vs Withdrawal %', fontweight='bold')
ax6.set_ylabel('Percentage (%)')
ax6.set_xlabel('Previous Attempts')
ax6.set_xticklabels(ax6.get_xticklabels(), rotation=0)
ax6.legend(['Not Withdrawn', 'Withdrawn'], fontsize=8)
ax6.grid(axis='y')

# Plot 7 – Missing values
ax7 = fig.add_subplot(gs[2, 0])
missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=True)
missing_pct = missing_pct[missing_pct > 0]
if len(missing_pct) > 0:
    bars = ax7.barh(missing_pct.index, missing_pct.values, color='#e85f6b', edgecolor='none')
    ax7.set_title('Missing Values (%)', fontweight='bold')
    ax7.set_xlabel('Missing %')
    for bar, val in zip(bars, missing_pct.values):
        ax7.text(val + 0.1, bar.get_y() + bar.get_height()/2,
                 f'{val:.1f}%', va='center', fontsize=9)
else:
    ax7.text(0.5, 0.5, 'No Missing Values', ha='center', va='center',
             transform=ax7.transAxes, fontsize=12)
    ax7.set_title('Missing Values', fontweight='bold')
ax7.grid(axis='x')

# Plot 8 – Correlation heatmap (numeric only)
ax8 = fig.add_subplot(gs[2, 1:])
numeric_df = df[['prev-attempts', 'studied-credits', 'avg-score', 'withdrawl']].copy()
corr = numeric_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, ax=ax8, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=0.5,
            cbar_kws={'shrink': 0.8},
            annot_kws={'size': 10})
ax8.set_title('Correlation Heatmap', fontweight='bold')

plt.savefig('phase2_ml_final2/figure1_eda.png', dpi=150, bbox_inches='tight',
            facecolor='#0f1117')
plt.close()
print("\n Saved: figure1_eda.png")


# 3. DATA PREPROCESSING

print("\n" + "=" * 60)
print("DATA PREPROCESSING")
print("=" * 60)

#  Step 1: Drop columns that cause leakage or are IDs 
DROP_COLS = ['id_student', 'code_module', 'code_presentation', 'final-result']
df_clean = df.drop(columns=DROP_COLS)
print(f"\nDropped columns: {DROP_COLS}")
print(f"  Reason: 'final-result' leaks the target; IDs have no predictive value.")

# Step 2: Handle missing values in 'imd-band' 
print(f"\nMissing in 'imd-band' before: {df_clean['imd-band'].isnull().sum()}")
df_clean['imd-band'] = df_clean['imd-band'].fillna('Unknown')
print(f"Missing in 'imd-band' after:  {df_clean['imd-band'].isnull().sum()}")
print("  Strategy: Filled with 'Unknown' category (preserves information)")

# Step 3: Encode categorical features
CATEGORICAL_COLS = ['gender', 'age-band', 'imd-band', 'high-education',
                    'region', 'disability']
le_dict = {}
for col in CATEGORICAL_COLS:
    le = LabelEncoder()
    df_clean[col] = le.fit_transform(df_clean[col].astype(str))
    le_dict[col] = le
    print(f"  Encoded '{col}': {list(le.classes_)}")

#  Step 4: Separate features and target 
X = df_clean.drop(columns=['withdrawl'])
y = df_clean['withdrawl']
print(f"\nFeatures shape: {X.shape}")
print(f"Target shape:   {y.shape}")
print(f"Feature names:  {list(X.columns)}")

#  Step 5: Train/Test Split (80/20, seed=42) 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {len(X_train):,} ({len(X_train)/len(X)*100:.0f}%)")
print(f"Test size:  {len(X_test):,}  ({len(X_test)/len(X)*100:.0f}%)")
print(f"Train withdrawal rate: {y_train.mean()*100:.1f}%")
print(f"Test  withdrawal rate: {y_test.mean()*100:.1f}%")

# Step 6: Scale numerical features 
# CRITICAL: fit ONLY on train, then transform both
NUMERICAL_COLS = ['prev-attempts', 'studied-credits', 'avg-score']
scaler = StandardScaler()
X_train[NUMERICAL_COLS] = scaler.fit_transform(X_train[NUMERICAL_COLS])
X_test[NUMERICAL_COLS]  = scaler.transform(X_test[NUMERICAL_COLS])   # transform only!
print(f"\nScaled numerical columns: {NUMERICAL_COLS}")
print("  Scaler fitted on TRAIN only → applied to TEST (no data leakage)")

# Save preprocessed data 
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

print("\n Saved: X_train.csv, X_test.csv, y_train.csv, y_test.csv")
print("\n" + "=" * 60)
print("STEP 1 COMPLETE — Run step2_models.py next")
print("=" * 60)
