# ============================================================
# DIABETES CDSS — STATISTICAL ANALYSIS
# Run this file ONCE to generate all charts and stats
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.preprocessing import MinMaxScaler
import warnings
import os
import json
warnings.filterwarnings('ignore')

os.makedirs("static/charts", exist_ok=True)

# =========================
# LOAD & CLEAN DATA
# =========================
df = pd.read_csv("diabetes_prediction_dataset.csv")

smoker_rate_val = round((df['smoking_history'].isin(['current', 'ever', 'former'])).mean()*100, 1)
# Encode categorical columns
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
if df['gender'].dtype == object:
    df['gender'] = le.fit_transform(df['gender'])
if df['smoking_history'].dtype == object:
    df['smoking_history'] = le.fit_transform(df['smoking_history'])

print("=" * 60)
print("  DIABETES CDSS — STATISTICAL ANALYSIS REPORT")
print("=" * 60)
print(f"  Dataset shape: {df.shape}")
print(f"  Columns: {list(df.columns)}")

# Feature columns (numeric)
FEATURE_COLS = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level',
                'hypertension', 'heart_disease']
TARGET = 'diabetes'

# ============================================================
# STEP 1 — DESCRIPTIVE STATISTICS
# ============================================================
print("\n[ STEP 1 ] DESCRIPTIVE STATISTICS")
print("-" * 60)

desc = df.describe().round(3)
print(desc)

skew = df.skew(numeric_only=True).round(3)
print("\nSKEWNESS:")
print(skew)

# Distribution plots
plot_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level',
             'hypertension', 'heart_disease', 'smoking_history', 'gender', 'diabetes']
colors = ['#22D3EE','#0891B2','#7C3AED','#EA580C',
          '#16A34A','#DC2626','#D97706','#0369A1','#B91C1C']

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.patch.set_facecolor('#0A1628')
axes = axes.flatten()

for i, feat in enumerate(plot_cols):
    axes[i].set_facecolor('#0D2240')
    if feat == 'diabetes':
        axes[i].bar(['No Diabetes\n(0)', 'Diabetes\n(1)'],
                    df['diabetes'].value_counts().sort_index().values,
                    color=['#16A34A','#DC2626'], edgecolor='none', width=0.5)
    else:
        axes[i].hist(df[feat], bins=25, color=colors[i], edgecolor='none', alpha=0.85)
    axes[i].set_title(feat, color='white', fontsize=11, fontweight='bold', pad=8)
    axes[i].tick_params(colors='#90CAD8', labelsize=9)
    for spine in axes[i].spines.values():
        spine.set_color('#1E3A5A')

fig.suptitle('Feature Distributions — Diabetes Prediction Dataset (100k)',
             color='white', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('static/charts/distributions.png', dpi=120,
            bbox_inches='tight', facecolor='#0A1628')
plt.close()
print("  ✓ Distribution chart saved")

# ============================================================
# STEP 2 — OUTLIER DETECTION (IQR METHOD)
# ============================================================
print("\n[ STEP 2 ] OUTLIER DETECTION — IQR METHOD")
print("-" * 60)

outlier_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
outlier_results = {}

for col in outlier_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_out = ((df[col] < lower) | (df[col] > upper)).sum()
    outlier_results[col] = {
        'Q1': round(Q1,2), 'Q3': round(Q3,2),
        'IQR': round(IQR,2), 'Lower': round(lower,2),
        'Upper': round(upper,2), 'Outliers': int(n_out)
    }
    print(f"  {col:25s} | IQR={IQR:.2f} | Bounds=[{lower:.1f}, {upper:.1f}] | Outliers={n_out}")

fig, axes = plt.subplots(1, 4, figsize=(16, 5))
fig.patch.set_facecolor('#0A1628')
for i, col in enumerate(outlier_cols):
    axes[i].set_facecolor('#0D2240')
    axes[i].boxplot(df[col], patch_artist=True,
                    boxprops=dict(facecolor='#0891B2', color='#22D3EE'),
                    medianprops=dict(color='#EA580C', linewidth=2),
                    whiskerprops=dict(color='#22D3EE'),
                    capprops=dict(color='#22D3EE'),
                    flierprops=dict(marker='o', color='#DC2626',
                                    markerfacecolor='#DC2626', markersize=3))
    axes[i].set_title(col, color='white', fontsize=10, fontweight='bold')
    axes[i].tick_params(colors='#90CAD8', labelsize=9)
    for spine in axes[i].spines.values():
        spine.set_color('#1E3A5A')

fig.suptitle('Outlier Detection — Box Plots (IQR Method)',
             color='white', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('static/charts/boxplots.png', dpi=120,
            bbox_inches='tight', facecolor='#0A1628')
plt.close()
print("  ✓ Boxplot chart saved")

# ============================================================
# STEP 3 — CORRELATION ANALYSIS
# ============================================================
print("\n[ STEP 3 ] PEARSON CORRELATION ANALYSIS")
print("-" * 60)

corr = df[FEATURE_COLS + [TARGET]].corr().round(3)
print("Correlation with diabetes outcome:")
print(corr[TARGET].sort_values(ascending=False))

fig, ax = plt.subplots(figsize=(10, 8))
fig.patch.set_facecolor('#0A1628')
ax.set_facecolor('#0D2240')
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            ax=ax, linewidths=0.5, linecolor='#1E3A5A',
            annot_kws={'size': 9, 'color': 'white'},
            cbar_kws={'shrink': 0.8})
ax.set_title('Pearson Correlation Matrix', color='white',
             fontsize=14, fontweight='bold', pad=15)
ax.tick_params(colors='white', labelsize=9)
plt.tight_layout()
plt.savefig('static/charts/correlation.png', dpi=120,
            bbox_inches='tight', facecolor='#0A1628')
plt.close()
print("  ✓ Correlation heatmap saved")

# ============================================================
# STEP 4 — HYPOTHESIS TESTING (T-TEST)
# ============================================================
print("\n[ STEP 4 ] INDEPENDENT SAMPLES T-TEST")
print("-" * 60)
print("  H0: No significant difference between diabetic and non-diabetic groups")
print("  H1: Significant difference exists (p < 0.05)")
print()

ttest_results = {}
ttest_cols = ['HbA1c_level', 'blood_glucose_level', 'bmi', 'age', 'hypertension', 'heart_disease']

for col in ttest_cols:
    grp0 = df[df[TARGET] == 0][col]
    grp1 = df[df[TARGET] == 1][col]
    t, p = stats.ttest_ind(grp0, grp1)
    sig = "SIGNIFICANT ***" if p < 0.001 else ("SIGNIFICANT *" if p < 0.05 else "NOT significant")
    ttest_results[col] = {'t_stat': round(float(t), 4), 'p_value': round(float(p), 6)}
    print(f"  {col:25s} | t={t:8.3f} | p={p:.6f} | {sig}")

fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor('#0A1628')
ax.set_facecolor('#0D2240')
cols_list = list(ttest_results.keys())
t_vals = [abs(ttest_results[c]['t_stat']) for c in cols_list]
bar_colors = ['#16A34A' if ttest_results[c]['p_value'] < 0.05 else '#64748B' for c in cols_list]
bars = ax.barh(cols_list, t_vals, color=bar_colors, edgecolor='none', height=0.55)
ax.axvline(x=1.96, color='#DC2626', linestyle='--', linewidth=1.5,
           label='Critical value (t=1.96, α=0.05)')
ax.set_xlabel('|T-Statistic|', color='#90CAD8', fontsize=11)
ax.set_title('T-Test Results — Feature Significance\nGreen = Statistically Significant (p < 0.05)',
             color='white', fontsize=13, fontweight='bold')
ax.tick_params(colors='white', labelsize=10)
ax.legend(facecolor='#0D2240', labelcolor='white', fontsize=9)
for spine in ax.spines.values():
    spine.set_color('#1E3A5A')
for bar, val in zip(bars, t_vals):
    ax.text(val + 0.3, bar.get_y() + bar.get_height()/2,
            f'{val:.2f}', va='center', color='white', fontsize=9)
plt.tight_layout()
plt.savefig('static/charts/ttest.png', dpi=120,
            bbox_inches='tight', facecolor='#0A1628')
plt.close()
print("  ✓ T-test chart saved")

# ============================================================
# STEP 5 — CHI-SQUARE FEATURE SELECTION
# ============================================================
print("\n[ STEP 5 ] CHI-SQUARE FEATURE SELECTION TEST")
print("-" * 60)

X = df[FEATURE_COLS]
y = df[TARGET]
scaler_mm = MinMaxScaler()
X_scaled = scaler_mm.fit_transform(X)

selector = SelectKBest(chi2, k='all')
selector.fit(X_scaled, y)

chi2_df = pd.DataFrame({
    'Feature': FEATURE_COLS,
    'Chi2':    selector.scores_.round(3),
    'P_Value': selector.pvalues_.round(6)
}).sort_values('Chi2', ascending=False)

print(chi2_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor('#0A1628')
ax.set_facecolor('#0D2240')
colors_chi = ['#7C3AED' if p < 0.05 else '#64748B' for p in chi2_df['P_Value']]
bars = ax.bar(chi2_df['Feature'], chi2_df['Chi2'],
              color=colors_chi, edgecolor='none', width=0.6)
ax.set_xlabel('Features', color='#90CAD8', fontsize=11)
ax.set_ylabel('Chi-Square Score', color='#90CAD8', fontsize=11)
ax.set_title('Chi-Square Feature Selection\nPurple = Statistically Significant (p < 0.05)',
             color='white', fontsize=13, fontweight='bold')
ax.tick_params(colors='white', labelsize=9, rotation=15)
for spine in ax.spines.values():
    spine.set_color('#1E3A5A')
for bar, val in zip(bars, chi2_df['Chi2']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val:.1f}', ha='center', color='white', fontsize=9)
plt.tight_layout()
plt.savefig('static/charts/chi2.png', dpi=120,
            bbox_inches='tight', facecolor='#0A1628')
plt.close()
print("  ✓ Chi-square chart saved")

# ============================================================
# STEP 6 — CLASS IMBALANCE ANALYSIS
# ============================================================
print("\n[ STEP 6 ] CLASS DISTRIBUTION ANALYSIS")
print("-" * 60)

counts = df[TARGET].value_counts()
total  = len(df)
print(f"  Non-Diabetic (0): {counts[0]} ({counts[0]/total*100:.1f}%)")
print(f"  Diabetic     (1): {counts[1]} ({counts[1]/total*100:.1f}%)")
print(f"  Imbalance Ratio : {counts[0]/counts[1]:.2f}:1")

# ============================================================
# SUMMARY JSON
# ============================================================
summary = {
    "total_patients":   int(total),
    "diabetic":         int(counts[1]),
    "non_diabetic":     int(counts[0]),
    "imbalance_ratio":  round(counts[0]/counts[1], 2),
    "top_feature":      chi2_df.iloc[0]['Feature'],
    "top_chi2":         float(chi2_df.iloc[0]['Chi2']),
    "hba1c_corr":       float(corr[TARGET]['HbA1c_level']),
    "glucose_corr":     float(corr[TARGET]['blood_glucose_level']),
    "ttest_results":    ttest_results,
    "chi2_results":     chi2_df.set_index('Feature')[['Chi2','P_Value']].to_dict(),

    # Extra fields for statistics.html
    "diabetes_prevalence": round(counts[1]/total*100, 1),
    "avg_age":             round(df['age'].mean(), 1),
    "avg_bmi":             round(df['bmi'].mean(), 1),
    "mean_hba1c_diabetic":     round(df[df[TARGET]==1]['HbA1c_level'].mean(), 2),
    "mean_hba1c_nondiabetic":  round(df[df[TARGET]==0]['HbA1c_level'].mean(), 2),
    "mean_glucose_diabetic":   round(df[df[TARGET]==1]['blood_glucose_level'].mean(), 1),
    "mean_glucose_nondiabetic":round(df[df[TARGET]==0]['blood_glucose_level'].mean(), 1),
    "hypertension_rate":   round(df['hypertension'].mean()*100, 1),
    "heart_disease_rate":  round(df['heart_disease'].mean()*100, 1),
    "smoker_rate": smoker_rate_val,
    "obese_rate":          round((df['bmi'] >= 30).mean()*100, 1),
}

# ============================================================
# SAVE JSON
# ============================================================
summary = {
    "total_patients":          int(total),
    "diabetic":                int(counts[1]),
    "non_diabetic":            int(counts[0]),
    "diabetes_prevalence":     round(counts[1]/total*100, 1),
    "avg_age":                 round(df['age'].mean(), 1),
    "avg_bmi":                 round(df['bmi'].mean(), 1),
    "mean_hba1c_diabetic":     round(df[df[TARGET]==1]['HbA1c_level'].mean(), 2),
    "mean_hba1c_nondiabetic":  round(df[df[TARGET]==0]['HbA1c_level'].mean(), 2),
    "mean_glucose_diabetic":   round(df[df[TARGET]==1]['blood_glucose_level'].mean(), 1),
    "mean_glucose_nondiabetic":round(df[df[TARGET]==0]['blood_glucose_level'].mean(), 1),
    "hypertension_rate":       round(df['hypertension'].mean()*100, 1),
    "heart_disease_rate":      round(df['heart_disease'].mean()*100, 1),
    "smoker_rate":             smoker_rate_val,
    "obese_rate":              round((df['bmi'] >= 30).mean()*100, 1),
    "hba1c_corr":              float(corr[TARGET]['HbA1c_level']),
    "glucose_corr":            float(corr[TARGET]['blood_glucose_level']),
    "ttest_results":           ttest_results,
}

with open("static/stats_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("  ✓ static/stats_summary.json saved!")