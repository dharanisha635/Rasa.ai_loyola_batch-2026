# ============================================================
# COMPUTE ALL STATISTICS FROM 100K DATASET
# Run once — saves results for website
# ============================================================

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import statsmodels.api as sm

print("Computing statistics from 100k dataset...")

# Load
df = pd.read_csv("diabetes_prediction_dataset.csv")
df.drop_duplicates(inplace=True)

le_gender  = LabelEncoder()
le_smoking = LabelEncoder()
df['gender']          = le_gender.fit_transform(df['gender'])
df['smoking_history'] = le_smoking.fit_transform(df['smoking_history'])

FEATURES = ['gender','age','hypertension','heart_disease',
            'smoking_history','bmi','HbA1c_level','blood_glucose_level']
X = df[FEATURES]
y = df['diabetes']

# ── 1. Descriptive Statistics ────────────────────────────
desc = df[FEATURES + ['diabetes']].describe().round(3)

# ── 2. Pearson Correlation ────────────────────────────────
corr = df[FEATURES + ['diabetes']].corr().round(4)
corr_with_outcome = corr['diabetes'].drop('diabetes').to_dict()

# ── 3. T-Test (diabetic vs non-diabetic) ─────────────────
ttest_results = {}
for feat in FEATURES:
    g1 = df[df['diabetes']==1][feat]
    g2 = df[df['diabetes']==0][feat]
    t, p = stats.ttest_ind(g1, g2)
    ttest_results[feat] = {
        'mean_diabetic':     round(float(g1.mean()), 3),
        'mean_nondiabetic':  round(float(g2.mean()), 3),
        't_stat':            round(float(t), 4),
        'p_value':           round(float(p), 8),
        'significant':       bool(p < 0.05)
    }

# ── 4. Chi-Square ─────────────────────────────────────────
X_mm  = MinMaxScaler().fit_transform(X)
sel   = SelectKBest(chi2, k='all')
sel.fit(X_mm, y)
chi2_results = {}
for feat, score, pval in zip(FEATURES, sel.scores_, sel.pvalues_):
    chi2_results[feat] = {
        'chi2':        round(float(score), 2),
        'p_value':     round(float(pval), 8),
        'significant': bool(pval < 0.05)
    }

# ── 5. Population percentiles for each feature ───────────
percentiles = {}
for feat in FEATURES:
    vals = df[feat].values
    percentiles[feat] = {
        'p10':  round(float(np.percentile(vals, 10)), 2),
        'p25':  round(float(np.percentile(vals, 25)), 2),
        'p50':  round(float(np.percentile(vals, 50)), 2),
        'p75':  round(float(np.percentile(vals, 75)), 2),
        'p90':  round(float(np.percentile(vals, 90)), 2),
        'mean': round(float(np.mean(vals)), 2),
        'std':  round(float(np.std(vals)), 2),
        'diabetic_mean':     round(float(df[df['diabetes']==1][feat].mean()), 3),
        'nondiabetic_mean':  round(float(df[df['diabetes']==0][feat].mean()), 3),
    }

# ── 6. Logistic Regression statsmodels summary ───────────
sc  = StandardScaler()
Xsc = sc.fit_transform(X)
Xsm = sm.add_constant(Xsc[:5000])  # sample for speed
lr  = sm.Logit(y[:5000], Xsm)
res = lr.fit(disp=False)

lr_summary = {}
params     = res.params
pvalues    = res.pvalues
tvalues    = res.tvalues
conf       = res.conf_int()
feat_names = ['const'] + FEATURES

for i, name in enumerate(feat_names):
    lr_summary[name] = {
        'coef':    round(float(params.iloc[i]), 4),
        'pvalue':  round(float(pvalues.iloc[i]), 6),
        'tvalue':  round(float(tvalues.iloc[i]), 4),
        'ci_low':  round(float(conf.iloc[i, 0]), 4),
        'ci_high': round(float(conf.iloc[i, 1]), 4),
        'sig':     '***' if pvalues.iloc[i] < 0.001
                   else ('**' if pvalues.iloc[i] < 0.01
                   else ('*'  if pvalues.iloc[i] < 0.05 else ''))
    }

# ── 7. Class distribution ─────────────────────────────────
class_dist = {
    'diabetic':     int(y.sum()),
    'nondiabetic':  int((y==0).sum()),
    'total':        int(len(y)),
    'diabetes_pct': round(float(y.mean()*100), 2)
}

# ── 8. Save everything ────────────────────────────────────
output = {
    'corr_with_outcome': corr_with_outcome,
    'ttest_results':     ttest_results,
    'chi2_results':      chi2_results,
    'percentiles':       percentiles,
    'lr_summary':        lr_summary,
    'class_dist':        class_dist,
    'features':          FEATURES,
}

import os
os.makedirs('static', exist_ok=True)
with open('static/precomputed_stats.json', 'w') as f:
    json.dump(output, f, indent=2)

print("Done! Saved: static/precomputed_stats.json")
print(f"\nTop correlations with diabetes outcome:")
for k, v in sorted(corr_with_outcome.items(),
                   key=lambda x: abs(x[1]), reverse=True):
    print(f"  {k:30s}: r = {v:.4f}")

print(f"\nChi-square top features:")
for k, v in sorted(chi2_results.items(),
                   key=lambda x: x[1]['chi2'], reverse=True):
    print(f"  {k:30s}: chi2 = {v['chi2']:.1f}  p = {v['p_value']:.6f}")
