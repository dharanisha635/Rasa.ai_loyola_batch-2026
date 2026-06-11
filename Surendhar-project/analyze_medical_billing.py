import mysql.connector
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
load_dotenv()
OUTPUT_DIR = "analysis_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CHART_STYLE = "seaborn-v0_8-darkgrid"
PALETTE     = "Set2"
plt.rcParams.update({
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
})

results_lines = []

def log(text=""):
    """Print to console and store for results.txt"""
    print(text)
    results_lines.append(str(text))

# ─────────────────────────────────────────────
# DB CONNECTION
# ─────────────────────────────────────────────
conn = mysql.connector.connect(
    host=os.getenv("DB_HOST", "localhost"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    database=os.getenv("DB_NAME")
)
df = pd.read_sql("SELECT * FROM encounters", conn)
conn.close()

# Derived columns
df["encounter_date"] = pd.to_datetime(df["encounter_date"])
df["dictation_date"]  = pd.to_datetime(df["dictation_date"])
df["turnaround_days"] = (df["dictation_date"] - df["encounter_date"]).dt.days.abs()
df["year_month"]      = df["encounter_date"].dt.to_period("M")

log("=" * 60)
log("  MEDICAL BILLING — STATISTICAL ANALYSIS REPORT")
log("=" * 60)
log(f"  Total records   : {len(df)}")
log(f"  Date range      : {df['encounter_date'].min().date()} → {df['encounter_date'].max().date()}")
log(f"  Departments     : {df['department'].nunique()}")
log(f"  Unique CPT codes: {df['cpt_code'].nunique()}")
log()

# ══════════════════════════════════════════════
# KPI 1 — CLAIM APPROVAL RATE
# ══════════════════════════════════════════════
log("─" * 60)
log("KPI 1 │ CLAIM APPROVAL RATE")
log("─" * 60)

approved = (df["claim_status"].str.lower() == "approved").sum()
total    = len(df)
rate     = approved / total

# 95% confidence interval (Wilson score)
z  = 1.96
lo = (rate + z**2/(2*total) - z*np.sqrt((rate*(1-rate)+z**2/(4*total))/total)) / (1+z**2/total)
hi = (rate + z**2/(2*total) + z*np.sqrt((rate*(1-rate)+z**2/(4*total))/total)) / (1+z**2/total)

log(f"  Approved claims : {approved}")
log(f"  Total claims    : {total}")
log(f"  Approval Rate   : {rate:.2%}")
log(f"  95% CI          : [{lo:.2%}, {hi:.2%}]")
log()

# Chart
fig, ax = plt.subplots(figsize=(6, 6))
wedges, texts, autotexts = ax.pie(
    [approved, total - approved],
    labels=["Approved", "Not Approved"],
    autopct="%1.1f%%",
    colors=["#2ecc71", "#e74c3c"],
    startangle=90,
    wedgeprops=dict(edgecolor="white", linewidth=2)
)
ax.set_title("Claim Approval Rate\n(with 95% CI shown in report)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_claim_approval_rate.png")
plt.close()

# ══════════════════════════════════════════════
# KPI 2 — CLAIM DENIAL RATE
# ══════════════════════════════════════════════
log("─" * 60)
log("KPI 2 │ CLAIM DENIAL RATE")
log("─" * 60)

status_freq = df["claim_status"].value_counts()
denial_rate = 1 - rate

log(f"  Claim Status Frequency Distribution:")
for status, count in status_freq.items():
    log(f"    {status:20s}: {count:4d}  ({count/total:.1%})")
log(f"  Overall Denial Rate: {denial_rate:.2%}")
log()

# Chart
fig, ax = plt.subplots(figsize=(8, 5))
colors = sns.color_palette(PALETTE, len(status_freq))
bars = ax.bar(status_freq.index, status_freq.values, color=colors, edgecolor="white", linewidth=1.5)
ax.bar_label(bars, fmt="%d", padding=4, fontsize=11)
ax.set_title("Claim Status Frequency Distribution")
ax.set_xlabel("Claim Status")
ax.set_ylabel("Number of Claims")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_claim_denial_rate.png")
plt.close()

# ══════════════════════════════════════════════
# KPI 3 — REVENUE BY DEPARTMENT
# ══════════════════════════════════════════════
log("─" * 60)
log("KPI 3 │ REVENUE BY DEPARTMENT")
log("─" * 60)

dept_rev = df.groupby("department")["billed_amount"].agg(["mean","median","std","sum"]).round(2)
dept_rev.columns = ["Mean","Median","Std Dev","Total"]
dept_rev = dept_rev.sort_values("Total", ascending=False)

log(dept_rev.to_string())
log()

# ANOVA across departments
dept_groups = [grp["billed_amount"].values for _, grp in df.groupby("department")]
f_stat, p_val = stats.f_oneway(*dept_groups)
log(f"  ANOVA — F-statistic: {f_stat:.4f}  |  p-value: {p_val:.4f}")
log(f"  Interpretation: {'Significant difference between departments (p<0.05)' if p_val < 0.05 else 'No significant difference'}")
log()

# Chart
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(dept_rev.index, dept_rev["Total"], color=sns.color_palette(PALETTE, len(dept_rev)), edgecolor="white")
ax.bar_label(bars, fmt="$%.0f", padding=4, fontsize=9)
ax.set_title("Total Revenue by Department")
ax.set_xlabel("Department")
ax.set_ylabel("Total Billed Amount ($)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_revenue_by_department.png")
plt.close()

# ══════════════════════════════════════════════
# KPI 4 — REVENUE BY CPT CODE
# ══════════════════════════════════════════════
log("─" * 60)
log("KPI 4 │ REVENUE BY CPT CODE")
log("─" * 60)

cpt_rev = df.groupby("cpt_code")["billed_amount"].agg(["mean","count","sum"]).round(2)
cpt_rev.columns = ["Mean Billing","Frequency","Total Revenue"]
cpt_rev = cpt_rev.sort_values("Total Revenue", ascending=False)

log("  Top 10 CPT Codes by Revenue:")
log(cpt_rev.head(10).to_string())
log()

cpt_groups = [grp["billed_amount"].values for _, grp in df.groupby("cpt_code") if len(grp) > 1]
if len(cpt_groups) >= 2:
    f2, p2 = stats.f_oneway(*cpt_groups)
    log(f"  ANOVA (CPT) — F: {f2:.4f}  |  p: {p2:.4f}")
log()

# Chart — top 15
top_cpt = cpt_rev.head(15)
fig, ax = plt.subplots(figsize=(12, 5))
sns.barplot(x=top_cpt.index, y=top_cpt["Total Revenue"], palette=PALETTE, ax=ax)
ax.set_title("Top 15 CPT Codes by Total Revenue")
ax.set_xlabel("CPT Code")
ax.set_ylabel("Total Revenue ($)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_revenue_by_cpt_code.png")
plt.close()

# ══════════════════════════════════════════════
# KPI 5 — AVERAGE BILLING AMOUNT
# ══════════════════════════════════════════════
log("─" * 60)
log("KPI 5 │ AVERAGE BILLING AMOUNT")
log("─" * 60)

ba = df["billed_amount"]
mode_val = ba.mode()[0]

log(f"  Mean   : ${ba.mean():,.2f}")
log(f"  Median : ${ba.median():,.2f}")
log(f"  Mode   : ${mode_val:,.2f}")
log(f"  Std Dev: ${ba.std():,.2f}")
log(f"  Min    : ${ba.min():,.2f}")
log(f"  Max    : ${ba.max():,.2f}")
log(f"  Range  : ${ba.max()-ba.min():,.2f}")
log()

# Chart
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].hist(ba, bins=30, color="#3498db", edgecolor="white", linewidth=0.8)
axes[0].axvline(ba.mean(),   color="red",    linestyle="--", label=f"Mean  ${ba.mean():,.0f}")
axes[0].axvline(ba.median(), color="orange", linestyle="--", label=f"Median ${ba.median():,.0f}")
axes[0].set_title("Billing Amount Distribution")
axes[0].set_xlabel("Billed Amount ($)")
axes[0].set_ylabel("Frequency")
axes[0].legend()

axes[1].boxplot(ba, vert=True, patch_artist=True,
                boxprops=dict(facecolor="#3498db", color="#2980b9"),
                medianprops=dict(color="orange", linewidth=2))
axes[1].set_title("Billing Amount Box Plot")
axes[1].set_ylabel("Billed Amount ($)")
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_average_billing_amount.png")
plt.close()

# ══════════════════════════════════════════════
# KPI 6 — BILLING TURNAROUND TIME
# ══════════════════════════════════════════════
log("─" * 60)
log("KPI 6 │ BILLING TURNAROUND TIME (Encounter → Dictation)")
log("─" * 60)

tat = df["turnaround_days"].dropna()
log(f"  Mean TAT   : {tat.mean():.2f} days")
log(f"  Median TAT : {tat.median():.2f} days")
log(f"  Std Dev    : {tat.std():.2f} days")
log(f"  Min / Max  : {tat.min()} / {tat.max()} days")
log(f"  % Same-Day : {(tat == 0).mean():.1%}")
log(f"  % ≤ 3 days : {(tat <= 3).mean():.1%}")
log()

# Survival-style: cumulative % processed by day
tat_sorted = np.sort(tat)
cumulative = np.arange(1, len(tat_sorted)+1) / len(tat_sorted)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].hist(tat, bins=20, color="#9b59b6", edgecolor="white")
axes[0].axvline(tat.mean(),   color="red",    linestyle="--", label=f"Mean {tat.mean():.1f}d")
axes[0].axvline(tat.median(), color="orange", linestyle="--", label=f"Median {tat.median():.1f}d")
axes[0].set_title("Turnaround Time Distribution")
axes[0].set_xlabel("Days")
axes[0].set_ylabel("Frequency")
axes[0].legend()

axes[1].plot(tat_sorted, cumulative, color="#9b59b6", linewidth=2)
axes[1].axhline(0.5, color="orange", linestyle="--", label="50th percentile")
axes[1].axhline(0.9, color="red",    linestyle="--", label="90th percentile")
axes[1].set_title("Cumulative % Processed (Survival Curve)")
axes[1].set_xlabel("Days")
axes[1].set_ylabel("Cumulative Proportion")
axes[1].legend()
axes[1].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_billing_turnaround_time.png")
plt.close()

# ══════════════════════════════════════════════
# KPI 7 — TOP DIAGNOSES (Pareto)
# ══════════════════════════════════════════════
log("─" * 60)
log("KPI 7 │ TOP DIAGNOSES (Pareto / 80-20 Rule)")
log("─" * 60)

diag_freq = df["diagnosis"].value_counts()
diag_cum  = diag_freq.cumsum() / diag_freq.sum()
pareto_80 = (diag_cum <= 0.80).sum()

log("  Top 10 Diagnoses:")
for i, (diag, cnt) in enumerate(diag_freq.head(10).items(), 1):
    log(f"    {i:2d}. {diag[:50]:50s}: {cnt}")
log(f"\n  Top {pareto_80} diagnoses account for 80% of all cases (Pareto).")
log()

# Chart — top 15
top15 = diag_freq.head(15)
cum15 = top15.cumsum() / diag_freq.sum() * 100

fig, ax1 = plt.subplots(figsize=(13, 6))
ax2 = ax1.twinx()
ax1.bar(range(len(top15)), top15.values, color="#e67e22", edgecolor="white")
ax2.plot(range(len(top15)), cum15.values, color="#c0392b", marker="o", linewidth=2)
ax2.axhline(80, color="grey", linestyle="--", label="80% line")
ax1.set_xticks(range(len(top15)))
ax1.set_xticklabels(top15.index, rotation=40, ha="right", fontsize=8)
ax1.set_title("Top 15 Diagnoses — Pareto Chart")
ax1.set_ylabel("Count")
ax2.set_ylabel("Cumulative %")
ax2.yaxis.set_major_formatter(mticker.PercentFormatter())
ax2.legend(loc="center right")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/07_top_diagnoses_pareto.png")
plt.close()

# ══════════════════════════════════════════════
# KPI 8 — CLAIM STATUS DISTRIBUTION (Chi-Square)
# ══════════════════════════════════════════════
log("─" * 60)
log("KPI 8 │ CLAIM STATUS DISTRIBUTION (Chi-Square Test)")
log("─" * 60)

observed   = df["claim_status"].value_counts().values
expected   = np.full(len(observed), total / len(observed))
chi2, p_chi = stats.chisquare(observed, f_exp=expected)

log(f"  Chi-Square Statistic : {chi2:.4f}")
log(f"  p-value              : {p_chi:.4f}")
log(f"  Interpretation       : {'Distribution is NOT uniform (p<0.05)' if p_chi < 0.05 else 'Distribution appears uniform'}")
log()

# Chart
fig, ax = plt.subplots(figsize=(8, 5))
colors = sns.color_palette(PALETTE, len(status_freq))
wedges, texts, autotexts = ax.pie(
    status_freq.values,
    labels=status_freq.index,
    autopct="%1.1f%%",
    colors=colors,
    startangle=140,
    wedgeprops=dict(edgecolor="white", linewidth=2)
)
ax.set_title("Claim Status Distribution")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/08_claim_status_distribution.png")
plt.close()

# ══════════════════════════════════════════════
# KPI 10 — MONTHLY REVENUE TREND
# ══════════════════════════════════════════════
log("─" * 60)
log("KPI 10 │ MONTHLY REVENUE TREND (Time Series + Forecast)")
log("─" * 60)

monthly = df.groupby("year_month")["billed_amount"].sum().reset_index()
monthly.columns = ["month","revenue"]
monthly["month_num"] = range(len(monthly))

# Moving average (3-month)
monthly["MA3"] = monthly["revenue"].rolling(3, min_periods=1).mean()

# Linear trend
slope, intercept, r, p_trend, _ = stats.linregress(monthly["month_num"], monthly["revenue"])
monthly["trend"] = intercept + slope * monthly["month_num"]

log("  Monthly Revenue:")
for _, row in monthly.iterrows():
    log(f"    {str(row['month']):10s}: ${row['revenue']:>10,.2f}  (MA3: ${row['MA3']:>10,.2f})")

log(f"\n  Linear Trend Slope : ${slope:+,.2f} per month")
log(f"  R-squared          : {r**2:.4f}")
log(f"  p-value (trend)    : {p_trend:.4f}")

# Forecast next 3 months
last_num = monthly["month_num"].max()
last_month = monthly["month"].max()
log("\n  3-Month Forecast:")
for i in range(1, 4):
    fcast_month = (last_month + i).strftime("%Y-%m")
    fcast_val   = intercept + slope * (last_num + i)
    log(f"    {fcast_month}: ${fcast_val:,.2f}")
log()

# Chart
fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(monthly["month"].astype(str), monthly["revenue"],
       color="#3498db", alpha=0.7, label="Monthly Revenue", edgecolor="white")
ax.plot(monthly["month"].astype(str), monthly["MA3"],
        color="orange", linewidth=2.5, marker="o", label="3-Month Moving Avg")
ax.plot(monthly["month"].astype(str), monthly["trend"],
        color="red", linewidth=2, linestyle="--", label="Linear Trend")
ax.set_title("Monthly Revenue Trend with Moving Average & Trend Line")
ax.set_xlabel("Month")
ax.set_ylabel("Total Revenue ($)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
plt.xticks(rotation=45, ha="right")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/10_monthly_revenue_trend.png")
plt.close()

# ─────────────────────────────────────────────
# WRITE RESULTS TO TXT
# ─────────────────────────────────────────────
txt_path = os.path.join(OUTPUT_DIR, "analysis_results.txt")
with open(txt_path, "w", encoding="utf-8") as f:
    f.write("\n".join(results_lines))

print()
print("=" * 60)
print(f"  ✅ DONE! All files saved to: {OUTPUT_DIR}/")
print("=" * 60)
print(f"  📄 analysis_results.txt   — all statistics")
print(f"  📊 01_claim_approval_rate.png")
print(f"  📊 02_claim_denial_rate.png")
print(f"  📊 03_revenue_by_department.png")
print(f"  📊 04_revenue_by_cpt_code.png")
print(f"  📊 05_average_billing_amount.png")
print(f"  📊 06_billing_turnaround_time.png")
print(f"  📊 07_top_diagnoses_pareto.png")
print(f"  📊 08_claim_status_distribution.png")
print(f"  📊 10_monthly_revenue_trend.png")
print("=" * 60)
