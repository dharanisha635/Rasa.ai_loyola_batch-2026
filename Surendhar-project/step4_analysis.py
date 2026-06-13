"""
=============================================================================
STEP 4 — Complete Statistical Analysis (9 KPIs)
Medical Billing Pipeline | Windows
=============================================================================
INSTALL (run once):
    pip install pandas scipy matplotlib seaborn scikit-learn imbalanced-learn
    pip install mysql-connector-python python-dotenv

USAGE:
    python step4_analysis.py

OUTPUT:
    charts/  → all chart images
    analysis_report.txt → full statistical results
=============================================================================
"""

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency
import mysql.connector
from dotenv import load_dotenv

# ML imports
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, classification_report)
from sklearn.preprocessing import LabelEncoder

# =============================================================================
# CONFIG
# =============================================================================
load_dotenv()
DB_CONFIG = {
    "host":     os.getenv("DB_HOST"),
    "port":     3306,
    "user":     os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME")
}

CHARTS_DIR  = "./charts"
REPORT_PATH = "./analysis_report.txt"

DEPT_COLORS = {
    "Orthopedics":      "#2E86AB",
    "General Medicine": "#A23B72",
    "Dermatology":      "#F18F01",
    "Cardiology":       "#C73E1D",
    "Pediatrics":       "#3B1F2B",
    "Neurology":        "#44BBA4"
}

report_lines = []

def log(text=""):
    print(text)
    report_lines.append(text)

# =============================================================================
# LOAD & CLEAN DATA
# =============================================================================
def load_data() -> pd.DataFrame:
    conn = mysql.connector.connect(**DB_CONFIG)
    df = pd.read_sql("""
        SELECT encounter_id, encounter_date, dictation_date,
               patient_age, patient_gender, department,
               diagnosis, icd10_code, cpt_code,
               billed_amount, claim_status, is_rushed
        FROM encounters
        WHERE billed_amount IS NOT NULL
          AND department IS NOT NULL
    """, conn)
    conn.close()

    df["encounter_date"] = pd.to_datetime(df["encounter_date"], errors="coerce")
    df["month"]          = df["encounter_date"].dt.to_period("M").astype(str)

    # --- CLEANING ---
    # Normalize CPT codes (remove dashes, spaces)
    df["cpt_code"] = df["cpt_code"].astype(str).str.strip().str.replace(r"[^0-9]", "", regex=True)

    # Normalize diagnosis (lowercase, strip)
    df["diagnosis"] = df["diagnosis"].astype(str).str.lower().str.strip()

    # Normalize department
    df["department"] = df["department"].astype(str).str.strip()

    log(f"  Loaded {len(df)} records from database.")
    log(f"  Departments : {df['department'].value_counts().to_dict()}")
    log(f"  Claim status: {df['claim_status'].value_counts().to_dict()}")
    log()

    return df


# =============================================================================
# HELPER — save chart
# =============================================================================
def save_chart(filename: str):
    os.makedirs(CHARTS_DIR, exist_ok=True)
    path = os.path.join(CHARTS_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"  Chart saved: {path}")


# =============================================================================
# KPI 1 — Claim Approval Rate
# =============================================================================
def kpi_claim_approval_rate(df: pd.DataFrame):
    log("=" * 60)
    log("KPI 1: CLAIM APPROVAL RATE")
    log("=" * 60)

    total   = len(df)
    paid    = (df["claim_status"] == "Paid").sum()
    denied  = (df["claim_status"] == "Denied").sum()
    pending = (df["claim_status"] == "Pending").sum()

    approval_rate = paid / total * 100
    denial_rate   = denied / total * 100
    pending_rate  = pending / total * 100

    # Confidence interval for approval rate (Wilson method)
    p    = paid / total
    z    = 1.96
    ci_low  = (p + z**2/(2*total) - z * np.sqrt(p*(1-p)/total + z**2/(4*total**2))) / (1 + z**2/total)
    ci_high = (p + z**2/(2*total) + z * np.sqrt(p*(1-p)/total + z**2/(4*total**2))) / (1 + z**2/total)

    log(f"  Total Records  : {total}")
    log(f"  Paid           : {paid}  ({approval_rate:.1f}%)")
    log(f"  Denied         : {denied}  ({denial_rate:.1f}%)")
    log(f"  Pending        : {pending}  ({pending_rate:.1f}%)")
    log(f"  95% CI (Approval Rate): [{ci_low*100:.1f}%, {ci_high*100:.1f}%]")
    log()

    # Chart
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Pie chart
    counts = df["claim_status"].value_counts()
    colors = ["#2ECC71", "#E74C3C", "#F39C12"][:len(counts)]
    axes[0].pie(counts.values, labels=counts.index, autopct="%1.1f%%",
                colors=colors, startangle=140,
                wedgeprops=dict(edgecolor="white", linewidth=2))
    axes[0].set_title("Claim Status Distribution", fontsize=13, fontweight="bold")

    # Bar chart by department
    dept_approval = df[df["claim_status"] == "Paid"].groupby("department").size() / \
                    df.groupby("department").size() * 100
    dept_colors   = [DEPT_COLORS.get(d, "#999") for d in dept_approval.index]
    axes[1].bar(dept_approval.index, dept_approval.values, color=dept_colors, edgecolor="white")
    axes[1].set_title("Approval Rate by Department (%)", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Approval Rate (%)")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].spines[["top", "right"]].set_visible(False)

    plt.suptitle("KPI 1 & 2: Claim Approval & Denial Rates", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_chart("01_claim_approval_denial.png")


# =============================================================================
# KPI 3 — Revenue by Department (ANOVA)
# =============================================================================
def kpi_revenue_by_department(df: pd.DataFrame):
    log("=" * 60)
    log("KPI 3: REVENUE BY DEPARTMENT")
    log("=" * 60)

    dept_stats = df.groupby("department")["billed_amount"].agg(
        ["mean", "median", "std", "count"]
    ).round(2)
    dept_stats.columns = ["Mean ($)", "Median ($)", "Std Dev", "Count"]
    log(dept_stats.to_string())
    log()

    # ANOVA test
    groups   = [g["billed_amount"].values for _, g in df.groupby("department")]
    f_stat, p_value = stats.f_oneway(*groups)
    log(f"  One-Way ANOVA:")
    log(f"  F-statistic : {f_stat:.4f}")
    log(f"  p-value     : {p_value:.4f}")
    log(f"  Result      : {'SIGNIFICANT difference across departments (p < 0.05)' if p_value < 0.05 else 'No significant difference'}")
    log()

    # Chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart — mean billed amount
    means   = df.groupby("department")["billed_amount"].mean().sort_values(ascending=False)
    colors  = [DEPT_COLORS.get(d, "#999") for d in means.index]
    bars    = axes[0].bar(means.index, means.values, color=colors, edgecolor="white")
    for bar, val in zip(bars, means.values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                     f"${val:.0f}", ha="center", fontsize=9, fontweight="bold")
    axes[0].set_title("Mean Billed Amount by Department", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Mean Billed Amount ($)")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].spines[["top", "right"]].set_visible(False)

    # Box plot
    dept_order = df.groupby("department")["billed_amount"].mean().sort_values(ascending=False).index.tolist()
    df_plot    = df[df["department"].isin(dept_order)]
    sns.boxplot(data=df_plot, x="department", y="billed_amount",
                order=dept_order, ax=axes[1],
                palette=DEPT_COLORS)
    axes[1].set_title("Billed Amount Distribution by Department", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Billed Amount ($)")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=30)

    plt.suptitle(f"KPI 3: Revenue by Department | ANOVA p={p_value:.4f}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_chart("02_revenue_by_department.png")


# =============================================================================
# KPI 4 — Revenue by CPT Code
# =============================================================================
def kpi_revenue_by_cpt(df: pd.DataFrame):
    log("=" * 60)
    log("KPI 4: REVENUE BY CPT CODE")
    log("=" * 60)

    cpt_stats = df.groupby("cpt_code")["billed_amount"].agg(
        ["mean", "count", "sum"]
    ).round(2).sort_values("sum", ascending=False)
    cpt_stats.columns = ["Mean ($)", "Count", "Total Revenue ($)"]
    log(cpt_stats.to_string())
    log()

    # Chart
    fig, ax = plt.subplots(figsize=(10, 5))
    cpt_avg  = df.groupby("cpt_code")["billed_amount"].mean().sort_values(ascending=True)
    bars     = ax.barh(cpt_avg.index.astype(str), cpt_avg.values, color="#2E86AB", edgecolor="white")
    for bar, val in zip(bars, cpt_avg.values):
        ax.text(val + 2, bar.get_y() + bar.get_height()/2,
                f"${val:.0f}", va="center", fontsize=9, fontweight="bold")
    ax.set_title("KPI 4: Average Billed Amount by CPT Code", fontsize=13, fontweight="bold")
    ax.set_xlabel("Average Billed Amount ($)")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    save_chart("03_revenue_by_cpt.png")


# =============================================================================
# KPI 5 — Average Billing Amount
# =============================================================================
def kpi_average_billing(df: pd.DataFrame):
    log("=" * 60)
    log("KPI 5: AVERAGE BILLING AMOUNT")
    log("=" * 60)

    amt  = df["billed_amount"]
    mode = amt.mode().iloc[0] if not amt.mode().empty else "N/A"

    log(f"  Mean   : ${amt.mean():.2f}")
    log(f"  Median : ${amt.median():.2f}")
    log(f"  Mode   : ${mode:.2f}")
    log(f"  Std Dev: ${amt.std():.2f}")
    log(f"  Min    : ${amt.min():.2f}")
    log(f"  Max    : ${amt.max():.2f}")
    log()

    # Chart
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(amt, bins=30, color="#2E86AB", edgecolor="white", alpha=0.85)
    ax.axvline(amt.mean(),   color="#E74C3C", linestyle="--", linewidth=2, label=f"Mean: ${amt.mean():.0f}")
    ax.axvline(amt.median(), color="#F39C12", linestyle="--", linewidth=2, label=f"Median: ${amt.median():.0f}")
    ax.set_title("KPI 5: Billed Amount Distribution", fontsize=13, fontweight="bold")
    ax.set_xlabel("Billed Amount ($)")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    save_chart("04_billing_amount_distribution.png")


# =============================================================================
# KPI 7 — Top Diagnoses (Pareto Analysis)
# =============================================================================
def kpi_top_diagnoses(df: pd.DataFrame):
    log("=" * 60)
    log("KPI 7: TOP DIAGNOSES (PARETO ANALYSIS)")
    log("=" * 60)

    diag_counts = df["diagnosis"].value_counts().head(10)
    cumulative  = diag_counts.cumsum() / diag_counts.sum() * 100

    log(diag_counts.to_string())
    log()

    # Find how many diagnoses cover 80%
    pareto_cutoff = (cumulative <= 80).sum() + 1
    log(f"  Pareto (80/20 Rule): Top {pareto_cutoff} diagnoses cover 80% of all cases")
    log()

    # Chart
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()

    bars = ax1.bar(range(len(diag_counts)), diag_counts.values,
                   color="#2E86AB", edgecolor="white", alpha=0.85)
    ax2.plot(range(len(diag_counts)), cumulative.values,
             color="#E74C3C", marker="o", linewidth=2, label="Cumulative %")
    ax2.axhline(80, color="#F39C12", linestyle="--", linewidth=1.5, label="80% line")

    ax1.set_xticks(range(len(diag_counts)))
    ax1.set_xticklabels(diag_counts.index, rotation=35, ha="right", fontsize=9)
    ax1.set_ylabel("Frequency")
    ax2.set_ylabel("Cumulative %")
    ax2.set_ylim(0, 110)
    ax2.legend(loc="center right")

    plt.title("KPI 7: Top Diagnoses — Pareto Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_chart("05_top_diagnoses_pareto.png")


# =============================================================================
# KPI 8 — Claim Status Distribution (Chi-Square)
# =============================================================================
def kpi_claim_status_distribution(df: pd.DataFrame):
    log("=" * 60)
    log("KPI 8: CLAIM STATUS DISTRIBUTION (CHI-SQUARE)")
    log("=" * 60)

    # Chi-square: is claim status independent of department?
    contingency = pd.crosstab(df["department"], df["claim_status"])
    chi2, p, dof, expected = chi2_contingency(contingency)

    log(f"  Contingency Table:")
    log(contingency.to_string())
    log()
    log(f"  Chi-Square Test (Department vs Claim Status):")
    log(f"  Chi2 statistic : {chi2:.4f}")
    log(f"  p-value        : {p:.4f}")
    log(f"  Degrees of freedom: {dof}")
    log(f"  Result: {'SIGNIFICANT association (p < 0.05)' if p < 0.05 else 'No significant association'}")
    log()

    # Chart — stacked bar
    fig, ax = plt.subplots(figsize=(12, 5))
    contingency_pct = contingency.div(contingency.sum(axis=1), axis=0) * 100
    contingency_pct.plot(kind="bar", stacked=True, ax=ax,
                         color=["#2ECC71", "#E74C3C", "#F39C12"],
                         edgecolor="white")
    ax.set_title(f"KPI 8: Claim Status by Department | Chi2 p={p:.4f}",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Percentage (%)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="Claim Status", bbox_to_anchor=(1.05, 1))
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    save_chart("06_claim_status_by_department.png")


# =============================================================================
# KPI 9 — Denial Prediction (ML Models)
# =============================================================================
def kpi_denial_prediction(df: pd.DataFrame):
    log("=" * 60)
    log("KPI 9: DENIAL PREDICTION MODEL")
    log("=" * 60)

    df_ml = df[df["claim_status"].isin(["Paid", "Denied"])].copy()
    df_ml["target"] = (df_ml["claim_status"] == "Denied").astype(int)

    log(f"  Class distribution: {df_ml['target'].value_counts().to_dict()}")
    log(f"  Note: Using class_weight='balanced' to handle imbalance")
    log()

    # Features
    le_dept   = LabelEncoder()
    le_cpt    = LabelEncoder()
    df_ml["dept_enc"] = le_dept.fit_transform(df_ml["department"].fillna("Unknown"))
    df_ml["cpt_enc"]  = le_cpt.fit_transform(df_ml["cpt_code"].fillna("0"))

    features = ["billed_amount", "dept_enc", "cpt_enc", "is_rushed", "patient_age"]
    df_ml    = df_ml.dropna(subset=features)

    X = df_ml[features]
    y = df_ml["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000),
        "Decision Tree":       DecisionTreeClassifier(class_weight="balanced", max_depth=5, random_state=42),
        "Random Forest":       RandomForestClassifier(class_weight="balanced", n_estimators=100, random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            "Accuracy":  round(accuracy_score(y_test, y_pred) * 100, 2),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0) * 100, 2),
            "Recall":    round(recall_score(y_test, y_pred, zero_division=0) * 100, 2),
            "F1 Score":  round(f1_score(y_test, y_pred, zero_division=0) * 100, 2)
        }
        log(f"  {name}:")
        log(f"    Accuracy  : {results[name]['Accuracy']}%")
        log(f"    Precision : {results[name]['Precision']}%")
        log(f"    Recall    : {results[name]['Recall']}%")
        log(f"    F1 Score  : {results[name]['F1 Score']}%")
        log()

    # Feature importance from Random Forest
    rf_model   = models["Random Forest"]
    importance = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=True)
    log(f"  Feature Importance (Random Forest):")
    log(importance.round(4).to_string())
    log()

    # Chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Model comparison bar chart
    metrics_df = pd.DataFrame(results).T
    metrics_df.plot(kind="bar", ax=axes[0], edgecolor="white",
                    color=["#2E86AB", "#A23B72", "#F18F01", "#44BBA4"])
    axes[0].set_title("Model Comparison", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Score (%)")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].set_ylim(0, 110)
    axes[0].legend(loc="lower right")
    axes[0].spines[["top", "right"]].set_visible(False)

    # Feature importance
    importance.plot(kind="barh", ax=axes[1], color="#2E86AB", edgecolor="white")
    axes[1].set_title("Feature Importance (Random Forest)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Importance Score")
    axes[1].spines[["top", "right"]].set_visible(False)

    plt.suptitle("KPI 9: Denial Prediction Model", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_chart("07_denial_prediction.png")


# =============================================================================
# KPI 10 — Monthly Revenue Trend
# =============================================================================
def kpi_monthly_revenue_trend(df: pd.DataFrame):
    log("=" * 60)
    log("KPI 10: MONTHLY REVENUE TREND")
    log("=" * 60)

    monthly = df.groupby("month")["billed_amount"].agg(["sum", "mean", "count"]).reset_index()
    monthly.columns = ["Month", "Total Revenue", "Avg Revenue", "Count"]

    log(monthly.to_string(index=False))
    log()
    log("  Note: Data is concentrated in a few months — trend should be interpreted with caution.")
    log()

    # Moving average
    monthly["Moving Avg (3M)"] = monthly["Total Revenue"].rolling(window=3, min_periods=1).mean()

    # Chart
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Total revenue
    axes[0].bar(monthly["Month"], monthly["Total Revenue"],
                color="#2E86AB", edgecolor="white", alpha=0.8, label="Total Revenue")
    axes[0].plot(monthly["Month"], monthly["Moving Avg (3M)"],
                 color="#E74C3C", marker="o", linewidth=2, label="3M Moving Avg")
    axes[0].set_title("Monthly Total Revenue", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Total Revenue ($)")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].legend()
    axes[0].spines[["top", "right"]].set_visible(False)

    # Record count
    axes[1].bar(monthly["Month"], monthly["Count"],
                color="#44BBA4", edgecolor="white", alpha=0.8)
    axes[1].set_title("Monthly Record Count", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Number of Encounters")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].spines[["top", "right"]].set_visible(False)

    plt.suptitle("KPI 10: Monthly Revenue Trend", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_chart("08_monthly_revenue_trend.png")


# =============================================================================
# MAIN
# =============================================================================
def main():
    if not DB_CONFIG["password"]:
        print("ERROR: DB_PASSWORD not found. Check your .env file.")
        return

    os.makedirs(CHARTS_DIR, exist_ok=True)

    log("=" * 60)
    log("MEDICAL BILLING PIPELINE — STATISTICAL ANALYSIS REPORT")
    log("=" * 60)
    log()

    df = load_data()

    if df.empty:
        print("No data found. Run step3_database.py first.")
        return

    # Run all KPIs
    kpi_claim_approval_rate(df)
    kpi_revenue_by_department(df)
    kpi_revenue_by_cpt(df)
    kpi_average_billing(df)
    kpi_top_diagnoses(df)
    kpi_claim_status_distribution(df)
    kpi_denial_prediction(df)
    kpi_monthly_revenue_trend(df)

    # Save report
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    log()
    log("=" * 60)
    log(f"  All done!")
    log(f"  Charts : {CHARTS_DIR}/")
    log(f"  Report : {REPORT_PATH}")
    log("=" * 60)


if __name__ == "__main__":
    main()
