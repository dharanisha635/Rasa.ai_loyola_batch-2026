"""
=============================================================================
RANDOM FOREST — Denial Prediction Model
Medical Billing Pipeline | Windows
=============================================================================
USAGE:
    python random_forest.py
=============================================================================
"""

import os
import warnings
warnings.filterwarnings("ignore")

import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import mysql.connector
from dotenv import load_dotenv

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             confusion_matrix, classification_report)
from sklearn.preprocessing import LabelEncoder
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

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
CHARTS_DIR     = "./charts"
MODELS_DIR     = "./models"
BEST_THRESHOLD = 0.35

# =============================================================================
# LOAD DATA
# =============================================================================
def load_data() -> pd.DataFrame:
    conn = mysql.connector.connect(**DB_CONFIG)
    df = pd.read_sql("""
        SELECT encounter_id, encounter_date, patient_age,
               patient_gender, department, diagnosis,
               icd10_code, cpt_code, billed_amount,
               claim_status, is_rushed
        FROM encounters
        WHERE billed_amount IS NOT NULL
          AND department IS NOT NULL
    """, conn)
    conn.close()

    df["encounter_date"] = pd.to_datetime(df["encounter_date"], errors="coerce")
    df["month_num"]      = df["encounter_date"].dt.month.fillna(0).astype(int)
    df["cpt_code"]       = df["cpt_code"].astype(str).str.strip().str.replace(r"[^0-9]", "", regex=True)
    df["diagnosis"]      = df["diagnosis"].astype(str).str.lower().str.strip()

    print(f"  Loaded {len(df)} records.")
    print(f"  Claim status: {df['claim_status'].value_counts().to_dict()}\n")
    return df


# =============================================================================
# PREPARE FEATURES
# =============================================================================
def prepare_features(df: pd.DataFrame):
    df_train = df[df["claim_status"].isin(["Paid", "Denied"])].copy()
    df_pend  = df[df["claim_status"] == "Pending"].copy()

    print(f"  Training set : {len(df_train)} records (Paid + Denied)")
    print(f"  Pending set  : {len(df_pend)} records (to predict)\n")

    le_dept   = LabelEncoder()
    le_cpt    = LabelEncoder()
    le_icd    = LabelEncoder()
    le_diag   = LabelEncoder()
    le_gender = LabelEncoder()

    all_data = pd.concat([df_train, df_pend], ignore_index=True)

    all_data["dept_enc"]   = le_dept.fit_transform(all_data["department"].fillna("Unknown"))
    all_data["cpt_enc"]    = le_cpt.fit_transform(all_data["cpt_code"].fillna("0"))
    all_data["icd_enc"]    = le_icd.fit_transform(all_data["icd10_code"].fillna("Unknown"))
    all_data["diag_enc"]   = le_diag.fit_transform(all_data["diagnosis"].fillna("Unknown"))
    all_data["gender_enc"] = le_gender.fit_transform(all_data["patient_gender"].fillna("Unknown"))

    features = [
        "billed_amount",
        "patient_age",
        "is_rushed",
        "dept_enc",
        "cpt_enc",
        "icd_enc",
        "diag_enc",
        "gender_enc",
        "month_num"
    ]

    train_idx = all_data.index[:len(df_train)]
    pend_idx  = all_data.index[len(df_train):]

    df_train_enc = all_data.loc[train_idx].copy()
    df_pend_enc  = all_data.loc[pend_idx].copy()

    df_train_enc["target"] = (df_train_enc["claim_status"] == "Denied").astype(int)
    df_train_enc = df_train_enc.dropna(subset=features)
    df_pend_enc  = df_pend_enc.dropna(subset=features)

    X       = df_train_enc[features]
    y       = df_train_enc["target"]
    X_pend  = df_pend_enc[features]
    enc_ids = df_pend_enc["encounter_id"].values

    # Save encoders and features for UI use
    encoders = {
        "le_dept":   le_dept,
        "le_cpt":    le_cpt,
        "le_icd":    le_icd,
        "le_diag":   le_diag,
        "le_gender": le_gender
    }

    return X, y, X_pend, enc_ids, features, encoders


# =============================================================================
# RANDOM FOREST MODEL
# =============================================================================
def run_random_forest(X, y, X_pend, enc_ids, features, encoders):
    print("=" * 60)
    print("RANDOM FOREST — DENIAL PREDICTION")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n  Class distribution (train):")
    print(f"    Paid   : {(y_train == 0).sum()}")
    print(f"    Denied : {(y_train == 1).sum()}")
    print(f"    (SMOTE will balance these inside each CV fold)\n")

    # Pipeline: SMOTE inside each CV fold
    pipeline = ImbPipeline([
        ("smote", SMOTE(random_state=42)),
        ("model", RandomForestClassifier(class_weight="balanced", random_state=42))
    ])

    param_dist = {
        "model__n_estimators":      [50, 100, 200, 300],
        "model__max_depth":         [3, 5, 7, 10, None],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf":  [1, 2, 4],
        "model__max_features":      ["sqrt", "log2"],
        "model__bootstrap":         [True, False]
    }

    # Tune with RandomizedSearchCV
    print("  Tuning hyperparameters (RandomizedSearchCV + SMOTE pipeline)...")
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=20,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    search.fit(X_train, y_train)

    print(f"\n  Best parameters found:")
    for k, v in search.best_params_.items():
        clean_key = k.replace("model__", "")
        print(f"    {clean_key:22s}: {v}")
    print(f"  Best CV F1 Score : {search.best_score_*100:.2f}%")

    # Evaluate on test set
    y_prob = search.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= BEST_THRESHOLD).astype(int)

    print(f"\n  MODEL PERFORMANCE (threshold = {BEST_THRESHOLD}):")
    print(f"  Accuracy  : {accuracy_score(y_test, y_pred)*100:.2f}%")
    print(f"  Precision : {precision_score(y_test, y_pred, zero_division=0)*100:.2f}%")
    print(f"  Recall    : {recall_score(y_test, y_pred, zero_division=0)*100:.2f}%")
    print(f"  F1 Score  : {f1_score(y_test, y_pred, zero_division=0)*100:.2f}%")

    # Cross validation
    cv_scores = cross_val_score(search.best_estimator_, X, y, cv=5, scoring="f1")
    print(f"\n  5-FOLD CROSS VALIDATION (F1):")
    print(f"  Scores : {[round(s*100, 2) for s in cv_scores]}")
    print(f"  Mean   : {cv_scores.mean()*100:.2f}%")
    print(f"  Std    : {cv_scores.std()*100:.2f}%")

    # Classification report
    print(f"\n  CLASSIFICATION REPORT (threshold = {BEST_THRESHOLD}):")
    print(classification_report(y_test, y_pred,
                                target_names=["Paid", "Denied"],
                                zero_division=0))

    # Feature importance
    best_rf = search.best_estimator_.named_steps["model"]
    imp_df  = pd.Series(best_rf.feature_importances_,
                        index=features).sort_values(ascending=False)
    print(f"  FEATURE IMPORTANCE:")
    print(imp_df.round(4).to_string())

    # Predict Pending claims
    pend_probs = search.predict_proba(X_pend)[:, 1]
    pend_preds = (pend_probs >= BEST_THRESHOLD).astype(int)

    pending_results = pd.DataFrame({
        "encounter_id":       enc_ids,
        "predicted_status":   ["Denied" if p == 1 else "Paid" for p in pend_preds],
        "denial_probability": (pend_probs * 100).round(2)
    }).sort_values("denial_probability", ascending=False)

    print(f"\n  PENDING CLAIM PREDICTIONS (Top 15 likely denials):")
    print(pending_results.head(15).to_string(index=False))

    os.makedirs("./results", exist_ok=True)
    pending_results.to_csv("./results/pending_predictions_rf.csv", index=False)
    print(f"\n  Full predictions saved: ./results/pending_predictions_rf.csv")

    # ── SAVE MODEL & ENCODERS ─────────────────────────────────
    os.makedirs(MODELS_DIR, exist_ok=True)

    joblib.dump(search,        f"{MODELS_DIR}/random_forest.pkl")
    joblib.dump(encoders,      f"{MODELS_DIR}/encoders.pkl")
    joblib.dump(features,      f"{MODELS_DIR}/features.pkl")
    joblib.dump(BEST_THRESHOLD,f"{MODELS_DIR}/threshold.pkl")

    print(f"\n  Model saved    : {MODELS_DIR}/random_forest.pkl")
    print(f"  Encoders saved : {MODELS_DIR}/encoders.pkl")
    print(f"  Features saved : {MODELS_DIR}/features.pkl")
    print(f"  Threshold saved: {MODELS_DIR}/threshold.pkl")

    plot_results(y_test, y_pred, imp_df, pending_results, search)

    return search, pending_results


# =============================================================================
# CHARTS
# =============================================================================
def plot_results(y_test, y_pred, imp_df, pending_results, search):
    os.makedirs(CHARTS_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Paid", "Denied"],
                yticklabels=["Paid", "Denied"], ax=axes[0])
    axes[0].set_title(f"Confusion Matrix\n(threshold = {BEST_THRESHOLD})", fontweight="bold")
    axes[0].set_ylabel("Actual")
    axes[0].set_xlabel("Predicted")

    # Feature Importance
    imp_df.sort_values().plot(kind="barh", ax=axes[1],
                              color="#3498db", edgecolor="white")
    axes[1].set_title("Feature Importance\nRandom Forest", fontweight="bold")
    axes[1].set_xlabel("Importance Score")

    # Denial probability distribution
    denied_probs = pending_results[pending_results["predicted_status"] == "Denied"]["denial_probability"]
    paid_probs   = pending_results[pending_results["predicted_status"] == "Paid"]["denial_probability"]
    axes[2].hist(paid_probs,   bins=15, color="#2ecc71", edgecolor="white",
                 alpha=0.75, label="Predicted Paid")
    axes[2].hist(denied_probs, bins=15, color="#e74c3c", edgecolor="white",
                 alpha=0.75, label="Predicted Denied")
    axes[2].axvline(BEST_THRESHOLD * 100, color="black", linestyle="--",
                    linewidth=2, label=f"{int(BEST_THRESHOLD*100)}% threshold")
    axes[2].set_title("Denial Probability Distribution\nPending Claims", fontweight="bold")
    axes[2].set_xlabel("Denial Probability (%)")
    axes[2].set_ylabel("Count")
    axes[2].legend()

    # Top 10 tuning combos
    cv_results = pd.DataFrame(search.cv_results_)
    top10 = cv_results.nlargest(10, "mean_test_score")
    axes[3].barh(range(len(top10)), top10["mean_test_score"] * 100,
                 color="#2ecc71", edgecolor="white")
    axes[3].set_yticks(range(len(top10)))
    axes[3].set_yticklabels([f"Combo {i+1}" for i in range(len(top10))], fontsize=8)
    axes[3].set_title("Top 10 Param Combos\n(RandomizedSearchCV F1)", fontweight="bold")
    axes[3].set_xlabel("Mean CV F1 Score (%)")

    plt.suptitle(f"Random Forest — Denial Prediction  |  Threshold = {BEST_THRESHOLD}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(CHARTS_DIR, "random_forest.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Chart saved: {path}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    if not DB_CONFIG["password"]:
        print("ERROR: DB_PASSWORD not found. Check your .env file.")
        return

    print("\n" + "=" * 60)
    print("RANDOM FOREST — DENIAL PREDICTION")
    print(f"  Fixed threshold : {BEST_THRESHOLD}")
    print("=" * 60 + "\n")

    df = load_data()
    X, y, X_pend, enc_ids, features, encoders = prepare_features(df)
    run_random_forest(X, y, X_pend, enc_ids, features, encoders)


if __name__ == "__main__":
    main()