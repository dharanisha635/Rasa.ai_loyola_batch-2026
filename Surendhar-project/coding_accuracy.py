"""
=============================================================================
MEDICAL CODING ACCURACY ANALYSIS — Groq AI (CPT Code Reviewer)
Medical Billing Pipeline | Windows
=============================================================================
FIXES APPLIED:
  1. Model updated    : llama3-70b-8192 → llama-3.3-70b-versatile
  2. JSON parsing     : regex-based extraction (handles extra text from model)
  3. Error logging    : API errors now print instead of silently returning 0
  4. Metrics fix      : Removed circular 100% problem
                        y_true = match  (stored_cpt == suggested_cpt)
                        y_pred = is_correct (Groq's holistic verdict)
                        These two differ → gives real, meaningful metrics
=============================================================================
USAGE:
    python coding_accuracy.py
=============================================================================
"""

import os
import re
import time
import json
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import mysql.connector
from dotenv import load_dotenv
from groq import Groq
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             confusion_matrix, classification_report)

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

GROQ_KEYS = [
    os.getenv("GROQ_API_KEY_1"),
    os.getenv("GROQ_API_KEY_2"),
    os.getenv("GROQ_API_KEY_3"),
]

CHARTS_DIR  = "./charts"
RESULTS_DIR = "./results"
MODEL       = "llama-3.3-70b-versatile"
DELAY       = 2

# =============================================================================
# GROQ KEY ROTATOR
# =============================================================================
class KeyRotator:
    def __init__(self, keys):
        self.keys  = [k for k in keys if k]
        self.index = 0
        print(f"  Loaded {len(self.keys)} Groq API keys for rotation.\n")

    def get_client(self):
        key = self.keys[self.index % len(self.keys)]
        self.index += 1
        return Groq(api_key=key)

# =============================================================================
# LOAD DATA
# =============================================================================
def load_data() -> pd.DataFrame:
    conn = mysql.connector.connect(**DB_CONFIG)
    df = pd.read_sql("""
        SELECT encounter_id, diagnosis, icd10_code,
               cpt_code, department, patient_age, patient_gender
        FROM encounters
        WHERE icd10_code IS NOT NULL
          AND cpt_code   IS NOT NULL
          AND diagnosis  IS NOT NULL
    """, conn)
    conn.close()

    df["cpt_code"]   = df["cpt_code"].astype(str).str.strip()
    df["icd10_code"] = df["icd10_code"].astype(str).str.strip()
    df["diagnosis"]  = df["diagnosis"].astype(str).str.strip()

    print(f"  Loaded {len(df)} records for coding review.\n")
    return df

# =============================================================================
# ASK GROQ FOR SUGGESTED CPT CODE
# =============================================================================
def ask_groq(rotator: KeyRotator, diagnosis: str, icd10: str, cpt_stored: str) -> dict:
    prompt = f"""You are a certified medical coder. A hospital record has the following information:

Diagnosis    : {diagnosis}
ICD-10 Code  : {icd10}
Stored CPT   : {cpt_stored}

Your task:
1. Based on the diagnosis and ICD-10 code, suggest the most appropriate CPT code.
2. Give a confidence percentage (0-100) for your suggestion.
3. State whether the stored CPT code is correct or incorrect.

Respond ONLY in this exact JSON format, nothing else:
{{
  "suggested_cpt": "XXXXX",
  "confidence": 85,
  "is_correct": true,
  "reason": "brief one line reason"
}}"""

    try:
        client   = rotator.get_client()
        response = client.chat.completions.create(
            model       = MODEL,
            messages    = [{"role": "user", "content": prompt}],
            temperature = 0.1,
            max_tokens  = 200
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()

        json_match = re.search(r'\{.*?\}', raw, re.DOTALL)
        if not json_match:
            print(f"    [WARN] No JSON block found. Raw: {raw[:200]}")
            raise json.JSONDecodeError("No JSON block found", raw, 0)

        result = json.loads(json_match.group())

        return {
            "suggested_cpt": str(result.get("suggested_cpt", "")).strip(),
            "confidence":    float(result.get("confidence", 0)),
            "is_correct":    bool(result.get("is_correct", False)),
            "reason":        str(result.get("reason", ""))
        }

    except json.JSONDecodeError:
        return {
            "suggested_cpt": "UNKNOWN",
            "confidence":    0.0,
            "is_correct":    False,
            "reason":        "Could not parse response"
        }

    except Exception as e:
        print(f"    [ERROR] API call failed for '{diagnosis}': {e}")
        return {
            "suggested_cpt": "ERROR",
            "confidence":    0.0,
            "is_correct":    False,
            "reason":        str(e)
        }

# =============================================================================
# MAIN ANALYSIS
# =============================================================================
def run_coding_analysis(df: pd.DataFrame):
    rotator = KeyRotator(GROQ_KEYS)
    results = []
    total   = len(df)

    print("=" * 60)
    print("MEDICAL CODING ACCURACY ANALYSIS — Groq AI")
    print("=" * 60)
    print(f"  Model   : {MODEL}")
    print(f"  Records : {total}")
    print(f"  Delay   : {DELAY}s between requests\n")

    for i, row in df.iterrows():
        enc_id     = row["encounter_id"]
        diagnosis  = row["diagnosis"]
        icd10      = row["icd10_code"]
        cpt_stored = row["cpt_code"]

        current = len(results) + 1

        if current % 20 == 0 or current == 1:
            print(f"  [{current}/{total}] Processing {enc_id}...")

        result = ask_groq(rotator, diagnosis, icd10, cpt_stored)

        if current <= 3:
            print(f"    Stored CPT   : {cpt_stored}")
            print(f"    Suggested CPT: {result['suggested_cpt']}")
            print(f"    Confidence   : {result['confidence']}%")
            print(f"    Is Correct   : {result['is_correct']}")
            print(f"    Reason       : {result['reason']}\n")

        results.append({
            "encounter_id":  enc_id,
            "diagnosis":     diagnosis,
            "icd10_code":    icd10,
            "stored_cpt":    cpt_stored,
            "suggested_cpt": result["suggested_cpt"],
            "confidence":    result["confidence"],
            "is_correct":    result["is_correct"],
            "reason":        result["reason"],
            "match":         cpt_stored == result["suggested_cpt"]
        })

        time.sleep(DELAY)

    results_df = pd.DataFrame(results)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_df.to_csv(f"{RESULTS_DIR}/coding_accuracy_results.csv", index=False)
    print(f"\n  Results saved: {RESULTS_DIR}/coding_accuracy_results.csv")

    parse_failures = (results_df["suggested_cpt"].isin(["UNKNOWN", "ERROR"])).sum()
    if parse_failures > 0:
        print(f"\n  [WARN] {parse_failures} records had parse/API errors.")

    return results_df

# =============================================================================
# CALCULATE STATISTICS
# =============================================================================
def calculate_statistics(results_df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("CODING ACCURACY STATISTICS")
    print("=" * 60)

    total     = len(results_df)
    avg_conf  = results_df["confidence"].mean()
    high_conf = (results_df["confidence"] >= 80).sum()

    # ------------------------------------------------------------------
    # PRIMARY METRIC: Direct CPT match
    #   stored_cpt == suggested_cpt  →  1 (match / correct)
    #   stored_cpt != suggested_cpt  →  0 (mismatch / needs review)
    # ------------------------------------------------------------------
    match_count    = int(results_df["match"].sum())
    mismatch_count = total - match_count

    print(f"\n  OVERALL SUMMARY:")
    print(f"  Total records reviewed   : {total}")
    print(f"  CPT codes matched (correct)  : {match_count}  ({match_count/total:.1%})")
    print(f"  CPT codes mismatched (errors): {mismatch_count}  ({mismatch_count/total:.1%})")
    print(f"  Average AI confidence    : {avg_conf:.1f}%")
    print(f"  High confidence (>=80%)  : {high_conf}  ({high_conf/total:.1%})")

    # ------------------------------------------------------------------
    # FIX: Meaningful Accuracy / Precision / Recall / F1
    #
    #   y_true = match
    #            (objective signal: did stored CPT equal suggested CPT?)
    #
    #   y_pred = is_correct
    #            (Groq's holistic verdict based on medical reasoning)
    #
    #   These two are computed differently and CAN disagree.
    #   e.g. Groq may say is_correct=True but suggest a slightly
    #   different code variant → match=False, is_correct=True → FP
    #
    #   The metrics now answer:
    #   "How well does Groq's medical verdict align with
    #    direct CPT code comparison?"
    # ------------------------------------------------------------------
    y_true = results_df["match"].astype(int).values       # direct comparison
    y_pred = results_df["is_correct"].astype(int).values  # Groq's verdict

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)

    print(f"\n  MODEL METRICS:")
    print(f"  (y_true = direct CPT match | y_pred = Groq verdict)")
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  Precision : {prec*100:.2f}%")
    print(f"  Recall    : {rec*100:.2f}%")
    print(f"  F1 Score  : {f1*100:.2f}%")

    print(f"\n  CLASSIFICATION REPORT:")
    print(classification_report(y_true, y_pred,
                                target_names=["Mismatch", "Match"],
                                labels=[0, 1],
                                zero_division=0))

    # Diagnosis-wise accuracy (based on match)
    diag_acc = results_df.groupby("diagnosis")["match"].mean().sort_values()
    print(f"\n  TOP 10 MOST ERROR-PRONE DIAGNOSES (by CPT mismatch):")
    for diag, acc_d in diag_acc.head(10).items():
        print(f"    {diag[:50]:50s}: {acc_d:.1%} match rate")

    # Confidence distribution
    print(f"\n  CONFIDENCE DISTRIBUTION:")
    bins   = [0, 50, 70, 80, 90, 100]
    labels = ["0-50%", "51-70%", "71-80%", "81-90%", "91-100%"]
    results_df["conf_bucket"] = pd.cut(results_df["confidence"],
                                       bins=bins, labels=labels)
    conf_dist = results_df["conf_bucket"].value_counts().sort_index()
    for bucket, count in conf_dist.items():
        print(f"    {bucket}: {count} records")

    # Top mismatches
    errors_df = results_df[results_df["match"] == False].sort_values(
        "confidence", ascending=False)
    print(f"\n  TOP 15 CPT MISMATCHES (highest AI confidence):")
    print(errors_df[["encounter_id", "diagnosis", "stored_cpt",
                      "suggested_cpt", "confidence", "reason"]
                    ].head(15).to_string(index=False))

    return y_true, y_pred, results_df

# =============================================================================
# CHARTS
# =============================================================================
def plot_results(y_true, y_pred, results_df):
    os.makedirs(CHARTS_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    # --- Chart 1: Confusion Matrix ---
    # Rows (Actual)    = direct CPT match (match column)
    # Cols (Predicted) = Groq's verdict   (is_correct column)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
                xticklabels=["Groq: Incorrect", "Groq: Correct"],
                yticklabels=["Codes differ", "Codes match"], ax=axes[0])
    axes[0].set_title("Confusion Matrix\n(match vs Groq verdict)", fontweight="bold")
    axes[0].set_ylabel("Actual (CPT match)")
    axes[0].set_xlabel("Predicted (Groq verdict)")

    # --- Chart 2: CPT Match Rate pie ---
    match_count    = int(results_df["match"].sum())
    mismatch_count = len(results_df) - match_count

    if match_count > 0 and mismatch_count > 0:
        axes[1].pie([mismatch_count, match_count],
                    labels=["Mismatch", "Match"],
                    autopct="%1.1f%%",
                    colors=["#e74c3c", "#2ecc71"],
                    startangle=90,
                    wedgeprops=dict(edgecolor="white", linewidth=2))
    else:
        label = "All Match" if mismatch_count == 0 else "All Mismatch"
        axes[1].pie([1], labels=[label],
                    colors=["#2ecc71" if mismatch_count == 0 else "#e74c3c"],
                    wedgeprops=dict(edgecolor="white", linewidth=2))
    axes[1].set_title("CPT Code\nMatch vs Mismatch", fontweight="bold")

    # --- Chart 3: Confidence distribution ---
    axes[2].hist(results_df["confidence"], bins=20,
                 color="#9b59b6", edgecolor="white", alpha=0.85)
    axes[2].axvline(results_df["confidence"].mean(), color="red",
                    linestyle="--", linewidth=2,
                    label=f"Mean {results_df['confidence'].mean():.1f}%")
    axes[2].set_title("AI Confidence Distribution", fontweight="bold")
    axes[2].set_xlabel("Confidence (%)")
    axes[2].set_ylabel("Count")
    axes[2].legend()

    # --- Chart 4: Match rate by confidence bucket ---
    conf_match = results_df.groupby("conf_bucket",
                                    observed=True)["match"].mean() * 100
    if not conf_match.empty:
        conf_match.plot(kind="bar", ax=axes[3], color="#3498db",
                        edgecolor="white", rot=30)
        for p in axes[3].patches:
            axes[3].annotate(f"{p.get_height():.1f}%",
                             (p.get_x() + p.get_width() / 2,
                              p.get_height() + 1),
                             ha="center", fontsize=9)
    else:
        axes[3].text(0.5, 0.5, "No data", transform=axes[3].transAxes,
                     ha="center", va="center")
    axes[3].set_title("CPT Match Rate by\nConfidence Bucket", fontweight="bold")
    axes[3].set_xlabel("Confidence Range")
    axes[3].set_ylabel("Match Rate (%)")
    axes[3].set_ylim(0, 110)

    plt.suptitle("Medical Coding Accuracy Analysis — Groq AI",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(CHARTS_DIR, "coding_accuracy.png")
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

    if not any(GROQ_KEYS):
        print("ERROR: No Groq API keys found. Check your .env file.")
        return

    print("\n" + "=" * 60)
    print("MEDICAL CODING ACCURACY ANALYSIS")
    print("=" * 60 + "\n")

    df         = load_data()
    results_df = run_coding_analysis(df)

    y_true, y_pred, results_df = calculate_statistics(results_df)
    plot_results(y_true, y_pred, results_df)

    print("\n" + "=" * 60)
    print("  DONE! Files saved:")
    print(f"  📄 {RESULTS_DIR}/coding_accuracy_results.csv")
    print(f"  📊 {CHARTS_DIR}/coding_accuracy.png")
    print("=" * 60)

if __name__ == "__main__":
    main()
