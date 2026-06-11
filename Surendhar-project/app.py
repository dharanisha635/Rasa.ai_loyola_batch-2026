"""
=============================================================================
MedBill AI — Flask Web Application  (Full Build)
Medical Billing Denial Prediction System
=============================================================================
Pages:
  /             Upload & Predict  (audio → transcription → extraction → model)
  /dashboard    Analytics Dashboard (claim stats, denial patterns)
  /statistics   Statistical Analysis (ANOVA, Chi-square, Model Comparison)
=============================================================================
"""

import os, json, time, re, warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

from flask import Flask, render_template, request, flash, redirect, url_for
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)

# =============================================================================
# PATHS
# =============================================================================
MODELS_DIR = "./models"
DATA_PATH  = "./data/medical_billing_backup.xlsx"
UPLOAD_DIR = "./uploads"
CODING_CSV = "./results/coding_accuracy_results.csv"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# =============================================================================
# GROQ KEY ROTATION
# =============================================================================
API_KEYS = []
for i in range(1, 6):
    k = os.getenv(f"GROQ_API_KEY_{i}")
    if k: API_KEYS.append(k)
if not API_KEYS:
    k = os.getenv("GROQ_API_KEY")
    if k: API_KEYS.append(k)

_key_idx = 0

def _groq_client():
    return Groq(api_key=API_KEYS[_key_idx])

def _rotate_key():
    global _key_idx
    _key_idx = (_key_idx + 1) % max(len(API_KEYS), 1)

# =============================================================================
# LOAD MODEL ARTIFACTS
# =============================================================================
def _load_artifacts():
    try:
        model     = joblib.load(f"{MODELS_DIR}/random_forest.pkl")
        encoders  = joblib.load(f"{MODELS_DIR}/encoders.pkl")
        features  = joblib.load(f"{MODELS_DIR}/features.pkl")
        threshold = joblib.load(f"{MODELS_DIR}/threshold.pkl")
        print(f"  [OK] Model loaded | threshold = {threshold}")
        return model, encoders, features, threshold
    except Exception as e:
        print(f"  [WARNING] Could not load model: {e}")
        return None, None, None, None

MODEL, ENCODERS, FEATURES, THRESHOLD = _load_artifacts()

# =============================================================================
# LOAD HISTORICAL DATA
# =============================================================================
def _load_historical():
    try:
        df = pd.read_excel(DATA_PATH)
        df["diagnosis"]  = df["diagnosis"].astype(str).str.lower().str.strip()
        df["cpt_code"]   = df["cpt_code"].astype(str).str.strip().str.replace(r"[^0-9]", "", regex=True)
        df["department"] = df["department"].astype(str).str.strip()
        print(f"  [OK] Historical data loaded | {len(df)} records")
        return df
    except Exception as e:
        print(f"  [WARNING] Could not load historical data: {e}")
        return None

HIST_DF = _load_historical()

# =============================================================================
# WHISPER TRANSCRIPTION
# =============================================================================
_whisper_model = None

def _get_whisper():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        _whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")
    return _whisper_model

def transcribe_audio(audio_path: str) -> str:
    wm = _get_whisper()
    segments, _ = wm.transcribe(audio_path, language="en", beam_size=5,
                                 vad_filter=True,
                                 vad_parameters=dict(min_silence_duration_ms=500))
    return " ".join(seg.text.strip() for seg in segments)

# =============================================================================
# GROQ EXTRACTION
# =============================================================================
_EXTRACTION_PROMPT = """
You are a medical billing data extraction engine.
Read the clinical dictation transcript and extract ALL fields.
Return ONLY a valid JSON object — no explanation, no markdown, no extra text.

RULES:
1. encounter_id   : use filename provided
2. encounter_date : YYYY-MM-DD if mentioned, else null
3. patient_age    : integer (look for "year-old"), else null
4. patient_gender : "M" or "F", else null
5. department     : infer from diagnosis:
                    ankle sprain/fracture/strapping → "Orthopedics"
                    bronchitis/strep throat → "General Medicine"
                    abscess/laceration/skin → "Dermatology"
                    chest pain/hypertension → "Cardiology"
                    otitis/fever/ear infection → "Pediatrics"
                    migraine/vertigo → "Neurology"
6. diagnosis      : plain English, lowercase
7. icd10_code     : code after "ICD-10" or "ICD10", else null
8. cpt_code       : numeric code after "CPT", else null
9. billed_amount  : float after "billed amount" or "$", else null
10. is_rushed     : true if "rushed" mentioned, else false
11. notes         : brief clinical summary

Filename: {encounter_id}
Transcript:
{transcript}
"""

_DEPT_FALLBACKS = {
    "ankle sprain":"Orthopedics","radius fracture":"Orthopedics",
    "fracture":"Orthopedics","sprain":"Orthopedics","strapping":"Orthopedics",
    "bronchitis":"General Medicine","strep":"General Medicine","pharyngitis":"General Medicine",
    "abscess":"Dermatology","laceration":"Dermatology","skin":"Dermatology","wound":"Dermatology",
    "chest pain":"Cardiology","hypertension":"Cardiology",
    "otitis":"Pediatrics","fever":"Pediatrics","ear infection":"Pediatrics",
    "migraine":"Neurology","vertigo":"Neurology",
}

def _apply_fallbacks(data, transcript):
    if not data.get("department"):
        text = (data.get("diagnosis") or "" + transcript).lower()
        for kw, dept in _DEPT_FALLBACKS.items():
            if kw in text:
                data["department"] = dept
                break
    if not data.get("claim_status"):
        data["claim_status"] = "Pending"
    if data.get("is_rushed") is None:
        data["is_rushed"] = "rushed" in transcript.lower()
    return data

def extract_fields(transcript, encounter_id="UPLOAD"):
    if not API_KEYS:
        return _skeleton(encounter_id, transcript)
    prompt = _EXTRACTION_PROMPT.format(encounter_id=encounter_id, transcript=transcript)
    for attempt in range(len(API_KEYS) * 2):
        try:
            resp = _groq_client().chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role":"user","content":prompt}]
            )
            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = "\n".join(l for l in raw.split("\n") if not l.strip().startswith("```"))
            data = json.loads(raw.strip())
            data["encounter_id"] = encounter_id
            return _apply_fallbacks(data, transcript)
        except json.JSONDecodeError:
            time.sleep(2)
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                _rotate_key(); time.sleep(5)
            else:
                time.sleep(2)
    return _skeleton(encounter_id, transcript)

def _skeleton(encounter_id, transcript):
    s = {k: None for k in ["encounter_id","encounter_date","patient_age","patient_gender",
                             "department","diagnosis","icd10_code","cpt_code",
                             "billed_amount","claim_status","is_rushed","notes"]}
    s["encounter_id"] = encounter_id
    return _apply_fallbacks(s, transcript)

# =============================================================================
# CPT CODE REVIEW  (coding_accuracy.py logic — single claim)
# =============================================================================
_CPT_REVIEW_PROMPT = """You are a certified medical coder. A hospital record has the following information:

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

def review_cpt_code(diagnosis, icd10, cpt_stored):
    """Call Groq to validate the CPT code for a single extracted claim."""
    if not API_KEYS or not diagnosis or not icd10 or not cpt_stored:
        return None
    prompt = _CPT_REVIEW_PROMPT.format(
        diagnosis=diagnosis, icd10=icd10, cpt_stored=cpt_stored
    )
    try:
        resp = _groq_client().chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role":"user","content":prompt}],
            temperature=0.1, max_tokens=200
        )
        raw = resp.choices[0].message.content.strip()
        raw = raw.replace("```json","").replace("```","").strip()
        m = re.search(r'\{.*?\}', raw, re.DOTALL)
        if m:
            r = json.loads(m.group())
            return {
                "suggested_cpt": str(r.get("suggested_cpt","")).strip(),
                "confidence":    float(r.get("confidence", 0)),
                "is_correct":    bool(r.get("is_correct", False)),
                "reason":        str(r.get("reason",""))
            }
    except Exception as e:
        print(f"  [WARNING] CPT review failed: {e}")
    return None

# =============================================================================
# AI DENIAL EXPLANATION  (Groq plain-English reason)
# =============================================================================
_EXPLANATION_PROMPT = """You are a medical billing expert. A claim has been analysed by a machine learning model and given a denial probability score. Based on the claim details and risk factors below, write a clear 2-3 sentence plain English explanation of WHY this claim is at risk of being denied. Be specific — mention the actual values (department, billed amount, diagnosis, etc). Do not use bullet points. Do not say "the model". Write as if you are a senior billing consultant advising a billing staff member.

Claim Details:
- Patient: {age} year old {gender}
- Department: {department}
- Diagnosis: {diagnosis}
- ICD-10: {icd10}
- CPT Code: {cpt}
- Billed Amount: ${amount}
- Rushed Dictation: {rushed}
- Denial Probability: {prob}%

Risk Factors Identified:
{risk_factors}

Write ONLY the explanation paragraph, nothing else."""

def generate_denial_explanation(extracted: dict, denial_prob: float, risk_factors: list) -> str:
    """Call Groq to generate a plain English explanation of why this claim is at risk."""
    if not API_KEYS or denial_prob < 35:
        return None
    try:
        risk_text = "\n".join(f"- {f['title']}: {f['detail']}" for f in risk_factors if f['type'] != 'safe')
        if not risk_text:
            return None

        prompt = _EXPLANATION_PROMPT.format(
            age        = extracted.get("patient_age") or "Unknown",
            gender     = "Male" if extracted.get("patient_gender") == "M" else "Female" if extracted.get("patient_gender") == "F" else "Unknown",
            department = extracted.get("department") or "Unknown",
            diagnosis  = (extracted.get("diagnosis") or "Unknown").title(),
            icd10      = extracted.get("icd10_code") or "Not provided",
            cpt        = extracted.get("cpt_code") or "Not provided",
            amount     = extracted.get("billed_amount") or "Unknown",
            rushed     = "Yes" if extracted.get("is_rushed") else "No",
            prob       = denial_prob,
            risk_factors = risk_text
        )
        resp = _groq_client().chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role":"user","content":prompt}],
            temperature=0.3,
            max_tokens=200
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"  [WARNING] Explanation generation failed: {e}")
        return None


def _safe_encode(encoder, value):
    try:
        val = str(value or "unknown").strip().lower()
        classes_lower = [str(c).lower() for c in encoder.classes_]
        if val in classes_lower:
            idx = classes_lower.index(val)
            return int(encoder.transform([encoder.classes_[idx]])[0])
        return 0
    except:
        return 0

def prepare_feature_vector(data):
    enc = ENCODERS
    month_num = datetime.now().month
    if data.get("encounter_date"):
        try: month_num = pd.to_datetime(data["encounter_date"]).month
        except: pass
    cpt  = "".join(c for c in str(data.get("cpt_code") or "0") if c.isdigit())
    diag = str(data.get("diagnosis") or "unknown").lower().strip()
    vector = [
        float(data.get("billed_amount") or 0),
        float(data.get("patient_age")   or 0),
        int(bool(data.get("is_rushed"))),
        _safe_encode(enc["le_dept"],   data.get("department")     or "Unknown"),
        _safe_encode(enc["le_cpt"],    cpt),
        _safe_encode(enc["le_icd"],    data.get("icd10_code")     or "Unknown"),
        _safe_encode(enc["le_diag"],   diag),
        _safe_encode(enc["le_gender"], data.get("patient_gender") or "Unknown"),
        month_num,
    ]
    return np.array(vector, dtype=float).reshape(1, -1)

# =============================================================================
# RISK FACTORS
# =============================================================================
def get_risk_factors(data, denial_prob, coding_review=None):
    factors = []
    if HIST_DF is None:
        return factors
    train = HIST_DF[HIST_DF["claim_status"].isin(["Paid","Denied"])].copy()
    overall_rate = (train["claim_status"] == "Denied").mean()

    if data.get("is_rushed"):
        rushed_rate = (train[train["is_rushed"]==1]["claim_status"]=="Denied").mean()
        factors.append({"type":"warning","icon":"⚡","title":"Rushed Dictation Detected",
            "detail":f"Rushed claims are denied {rushed_rate*100:.0f}% of the time vs {overall_rate*100:.0f}% overall."})

    dept = data.get("department")
    if dept:
        dept_df   = train[train["department"]==dept]
        if len(dept_df) >= 3:
            dept_rate = (dept_df["claim_status"]=="Denied").mean()
            if dept_rate > overall_rate * 1.2:
                factors.append({"type":"warning","icon":"🏥","title":f"High-Risk Department: {dept}",
                    "detail":f"{dept} has a {dept_rate*100:.0f}% denial rate (above the {overall_rate*100:.0f}% average)."})
            else:
                factors.append({"type":"safe","icon":"✅","title":f"Low-Risk Department: {dept}",
                    "detail":f"{dept} has a {dept_rate*100:.0f}% denial rate — below average."})

    diag = str(data.get("diagnosis") or "").lower().strip()
    if diag:
        diag_df = train[train["diagnosis"].str.lower().str.strip()==diag]
        if len(diag_df) >= 3:
            diag_rate = (diag_df["claim_status"]=="Denied").mean()
            if diag_rate > overall_rate * 1.2:
                factors.append({"type":"warning","icon":"🩺","title":f"High-Risk Diagnosis: {diag.title()}",
                    "detail":f"Claims with this diagnosis are denied {diag_rate*100:.0f}% of the time historically."})

    amount = data.get("billed_amount")
    if amount:
        amount = float(amount)
        ref  = train[train["department"]==dept]["billed_amount"] if dept else train["billed_amount"]
        p75  = ref.quantile(0.75)
        if amount > p75:
            factors.append({"type":"warning","icon":"💰","title":"Above-Average Billed Amount",
                "detail":f"${amount:,.0f} exceeds the 75th percentile (${p75:,.0f}). High amounts face more scrutiny."})

    # CPT code mismatch from coding review
    if coding_review and not coding_review.get("is_correct"):
        factors.append({"type":"danger","icon":"🔢","title":"CPT Code Mismatch",
            "detail":f"Stored CPT {data.get('cpt_code')} — Suggested: {coding_review.get('suggested_cpt')}. Wrong codes are a leading denial cause."})

    missing = []
    if not data.get("icd10_code"):    missing.append("ICD-10 code")
    if not data.get("cpt_code"):      missing.append("CPT code")
    if not data.get("patient_age"):   missing.append("patient age")
    if not data.get("billed_amount"): missing.append("billed amount")
    if missing:
        factors.append({"type":"danger","icon":"⚠️","title":"Missing Required Fields",
            "detail":f"Could not extract: {', '.join(missing)}. Incomplete claims are frequently denied."})

    if not factors:
        factors.append({"type":"safe","icon":"✅","title":"No Major Risk Factors",
            "detail":"This claim appears complete and within normal billing parameters."})
    return factors

# =============================================================================
# RECOMMENDATION
# =============================================================================
def get_recommendation(denial_prob, risk_factors):
    if denial_prob > 0.45:
        return {"level":"danger","action":"Hold — Do Not Submit",
            "steps":["This claim is highly risky. Do not submit without a full review.",
                     "Verify all ICD-10 and CPT codes match the diagnosis exactly.",
                     "Check that all required modifiers are correctly applied.",
                     "Escalate to senior billing staff immediately."]}
    elif denial_prob > 0.40:
        return {"level":"warning","action":"Needs Serious Review Before Submitting",
            "steps":["Review all documentation carefully before submission.",
                     "Double-check procedure codes and diagnosis codes.",
                     "Confirm payer-specific requirements are fully met.",
                     "A mandatory secondary review is strongly recommended."]}
    elif denial_prob >= 0.35:
        return {"level":"warning","action":"Quick Review Recommended",
            "steps":["Claim is likely to be paid but a quick check is advised.",
                     "Verify that all required fields are completely filled.",
                     "Confirm codes are accurate before submitting.",
                     "Proceed after a brief review."]}
    else:
        return {"level":"success","action":"Safe to Submit",
            "steps":["Claim appears complete and low risk.",
                     "Proceed with the standard submission process.",
                     "Keep documentation on file for any future appeal."]}

# =============================================================================
# SIMILAR CLAIMS
# =============================================================================
def get_similar_claims(data, n=5):
    if HIST_DF is None: return []
    df   = HIST_DF[HIST_DF["claim_status"].isin(["Paid","Denied"])].copy()
    dept = data.get("department")
    diag = str(data.get("diagnosis") or "").lower().strip()
    pool = df[df["department"]==dept] if dept else df
    if diag:
        narrowed = pool[pool["diagnosis"].str.lower().str.strip()==diag]
        if len(narrowed) >= 3: pool = narrowed
    if pool.empty: pool = df
    sample = pool.sample(min(n, len(pool)), random_state=42)
    return [{"encounter_id":row["encounter_id"],"department":row["department"],
             "diagnosis":str(row["diagnosis"]).title(),
             "billed_amount":float(row["billed_amount"]) if pd.notna(row["billed_amount"]) else 0,
             "is_rushed":bool(row["is_rushed"]),"status":row["claim_status"]}
            for _, row in sample.iterrows()]

# =============================================================================
# DASHBOARD STATISTICS
# =============================================================================
def compute_dashboard_stats():
    if HIST_DF is None: return {}
    df    = HIST_DF.copy()
    train = df[df["claim_status"].isin(["Paid","Denied"])].copy()
    status_counts = df["claim_status"].value_counts().to_dict()
    dept_stats = {}
    for dept, grp in train.groupby("department"):
        dept_stats[dept] = {"denial_rate":round((grp["claim_status"]=="Denied").mean()*100,1),"total":len(grp)}
    diag_counts = train.groupby("diagnosis").size()
    top_diags   = diag_counts[diag_counts >= 5].index
    diag_stats  = {}
    for diag in top_diags:
        grp = train[train["diagnosis"]==diag]
        diag_stats[str(diag).title()] = round((grp["claim_status"]=="Denied").mean()*100,1)
    rushed_rate     = (train[train["is_rushed"]==1]["claim_status"]=="Denied").mean()*100
    not_rushed_rate = (train[train["is_rushed"]==0]["claim_status"]=="Denied").mean()*100
    paid_avg   = train[train["claim_status"]=="Paid"]["billed_amount"].mean()
    denied_avg = train[train["claim_status"]=="Denied"]["billed_amount"].mean()
    total_pending = df[df["claim_status"]=="Pending"]["billed_amount"].sum()
    bins   = [0,18,35,50,65,120]
    labels = ["0–18","19–35","36–50","51–65","65+"]
    train["age_group"] = pd.cut(train["patient_age"],bins=bins,labels=labels)
    age_denial = {str(g):round((gdf["claim_status"]=="Denied").mean()*100,1)
                  for g, gdf in train.groupby("age_group", observed=True)}
    return {"status_counts":status_counts,"dept_stats":dept_stats,"diag_stats":diag_stats,
            "rushed_rate":round(float(rushed_rate),1),"not_rushed_rate":round(float(not_rushed_rate),1),
            "paid_avg":round(float(paid_avg),2),"denied_avg":round(float(denied_avg),2),
            "total_pending":round(float(total_pending),2),"total_records":len(df),
            "overall_denial_rate":round(float((train["claim_status"]=="Denied").mean()*100),1),
            "age_denial":age_denial}

# =============================================================================
# STATISTICAL ANALYSIS  (ANOVA + Chi-square + Model Comparison + KPIs)
# =============================================================================
_STATS_CACHE = None

def compute_statistical_analysis():
    global _STATS_CACHE
    if _STATS_CACHE is not None:
        return _STATS_CACHE

    if HIST_DF is None:
        return {}

    from scipy import stats as scipy_stats
    from scipy.stats import chi2_contingency
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.preprocessing import LabelEncoder

    df    = HIST_DF.copy()
    train = df[df["claim_status"].isin(["Paid","Denied"])].copy()
    result = {}

    # ── KPI 1: Approval Rate + Wilson 95% CI ─────────────────────────────
    total   = len(df)
    paid    = int((df["claim_status"]=="Paid").sum())
    denied  = int((df["claim_status"]=="Denied").sum())
    pending = int((df["claim_status"]=="Pending").sum())
    p  = paid / total
    z  = 1.96
    ci_low  = (p + z**2/(2*total) - z*np.sqrt(p*(1-p)/total + z**2/(4*total**2))) / (1+z**2/total)
    ci_high = (p + z**2/(2*total) + z*np.sqrt(p*(1-p)/total + z**2/(4*total**2))) / (1+z**2/total)
    result["kpi1"] = {
        "total":total,"paid":paid,"denied":denied,"pending":pending,
        "approval_rate":round(p*100,1),"denial_rate":round(denied/total*100,1),
        "pending_rate":round(pending/total*100,1),
        "ci_low":round(ci_low*100,1),"ci_high":round(ci_high*100,1)
    }

    # Approval rate by department
    dept_approval = {}
    for dept, grp in df.groupby("department"):
        dept_approval[dept] = round((grp["claim_status"]=="Paid").sum()/len(grp)*100,1)
    result["dept_approval"] = dept_approval

    # ── KPI 5: Billing Amount Summary ────────────────────────────────────
    amt = df["billed_amount"].dropna()
    result["kpi5"] = {
        "mean":round(float(amt.mean()),2),"median":round(float(amt.median()),2),
        "std":round(float(amt.std()),2),"min":round(float(amt.min()),2),
        "max":round(float(amt.max()),2),
        "mode":round(float(amt.mode().iloc[0]),2) if not amt.mode().empty else 0
    }

    # ── KPI 7: Top Diagnoses (Pareto) ────────────────────────────────────
    diag_counts = df["diagnosis"].value_counts().head(10)
    cumulative  = (diag_counts.cumsum() / diag_counts.sum() * 100).round(1)
    pareto_cutoff = int((cumulative <= 80).sum()) + 1
    result["kpi7"] = {
        "top_diagnoses": {str(d).title():int(c) for d,c in diag_counts.items()},
        "cumulative":    {str(d).title():float(c) for d,c in cumulative.items()},
        "pareto_cutoff": pareto_cutoff
    }

    # ── KPI 3: ANOVA — Revenue by Department ─────────────────────────────
    groups  = [grp["billed_amount"].dropna().values for _, grp in df.groupby("department")]
    f_stat, p_val = scipy_stats.f_oneway(*groups)
    dept_revenue  = {}
    for dept, grp in df.groupby("department"):
        dept_revenue[dept] = {
            "mean":round(float(grp["billed_amount"].mean()),2),
            "median":round(float(grp["billed_amount"].median()),2),
            "std":round(float(grp["billed_amount"].std()),2),
            "count":len(grp)
        }
    result["anova"] = {
        "f_stat":round(float(f_stat),4),"p_value":round(float(p_val),4),
        "significant":bool(p_val < 0.05),"dept_revenue":dept_revenue
    }

    # ── KPI 8: Chi-square — Department vs Claim Status ───────────────────
    contingency = pd.crosstab(df["department"], df["claim_status"])
    chi2, p_chi, dof, _ = chi2_contingency(contingency)
    result["chi_square"] = {
        "chi2":round(float(chi2),4),"p_value":round(float(p_chi),4),
        "dof":int(dof),"significant":bool(p_chi < 0.05),
        "contingency": {dept: {s: int(v) for s,v in row.items()}
                        for dept, row in contingency.to_dict("index").items()}
    }

    # ── KPI 9: Model Comparison ───────────────────────────────────────────
    df_ml = train.copy()
    df_ml["target"] = (df_ml["claim_status"]=="Denied").astype(int)
    le_dept = LabelEncoder()
    le_cpt  = LabelEncoder()
    df_ml["dept_enc"] = le_dept.fit_transform(df_ml["department"].fillna("Unknown"))
    df_ml["cpt_enc"]  = le_cpt.fit_transform(df_ml["cpt_code"].fillna("0"))
    feats = ["billed_amount","dept_enc","cpt_enc","is_rushed","patient_age"]
    df_ml = df_ml.dropna(subset=feats)
    X = df_ml[feats]; y = df_ml["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    ml_models = {
        "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000),
        "Decision Tree":       DecisionTreeClassifier(class_weight="balanced", max_depth=5, random_state=42),
        "Random Forest":       RandomForestClassifier(class_weight="balanced", n_estimators=100, random_state=42)
    }
    model_results = {}
    rf_importance  = {}
    for name, mdl in ml_models.items():
        mdl.fit(X_train, y_train)
        yp = mdl.predict(X_test)
        model_results[name] = {
            "accuracy":  round(accuracy_score(y_test, yp)*100,1),
            "precision": round(precision_score(y_test, yp, zero_division=0)*100,1),
            "recall":    round(recall_score(y_test, yp, zero_division=0)*100,1),
            "f1":        round(f1_score(y_test, yp, zero_division=0)*100,1),
        }
    rf = ml_models["Random Forest"]
    rf_importance = {f: round(float(v),4) for f,v in zip(feats, rf.feature_importances_)}
    result["model_comparison"] = model_results
    result["rf_importance"]    = rf_importance

    # ── Coding Accuracy (load from CSV if available) ──────────────────────
    coding_stats = None
    if os.path.exists(CODING_CSV):
        try:
            ca = pd.read_csv(CODING_CSV)
            if "match" in ca.columns and "is_correct" in ca.columns:
                total_c      = len(ca)
                match_count  = int(ca["match"].sum())
                avg_conf     = round(float(ca["confidence"].mean()), 1)
                high_conf    = int((ca["confidence"] >= 80).sum())
                y_true = ca["match"].astype(int).values
                y_pred = ca["is_correct"].astype(int).values
                coding_stats = {
                    "total":total_c,"match_count":match_count,
                    "mismatch_count":total_c - match_count,
                    "match_rate":round(match_count/total_c*100,1),
                    "avg_confidence":avg_conf,
                    "high_confidence":high_conf,
                    "accuracy": round(accuracy_score(y_true, y_pred)*100,1),
                    "precision":round(precision_score(y_true, y_pred, zero_division=0)*100,1),
                    "recall":   round(recall_score(y_true, y_pred, zero_division=0)*100,1),
                    "f1":       round(f1_score(y_true, y_pred, zero_division=0)*100,1),
                }
                # Confidence distribution
                bins   = [0,50,70,80,90,100]
                labels = ["0–50%","51–70%","71–80%","81–90%","91–100%"]
                ca["conf_bucket"] = pd.cut(ca["confidence"], bins=bins, labels=labels)
                coding_stats["conf_dist"] = {
                    str(k): int(v) for k,v in ca["conf_bucket"].value_counts().sort_index().items()
                }
        except Exception as e:
            print(f"  [WARNING] Could not load coding CSV: {e}")
    result["coding_stats"] = coding_stats

    _STATS_CACHE = result
    return result

# =============================================================================
# ROUTES
# =============================================================================

@app.route("/")
def index():
    return render_template("index.html",
        model_loaded  = (MODEL is not None),
        data_loaded   = (HIST_DF is not None),
        groq_keys     = len(API_KEYS),
        threshold_pct = int((THRESHOLD or 0.35)*100))


@app.route("/predict", methods=["POST"])
def predict():
    if MODEL is None:
        flash("Model not loaded. Run random_forest.py first.", "error")
        return redirect(url_for("index"))
    if "audio" not in request.files or request.files["audio"].filename == "":
        flash("Please select an audio file to upload.", "error")
        return redirect(url_for("index"))

    audio_file = request.files["audio"]
    ext        = Path(audio_file.filename).suffix.lower() or ".wav"
    tmp_path   = os.path.join(UPLOAD_DIR, f"upload_{int(time.time())}{ext}")
    audio_file.save(tmp_path)

    try:
        transcript  = transcribe_audio(tmp_path)
        enc_id      = Path(audio_file.filename).stem.upper()
        extracted   = extract_fields(transcript, enc_id)

        X           = prepare_feature_vector(extracted)
        denial_prob = float(MODEL.predict_proba(X)[0][1])
        prediction  = "Denied" if denial_prob >= THRESHOLD else "Paid"

        # CPT Code Review (only if user opted in)
        coding_review = None
        if request.form.get("run_coding_review"):
            coding_review = review_cpt_code(
                extracted.get("diagnosis"),
                extracted.get("icd10_code"),
                extracted.get("cpt_code")
            )

        risk_factors   = get_risk_factors(extracted, denial_prob, coding_review)
        recommendation = get_recommendation(denial_prob, risk_factors)
        similar_claims = get_similar_claims(extracted)
        explanation    = generate_denial_explanation(extracted, round(denial_prob*100,1), risk_factors)

        return render_template("result.html",
            transcript=transcript, extracted=extracted,
            denial_prob=round(denial_prob*100,1),
            paid_prob=round((1-denial_prob)*100,1),
            prediction=prediction,
            threshold_pct=int(THRESHOLD*100),
            risk_factors=risk_factors,
            recommendation=recommendation,
            similar_claims=similar_claims,
            coding_review=coding_review,
            explanation=explanation)

    except Exception as e:
        flash(f"Error processing audio: {str(e)}", "error")
        return redirect(url_for("index"))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html", stats=compute_dashboard_stats())


@app.route("/statistics")
def statistics():
    return render_template("statistics.html", stats=compute_statistical_analysis())


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("\n" + "="*55)
    print("  MedBill AI — Starting Flask Server")
    print("="*55)
    app.run(debug=True, port=5000)
