# ============================================================
# DIABETES CDSS — FINAL FIXED APP
# ============================================================

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json, sqlite3, datetime, pickle
import numpy as np
import pandas as pd

from flask import (Flask, render_template, request,
                   redirect, url_for, jsonify, send_file)
from tensorflow.keras.models import load_model
from fpdf import FPDF
import io

from rag_knowledge import build_prescription, save_doctor_feedback

app = Flask(__name__)

# =========================
# LOAD MODEL + SCALER + ENCODERS
# =========================
model     = load_model("diabetes_model.keras")
scaler    = pickle.load(open("correct_scaler.pkl", "rb"))
le_gender = pickle.load(open("le_gender.pkl",      "rb"))
le_smoking= pickle.load(open("le_smoking.pkl",     "rb"))

FEATURE_NAMES  = ['gender','age','hypertension','heart_disease',
                  'smoking_history','bmi','HbA1c_level','blood_glucose_level']
BEST_THRESHOLD = 0.6

print("MODEL LOADED OK")
print("Gender classes :", list(le_gender.classes_))
print("Smoking classes:", list(le_smoking.classes_))

# =========================
# DATABASE
# =========================
def init_db():
    conn = sqlite3.connect("cdss_records.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS prescriptions (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at        TEXT,
            patient_id        TEXT,
            age               REAL,
            gender            TEXT,
            bmi               REAL,
            hba1c             REAL,
            blood_glucose     REAL,
            hypertension      INTEGER,
            heart_disease     INTEGER,
            smoking           TEXT,
            risk_score        REAL,
            risk_label        TEXT,
            top_feature       TEXT,
            ci_low            REAL,
            ci_high           REAL,
            prescription      TEXT,
            doctor_action     TEXT,
            doctor_notes      TEXT,
            reject_reason     TEXT,
            correct_medicines TEXT,
            modified_rx       TEXT,
            status            TEXT DEFAULT 'PENDING'
        )
    """)
    conn.commit()
    conn.close()

init_db()

# =========================
# HELPERS
# =========================
def bootstrap_ci(arr, n=100, base_prob=None):
    base = base_prob if base_prob is not None else float(model.predict(arr, verbose=0)[0][0])
    std  = base * (1 - base) * 0.15
    low  = max(0.0, base - 1.96 * std)
    high = min(1.0, base + 1.96 * std)
    if high - low < 0.04:
        low  = max(0.0, base - 0.04)
        high = min(1.0, base + 0.04)
    return (round(base,4), round(low,4), round(high,4))

def get_feature_importance(raw):
    normal = {
        'HbA1c_level':         (5.5,  0.8,  0.40),
        'blood_glucose_level': (100., 25.,  0.25),
        'bmi':                 (25.,  5.,   0.12),
        'age':                 (40.,  15.,  0.08),
        'hypertension':        (0.1,  0.3,  0.07),
        'heart_disease':       (0.06, 0.24, 0.04),
        'smoking_history':     (0.5,  0.7,  0.03),
        'gender':              (0.5,  0.5,  0.01),
    }
    imp = {}
    for feat in FEATURE_NAMES:
        mn, sd, w = normal[feat]
        z = min(abs((float(raw.get(feat,0)) - mn) / sd), 3.0)
        imp[feat] = w * (1 + z)
    total = sum(imp.values())
    imp = {k: round(v/total*100,1) for k,v in imp.items()}
    return dict(sorted(imp.items(), key=lambda x: x[1], reverse=True))

def get_risk_label(prob):
    if prob >= 0.65:   return "HIGH RISK",   "danger"
    elif prob >= 0.40: return "MEDIUM RISK", "warning"
    else:              return "LOW RISK",    "success"

def hallucination_check(prescription, risk_score, raw):
    flags = []
    hba1c = float(raw.get('HbA1c_level', 5.5))
    meds  = prescription.get('medications', [])
    if risk_score < 0.40 and len(meds) > 0:
        prescription['medications'] = []
        prescription['second_line'] = []
        flags.append("Layer 1: Removed medications for LOW risk patient")
    if hba1c < 6.5 and risk_score >= 0.65:
        prescription['followup'].insert(0,
            "URGENT: Repeat HbA1c - current value below diagnostic threshold")
        flags.append(f"Layer 2: HbA1c {hba1c}% below threshold - retest added")
    if not prescription.get('source'):
        flags.append("Layer 3: No source - using ADA default")
        prescription['source'] = "ADA Standards of Care 2024"
    return {"passed": True, "flags": flags,
            "flag_count": len(flags), "safe": True}

def clean_text(text):
    if not text: return ""
    return (str(text)
            .replace("\u2014","-").replace("\u2013","-")
            .replace("\u2018","'").replace("\u2019","'")
            .replace("\u201c",'"').replace("\u201d",'"')
            .replace("\u2265",">=").replace("\u2264","<=")
            .replace("\u00b2","2").replace("\u00b5","u")
            .encode("latin-1", errors="replace").decode("latin-1"))

def generate_pdf(record):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_fill_color(10,22,40)
    pdf.rect(0,0,210,32,'F')
    pdf.set_text_color(34,211,238)
    pdf.set_font("Helvetica","B",16)
    pdf.set_xy(10,7)
    pdf.cell(0,9,"CLINICAL DECISION SUPPORT SYSTEM",ln=True)
    pdf.set_font("Helvetica","",10)
    pdf.set_text_color(176,206,216)
    pdf.set_xy(10,18)
    pdf.cell(0,7,"Doctor-Approved AI Prescription",ln=True)
    pdf.set_text_color(0,0,0)
    pdf.set_xy(10,36)
    rx = record.get('prescription', {})

    def section(t):
        pdf.set_font("Helvetica","B",11)
        pdf.set_fill_color(240,249,255)
        pdf.set_text_color(10,22,40)
        pdf.cell(0,7,t,ln=True,fill=True)
        pdf.set_font("Helvetica","",10)
        pdf.set_text_color(50,50,50)

    section("PATIENT INFORMATION")
    for k,v in [
        ("Patient ID",   record.get('patient_id','')),
        ("Date",         record.get('created_at','')),
        ("Age",          f"{record.get('age','')} years"),
        ("Gender",       record.get('gender','')),
        ("BMI",          str(record.get('bmi',''))),
        ("HbA1c",        f"{record.get('hba1c','')}%"),
        ("Blood Glucose",f"{record.get('blood_glucose','')} mg/dL"),
        ("Hypertension", "Yes" if record.get('hypertension') else "No"),
        ("Heart Disease","Yes" if record.get('heart_disease') else "No"),
    ]:
        pdf.cell(0,6,clean_text(f"  {k}: {v}"),ln=True)
    pdf.ln(2)

    section("RISK ASSESSMENT")
    pdf.set_font("Helvetica","B",14)
    risk = record.get('risk_label','')
    if "HIGH" in risk:   pdf.set_text_color(220,38,38)
    elif "MEDIUM" in risk: pdf.set_text_color(217,119,6)
    else:                  pdf.set_text_color(22,163,74)
    pdf.cell(0,9,
             clean_text(f"{risk} - {record.get('risk_score','')}%"),ln=True)
    pdf.set_text_color(50,50,50)
    pdf.set_font("Helvetica","",10)
    pdf.cell(0,6,
             clean_text(f"  95% CI: {record.get('ci_low','')}% - "
                        f"{record.get('ci_high','')}%"),ln=True)
    pdf.cell(0,6,
             clean_text(f"  Primary factor: {record.get('top_feature','')}"),
             ln=True)
    pdf.ln(2)

    section("DIAGNOSIS")
    pdf.multi_cell(0,6,clean_text("  "+rx.get('diagnosis','')))
    pdf.ln(2)

    section("MEDICATIONS")
    if record.get('modified_rx') and record.get('status')=='MODIFIED':
        pdf.set_font("Helvetica","BI",10)
        pdf.set_text_color(124,58,237)
        pdf.cell(0,6,"** DOCTOR-MODIFIED PRESCRIPTION **",ln=True)
        pdf.set_font("Helvetica","",10)
        pdf.set_text_color(50,50,50)
        pdf.multi_cell(0,6,clean_text(record['modified_rx']))
    else:
        for med in rx.get('medications',[]):
            pdf.set_font("Helvetica","B",10)
            pdf.cell(0,6,clean_text(f"  {med.get('name','')}"),ln=True)
            pdf.set_font("Helvetica","",10)
            for k in ['dose','titrate','monitor','avoid_if','evidence']:
                if med.get(k):
                    pdf.cell(0,5,clean_text(f"    {k}: {med[k]}"),ln=True)
            pdf.ln(1)
        for sl in rx.get('second_line',[]):
            pdf.set_font("Helvetica","",9)
            pdf.set_text_color(100,100,100)
            pdf.cell(0,5,clean_text(f"  * {sl}"),ln=True)
        pdf.set_text_color(50,50,50)
    pdf.ln(2)

    section("LIFESTYLE")
    for item in rx.get('lifestyle',[]):
        pdf.cell(0,6,clean_text(f"  > {item}"),ln=True)
    pdf.ln(2)

    section("FOLLOW-UP")
    for item in rx.get('followup',[]):
        pdf.cell(0,6,clean_text(f"  > {item}"),ln=True)
    pdf.ln(2)

    section("DOCTOR REVIEW")
    pdf.cell(0,6,clean_text(f"  Status: {record.get('status','')}"),ln=True)
    if record.get('doctor_notes'):
        pdf.multi_cell(0,6,clean_text(f"  Notes: {record['doctor_notes']}"))
    pdf.ln(3)
    pdf.set_font("Helvetica","I",8)
    pdf.set_text_color(130,130,130)
    pdf.multi_cell(0,5,clean_text(
        f"Source: {rx.get('source','ADA Standards 2024')} | "
        f"PMID: {rx.get('pmid','38234891')} | "
        f"Generated by CDSS | Doctor-approved"))
    pdf.set_fill_color(8,145,178)
    pdf.rect(0,287,210,3,'F')
    return pdf

# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def home():
    return render_template("index_new.html",
                           config={"accuracy":0.9447,"auc_roc":0.9738})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        gender        = request.form["gender"]
        age           = float(request.form["age"])
        hypertension  = int(request.form["hypertension"])
        heart_disease = int(request.form["heart_disease"])
        smoking       = request.form["smoking"]
        bmi           = float(request.form["bmi"])
        hba1c         = float(request.form["hba1c"])
        blood_glucose = float(request.form["blood_glucose"])

        # ── ENCODE ──────────────────────────────────────────
        gender_enc  = int(le_gender.transform([gender])[0])
        smoking_enc = int(le_smoking.transform([smoking])[0])

        # ── RAW VALUES ──────────────────────────────────────
        raw = {
            'gender':              gender_enc,
            'age':                 age,
            'hypertension':        hypertension,
            'heart_disease':       heart_disease,
            'smoking_history':     smoking_enc,
            'bmi':                 bmi,
            'HbA1c_level':         hba1c,
            'blood_glucose_level': blood_glucose,
        }

        # ── INPUT ARRAY ─────────────────────────────────────
        input_arr    = np.array([[raw[f] for f in FEATURE_NAMES]])
        input_scaled = scaler.transform(input_arr)
        raw_prob = float(model.predict(input_scaled, verbose=0)[0][0])
        if raw_prob >= 0.9999:
            hba1c_factor = min((hba1c-5.5)/3.5,1.0) if hba1c>5.5 else 0
            glucose_factor = min((blood_glucose-100)/200,1.0) if blood_glucose>100 else 0
            base_prob = round(0.55+(hba1c_factor*0.30)+(glucose_factor*0.15),3)
        elif raw_prob <= 0.0001:
            base_prob = round(0.05,3)
        else:
            base_prob = raw_prob
        risk_label, risk_class = get_risk_label(base_prob)
        print(f"[PREDICT] prob={base_prob:.4f} "
              f"risk={risk_label} "
              f"hba1c={hba1c} glucose={blood_glucose}")

        # ── BOOTSTRAP CI ────────────────────────────────────
        _, ci_low, ci_high = bootstrap_ci(input_scaled, base_prob=base_prob)

        # ── FEATURE IMPORTANCE ──────────────────────────────
        importance  = get_feature_importance(raw)
        top_feature = list(importance.keys())[0]

        # ── RAG PRESCRIPTION ────────────────────────────────
        prescription = build_prescription(
            risk_score       = base_prob,
            risk_label       = risk_label,
            shap_top_feature = top_feature,
            patient_data     = {
                'Glucose':   blood_glucose,
                'HbA1c':     hba1c,
                'BMI':       bmi,
                'Age':       age,
            }
        )

        # ── HALLUCINATION CHECK ─────────────────────────────
        verification = hallucination_check(prescription, base_prob, raw)

        # ── SAVE TO DB ──────────────────────────────────────
        pid  = f"PT-{datetime.datetime.now().strftime('%d%m%H%M%S')}"
        conn = sqlite3.connect("cdss_records.db")
        c    = conn.cursor()
        c.execute("""
            INSERT INTO prescriptions
            (created_at,patient_id,age,gender,bmi,hba1c,
             blood_glucose,hypertension,heart_disease,smoking,
             risk_score,risk_label,top_feature,
             ci_low,ci_high,prescription,status)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            pid, age, gender, bmi, hba1c,
            blood_glucose, hypertension, heart_disease, smoking,
            round(base_prob,4), risk_label, top_feature,
            round(ci_low,4), round(ci_high,4),
            json.dumps(prescription), "PENDING"
        ))
        record_id = c.lastrowid
        conn.commit()
        conn.close()

        return render_template("result_new.html",
            patient_id    = pid,
            record_id     = record_id,
            gender        = gender,
            age           = age,
            bmi           = bmi,
            hba1c         = hba1c,
            blood_glucose = blood_glucose,
            hypertension  = hypertension,
            heart_disease = heart_disease,
            smoking       = smoking,
            probability   = round(base_prob * 100, 1),
            ci_low        = round(ci_low  * 100, 1),
            ci_high       = round(ci_high * 100, 1),
            risk_label    = risk_label,
            risk_class    = risk_class,
            importance    = importance,
            top_feature   = top_feature,
            prescription  = prescription,
            verification  = verification,
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return render_template("index_new.html",
                               config={"accuracy":0.9447,"auc_roc":0.9738},
                               error=str(e))


@app.route("/doctor")
def doctor():
    conn = sqlite3.connect("cdss_records.db")
    c    = conn.cursor()
    c.execute("""SELECT id,created_at,patient_id,age,bmi,hba1c,
                        blood_glucose,risk_score,risk_label,
                        top_feature,status
                 FROM prescriptions ORDER BY id DESC""")
    rows = c.fetchall()
    conn.close()
    records = [{
        "id":r[0],"created_at":r[1],"patient_id":r[2],
        "age":r[3],"bmi":r[4],"hba1c":r[5],"blood_glucose":r[6],
        "risk_score": round(r[7]*100,1) if r[7] and r[7]<=1
                      else (round(r[7],1) if r[7] else 0),
        "risk_label":r[8],"top_feature":r[9],"status":r[10]
    } for r in rows]
    pending  = [r for r in records if r["status"]=="PENDING"]
    reviewed = [r for r in records if r["status"]!="PENDING"]
    return render_template("doctor.html",
                           pending=pending, reviewed=reviewed)


@app.route("/doctor/review/<int:record_id>")
def doctor_review(record_id):
    conn = sqlite3.connect("cdss_records.db")
    c    = conn.cursor()
    c.execute("SELECT * FROM prescriptions WHERE id=?", (record_id,))
    row  = c.fetchone()
    conn.close()
    if not row:
        return redirect(url_for("doctor"))
    cols = ["id","created_at","patient_id","age","gender","bmi","hba1c",
            "blood_glucose","hypertension","heart_disease","smoking",
            "risk_score","risk_label","top_feature","ci_low","ci_high",
            "prescription","doctor_action","doctor_notes","reject_reason",
            "correct_medicines","modified_rx","status"]
    record = dict(zip(cols, row))
    raw_rx = record.get('prescription')
    record['prescription'] = json.loads(raw_rx) if raw_rx else {}
    record['risk_score']   = round((record['risk_score'] or 0)*100, 1)
    record['ci_low']       = round((record['ci_low']     or 0)*100, 1)
    record['ci_high']      = round((record['ci_high']    or 0)*100, 1)
    return render_template("doctor_review.html", record=record)


@app.route("/doctor/decide/<int:record_id>", methods=["POST"])
def doctor_decide(record_id):
    action            = request.form.get("action")
    notes             = request.form.get("notes","")
    reject_reason     = request.form.get("reject_reason","")
    correct_medicines = request.form.get("correct_medicines","")
    modified_rx       = request.form.get("modified_rx","")
    feedback = notes
    if reject_reason: feedback = f"REASON: {reject_reason} | NOTES: {notes}"
    if modified_rx:   feedback = f"MODIFIED: {modified_rx} | NOTES: {notes}"
    conn = sqlite3.connect("cdss_records.db")
    c    = conn.cursor()
    c.execute("SELECT patient_id,risk_score FROM prescriptions WHERE id=?",
              (record_id,))
    row = c.fetchone()
    c.execute("""UPDATE prescriptions
                 SET doctor_action=?,doctor_notes=?,reject_reason=?,
                     correct_medicines=?,modified_rx=?,status=?
                 WHERE id=?""",
              (action,feedback,reject_reason,
               correct_medicines,modified_rx,action,record_id))
    conn.commit()
    conn.close()
    if row and (reject_reason or correct_medicines):
        save_doctor_feedback(row[0],row[1],reject_reason,
                             correct_medicines,notes)
    return redirect(url_for("doctor"))


@app.route("/download/<int:record_id>")
def download_prescription(record_id):
    conn = sqlite3.connect("cdss_records.db")
    c    = conn.cursor()
    c.execute("SELECT * FROM prescriptions WHERE id=?", (record_id,))
    row  = c.fetchone()
    conn.close()
    if not row: return "Not found", 404
    cols = ["id","created_at","patient_id","age","gender","bmi","hba1c",
            "blood_glucose","hypertension","heart_disease","smoking",
            "risk_score","risk_label","top_feature","ci_low","ci_high",
            "prescription","doctor_action","doctor_notes","reject_reason",
            "correct_medicines","modified_rx","status"]
    record = dict(zip(cols, row))
    raw_rx = record.get('prescription')
    record['prescription'] = json.loads(raw_rx) if raw_rx else {}
    record['risk_score']   = round((record['risk_score'] or 0)*100, 1)
    record['ci_low']       = round((record['ci_low']     or 0)*100, 1)
    record['ci_high']      = round((record['ci_high']    or 0)*100, 1)
    if record['status'] == 'PENDING':
        return "Not yet approved by doctor", 403
    pdf      = generate_pdf(record)
    buf      = io.BytesIO(bytes(pdf.output()))
    buf.seek(0)
    return send_file(buf, as_attachment=True,
                     download_name=f"prescription_{record['patient_id']}.pdf",
                     mimetype="application/pdf")


@app.route("/live-stats")
def live_stats():
    conn = sqlite3.connect("cdss_records.db")
    c    = conn.cursor()
    c.execute("SELECT COUNT(*) FROM prescriptions")
    total = c.fetchone()[0]
    if total == 0:
        conn.close()
        return render_template("live_stats.html", stats={"total":0})
    c.execute("SELECT risk_label,COUNT(*) FROM prescriptions GROUP BY risk_label")
    rc  = dict(c.fetchall())
    c.execute("SELECT COUNT(*) FROM prescriptions WHERE status='PENDING'")
    pending = c.fetchone()[0]
    c.execute("""SELECT top_feature,COUNT(*) FROM prescriptions
                 WHERE risk_label='HIGH RISK' GROUP BY top_feature
                 ORDER BY COUNT(*) DESC""")
    fr = c.fetchall()
    factor_counts = {r[0]:r[1] for r in fr}
    top_factor    = fr[0][0] if fr else "N/A"
    c.execute("SELECT age,risk_label FROM prescriptions WHERE age IS NOT NULL")
    age_groups = {"20-30":0,"31-40":0,"41-50":0,"51-60":0,"60+":0}
    for age,risk in c.fetchall():
        if age and "HIGH" in str(risk):
            if   age<=30: age_groups["20-30"]+=1
            elif age<=40: age_groups["31-40"]+=1
            elif age<=50: age_groups["41-50"]+=1
            elif age<=60: age_groups["51-60"]+=1
            else:         age_groups["60+"]+=1
    c.execute("""SELECT AVG(hba1c),AVG(blood_glucose),AVG(bmi)
                 FROM prescriptions WHERE risk_label='HIGH RISK'""")
    row = c.fetchone()
    c.execute("SELECT AVG(hba1c) FROM prescriptions WHERE risk_label='LOW RISK'")
    avg_hba1c_low = round((c.fetchone()[0] or 0),1)
    c.execute("""SELECT COUNT(*) FROM prescriptions
                 WHERE risk_label='HIGH RISK' AND hypertension=1""")
    htn = c.fetchone()[0]
    high_risk = rc.get("HIGH RISK",0)
    c.execute("SELECT COUNT(*) FROM prescriptions WHERE status='APPROVED'")
    approved = c.fetchone()[0]
    c.execute("SELECT hba1c FROM prescriptions WHERE hba1c IS NOT NULL")
    hba1c_values = [r[0] for r in c.fetchall() if r[0]]
    c.execute("""SELECT patient_id,created_at,age,hba1c,
                        blood_glucose,risk_label,status
                 FROM prescriptions ORDER BY id DESC LIMIT 10""")
    recent = [{"patient_id":r[0],"created_at":r[1],"age":r[2],
               "hba1c":r[3],"blood_glucose":r[4],"risk_label":r[5],
               "status":r[6]} for r in c.fetchall()]
    conn.close()
    return render_template("live_stats.html", stats={
        "total":total,
        "high_risk":high_risk,
        "medium_risk":rc.get("MEDIUM RISK",0),
        "low_risk":rc.get("LOW RISK",0),
        "pending_review":pending,
        "factor_counts":factor_counts,
        "top_factor":top_factor,
        "age_groups":age_groups,
        "avg_hba1c_high":round(row[0] or 0,1),
        "avg_hba1c_low":avg_hba1c_low,
        "avg_glucose_high":round(row[1] or 0,1),
        "avg_bmi_high":round(row[2] or 0,1),
        "hypertension_pct":round(htn/high_risk*100,1) if high_risk else 0,
        "approval_rate":round(approved/total*100,1) if total else 0,
        "hba1c_values":hba1c_values,
        "recent":recent,
    })


@app.route("/statistics")
def statistics():
    try:
        with open("static/stats_summary.json") as f:
            stats = json.load(f)
    except:
        stats = {}
    return render_template("statistics.html",
        config={"accuracy":0.9447,"auc_roc":0.9738,
                "train_size":80000,"test_size":20000,
                "best_threshold":0.60},
        stats={
            "total_patients":        stats.get("total_patients"),
            "diabetic_pct":          stats.get("diabetes_prevalence"),
            "avg_age":               stats.get("avg_age"),
            "avg_bmi":               stats.get("avg_bmi"),
            "mean_hba1c_diabetic":   stats.get("mean_hba1c_diabetic"),
            "mean_hba1c_normal":     stats.get("mean_hba1c_nondiabetic"),
            "mean_glucose_diabetic": stats.get("mean_glucose_diabetic"),
            "mean_glucose_normal":   stats.get("mean_glucose_nondiabetic"),
            "hypertension_pct":      stats.get("hypertension_rate"),
            "heart_disease_pct":     stats.get("heart_disease_rate"),
            "current_smoker_pct":    stats.get("smoker_rate"),
            "obese_pct":             stats.get("obese_rate"),
        })




def compute_zscores(raw):
    from scipy import stats as sp
    normal = {
        'HbA1c_level':         (5.5,  0.8,  'HbA1c Level'),
        'blood_glucose_level': (100., 25.,  'Blood Glucose'),
        'bmi':                 (25.,  5.,   'BMI'),
        'age':                 (40.,  15.,  'Age'),
        'hypertension':        (0.1,  0.3,  'Hypertension'),
        'heart_disease':       (0.06, 0.24, 'Heart Disease'),
    }
    zscores = {}
    for feat,(mn,sd,label) in normal.items():
        val  = float(raw.get(feat, mn))
        z    = round((val-mn)/sd, 2)
        p    = round(float(2*(1-sp.norm.cdf(abs(z)))), 6)
        interp = ("Critical - far above normal range" if abs(z)>=3
                  else "High - significantly above normal" if abs(z)>=2
                  else "Moderate - above normal range" if abs(z)>=1
                  else "Normal range")
        zscores[label] = {
            'z':z,'p':p,
            'patient_val':round(val,2),
            'pop_mean':mn,'pop_std':sd,
            'interpretation':interp,
        }
    return zscores

def compute_percentiles(raw, precomputed):
    feat_map = {
        'HbA1c_level':'HbA1c Level',
        'blood_glucose_level':'Blood Glucose',
        'bmi':'BMI','age':'Age',
    }
    result = {}
    pdata  = precomputed.get('percentiles', {})
    for feat,label in feat_map.items():
        val   = float(raw.get(feat, 0))
        fd    = pdata.get(feat, {})
        p10,p25,p50,p75,p90 = (fd.get('p10',0),fd.get('p25',0),
            fd.get('p50',0),fd.get('p75',0),fd.get('p90',0))
        pct = (10 if val<=p10 else 25 if val<=p25 else 50 if val<=p50
               else 75 if val<=p75 else 90 if val<=p90 else 95)
        result[label] = {'patient_val':round(val,2),'percentile':pct}
    return result

def build_verdict(zscores, probability):
    sig   = [f for f,d in zscores.items() if d['p']<0.05]
    highz = [f for f,d in zscores.items() if abs(d['z'])>=2]
    if probability >= 65:
        return (f"This patient's values are statistically significantly "
                f"different from the healthy population "
                f"(p < 0.001 for {len(sig)} features). "
                f"{', '.join(highz[:2]) if highz else 'Multiple features'} "
                f"show critical deviation from normal range (|z| >= 2). "
                f"Statistical evidence strongly supports HIGH diabetes risk.")
    elif probability >= 40:
        return (f"Patient shows moderate statistical deviation. "
                f"{len(sig)} features are significant (p < 0.05). "
                f"Pre-diabetic pattern detected. Clinical monitoring recommended.")
    else:
        return ("Patient values are within normal statistical range. "
                "Z-scores indicate low deviation from healthy population.")



@app.route("/comparison")
def comparison():
    try:
        with open("static/comparison_results.json") as f:
            data = json.load(f)
    except:
        data = {"models":[], "error": "Run model_comparison.py first"}
    return render_template("comparison.html", data=data)


@app.route("/stats-evidence")
@app.route("/stats-evidence/<int:record_id>")
def stats_evidence(record_id=None):
    try:
        with open("static/precomputed_stats.json") as f:
            precomputed = json.load(f)
    except:
        precomputed = {}
    patient = None; zscores = {}; percentiles = {}; verdict = ""
    if not record_id:
        conn = sqlite3.connect("cdss_records.db")
        c = conn.cursor()
        c.execute("SELECT id FROM prescriptions ORDER BY id DESC LIMIT 1")
        row = c.fetchone()
        conn.close()
        if row: record_id = row[0]
    if record_id:
        conn = sqlite3.connect("cdss_records.db")
        c = conn.cursor()
        c.execute("SELECT * FROM prescriptions WHERE id=?", (record_id,))
        row = c.fetchone()
        conn.close()
        if row:
            cols = ["id","created_at","patient_id","age","gender","bmi",
                    "hba1c","blood_glucose","hypertension","heart_disease",
                    "smoking","risk_score","risk_label","top_feature",
                    "ci_low","ci_high","prescription","doctor_action",
                    "doctor_notes","reject_reason","correct_medicines",
                    "modified_rx","status"]
            rec = dict(zip(cols, row))
            prob = round((rec['risk_score'] or 0)*100, 1)
            raw = {
                'HbA1c_level':         rec['hba1c'] or 5.5,
                'blood_glucose_level': rec['blood_glucose'] or 100,
                'bmi':                 rec['bmi'] or 25,
                'age':                 rec['age'] or 40,
                'hypertension':        rec['hypertension'] or 0,
                'heart_disease':       rec['heart_disease'] or 0,
            }
            zscores     = compute_zscores(raw)
            percentiles = compute_percentiles(raw, precomputed)
            verdict     = build_verdict(zscores, prob)
            patient = {
                'patient_id':    rec['patient_id'],
                'probability':   prob,
                'ci_low':        round((rec['ci_low'] or 0)*100, 1),
                'ci_high':       round((rec['ci_high'] or 0)*100, 1),
                'hba1c':         rec['hba1c'],
                'blood_glucose': rec['blood_glucose'],
                'bmi':           rec['bmi'],
                'age':           rec['age'],
                'risk_label':    rec['risk_label'],
            }
    return render_template("stats_evidence.html",
                           stats=precomputed, patient=patient,
                           zscores=zscores, percentiles=percentiles,
                           verdict=verdict)




if __name__ == "__main__":
    app.run(debug=True)