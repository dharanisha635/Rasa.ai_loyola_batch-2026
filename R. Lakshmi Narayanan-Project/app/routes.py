from app.predictor import client
import hashlib
import os
from flask import Blueprint, render_template, request, jsonify
from app.predictor import predict_leaf
from app.database import (
    init_db, save_scan, save_feedback,
    get_disease_counts, get_severity_scores,
    get_feedback_labels, get_trend_data, get_total_scans,
    save_plant_correction, get_correction_by_hash, get_most_corrected_plant
)
from app.pixel_analysis import measure_affected_area
from app.statistics_engine import (
    severity_summary, prevalence_rates,
    compute_prf_from_feedback, moving_average
)

main = Blueprint("main", __name__)
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "static", "uploads")

def hash_image(image_path: str) -> str:
    """Generate a hash fingerprint of the image for correction lookup."""
    try:
        with open(image_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except:
        return ""


@main.route("/", methods=["GET", "POST"])
def index():
    data = None
    error = None
    image_path = None
    scan_id = None
    image_hash = ""

    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename:
            if not os.path.exists(UPLOAD_FOLDER):
                os.makedirs(UPLOAD_FOLDER)

            import time as _time
            safe_name = file.filename.replace(" ", "_")
            unique_name = f"{int(_time.time())}_{safe_name}"
            image_path = os.path.join(UPLOAD_FOLDER, unique_name)
            file.save(image_path)

            data, error = predict_leaf(image_path)

            if data:
                # ── Image hash for correction memory ──
                img_hash = hash_image(image_path)
                data["image_hash"] = img_hash
                image_hash = img_hash

                # Check if this exact image was corrected before
                exact_correction = get_correction_by_hash(img_hash)
                if exact_correction:
                    data["original_ai_plant"] = data["plant_name"]
                    data["plant_name"] = exact_correction
                    data["correction_applied"] = True
                    data["correction_note"] = "Plant name corrected from community feedback"
                else:
                    # Check if AI's guess is frequently wrong
                    frequent_correction = get_most_corrected_plant(data["plant_name"])
                    if frequent_correction:
                        data["original_ai_plant"] = data["plant_name"]
                        data["plant_name"] = frequent_correction
                        data["correction_applied"] = True
                        data["correction_note"] = "Plant name updated based on previous user corrections"
                    else:
                        data["correction_applied"] = False
                        data["correction_note"] = ""
                        data["original_ai_plant"] = data["plant_name"]

                pixel_result = measure_affected_area(image_path)
                if pixel_result["affected_area_real"] is not None:
                    data["affected_area_real"] = pixel_result["affected_area_real"]
                    data["healthy_area_real"]  = pixel_result["healthy_area_real"]
                else:
                    data["affected_area_real"] = data["affected_area"]
                    data["healthy_area_real"]  = round(100 - data["affected_area"], 1)

                scan_id = save_scan(data)
                data["scan_id"] = scan_id

            filename = os.path.basename(image_path).replace("\\", "/").replace(" ", "_")
            image_path = "/static/uploads/" + filename

    return render_template(
        "index.html",
        data=data,
        error=error,
        image_path=image_path,
        scan_id=scan_id,
        image_hash=image_hash
    )


@main.route("/feedback", methods=["POST"])
def feedback():
    body     = request.get_json()
    scan_id  = body.get("scan_id")
    response = body.get("feedback")

    if scan_id and response in ("correct", "incorrect"):
        save_feedback(int(scan_id), response)
        return jsonify({"status": "saved"})
    return jsonify({"status": "error"}), 400


@main.route("/correct_plant", methods=["POST"])
def correct_plant():
    body         = request.get_json()
    scan_id      = body.get("scan_id")
    correct_name = body.get("correct_plant", "").strip()
    image_hash   = body.get("image_hash", "")

    if scan_id and correct_name and image_hash:
        save_plant_correction(int(scan_id), correct_name, image_hash)
        return jsonify({"status": "saved", "plant": correct_name})
    return jsonify({"status": "error"}), 400


def generate_ai_report(prompt):
    response = client.chat(
        model="pixtral-12b-2409",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ]
            }
        ],
        temperature=0
    )
    return response.choices[0].message.content


@main.route("/generate_report", methods=["POST"])
def generate_report():
    body = request.get_json()
    data = body.get("data", {})

    prompt = f"""You are an expert plant pathologist. Write a formal diagnostic report based on the following scan data.

Plant: {data.get('plant_name')}
Disease: {data.get('disease')}
Severity: {data.get('severity_level')}
Confidence: {data.get('confidence')}%
Affected Area: {data.get('affected_area')}%
Recovery Chance: {data.get('recovery_chance')}%
Risk Score: {data.get('risk_score')}
Summary: {data.get('summary')}
Cause: {data.get('cause')}
Remedies: {', '.join(data.get('remedies', []))}
Precautions: {', '.join(data.get('precautions', []))}
Favourable Temp: {data.get('temp_min')}C - {data.get('temp_max')}C
Favourable Humidity: {data.get('hum_min')}% - {data.get('hum_max')}%

Write a neat and brief professional report with these sections:
1. Executive Summary
2. Disease Identification
3. Environmental & Causal Analysis
4. Risk Assessment
5. Treatment Recommendations
6. Preventive Measures
7. Prognosis

Be specific, professional, and actionable. Use plain text only. No markdown whatsoever. No asterisks, no bold, no italics, no dashes for separators. Use plain section headings like "1. EXECUTIVE SUMMARY" followed by normal paragraphs."""

    try:
        report_text = generate_ai_report(prompt)
        return jsonify({"report": report_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@main.route("/analytics")
def analytics():
    total           = get_total_scans()
    disease_counts  = get_disease_counts()
    severity_scores = get_severity_scores()
    feedback_rows   = get_feedback_labels()
    trend_data      = get_trend_data(days=30)

    prevalence     = prevalence_rates(disease_counts, total)
    sev_stats      = severity_summary(severity_scores)
    prf_metrics    = compute_prf_from_feedback(feedback_rows)

    trend_labels   = [d["day"] for d in trend_data]
    trend_values   = [d["avg_severity"] for d in trend_data]
    trend_smoothed = moving_average(trend_values, window=7)

    return render_template(
        "analytics.html",
        total          = total,
        prevalence     = prevalence,
        sev_stats      = sev_stats,
        prf_metrics    = prf_metrics,
        trend_labels   = trend_labels,
        trend_smoothed = trend_smoothed
    )