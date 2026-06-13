import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import io, base64, json, os
from PIL import Image

app = Flask(__name__)
CORS(app)

MODEL_PATH = "model.keras"   
IMG_SIZE   = (128, 128)
THRESHOLD  = 0.5              # optimal threshold from your analysis
MC_PASSES  = 50

model = load_model(MODEL_PATH)

def mc_predict(img_array):
    """Run MC Dropout inference — returns mean, std, label, confidence."""
    preds = np.array([
        model(img_array, training=True)[0][0]
        for _ in range(MC_PASSES)
    ])
    mean        = float(np.mean(preds))
    uncertainty = float(np.std(preds))
    label       = "Dirty" if mean >= THRESHOLD else "Clean"
    confidence  = mean if label == "Dirty" else (1 - mean)
    eps         = 1e-8
    entropy     = -(mean * np.log(mean + eps) + (1 - mean) * np.log(1 - mean + eps))
    entropy_norm = float(entropy / np.log(2))
    return {
        "label":        label,
        "confidence":   round(confidence,    4),
        "raw_score":    round(mean,          4),
        "uncertainty":  round(uncertainty,   4),
        "entropy":      round(entropy_norm,  4),
        "is_ambiguous": bool(uncertainty > 0.15),
        "threshold_used": THRESHOLD
    }

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    img  = Image.open(file.stream).convert("RGB").resize(IMG_SIZE)
    arr  = np.expand_dims(np.array(img) / 255.0, axis=0)
    result = mc_predict(arr)
    return jsonify(result)

@app.route("/stats", methods=["GET"])
def stats():
    """Serve pre-computed stats for the Statistics tab."""
    files = {
        "summary":     "stats_summary.json",
        "tests":       "test_results.json",
        "uncertainty": "uncertainty_results.json"
    }
    out = {}
    for key, path in files.items():
        if os.path.exists(path):
            with open(path) as f:
                out[key] = json.load(f)
        else:
            out[key] = None
    return jsonify(out)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": MODEL_PATH, "threshold": THRESHOLD})

if __name__ == "__main__":
    app.run(debug=True, port=5000)