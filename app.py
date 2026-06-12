import cv2
import mediapipe as mp
import numpy as np
import pickle
import threading
import time
import collections
import math
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"]   = "3"
os.environ["CUDA_VISIBLE_DEVICES"]    = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"]  = "0"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

from flask import Flask, Response, render_template, jsonify, request
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    print("⚠️  No GEMINI_API_KEY found in .env file.")
else:
    genai.configure(api_key=GEMINI_API_KEY)
    print("✅ Gemini API key loaded.")

try:
    from features import extract_statistical_features
except ImportError:
    def extract_statistical_features(landmarks):
        if len(landmarks) < 63:
            return [0.0] * 7
        a = np.array(landmarks[:63]).reshape(21, 3)
        return [float(np.mean(a)), float(np.std(a)), float(np.min(a)),
                float(np.max(a)), float(np.ptp(a)), float(np.median(a)),
                float(np.var(a))]

# ── Vocabulary ──────────────────────────────────────────────────────────────
GESTURES = {
    0:"Me", 1:"Hello", 2:"Please", 3:"You", 4:"Sorry",
    5:"House", 6:"Thank you", 7:"Time", 8:"Food", 9:"See"
}
N              = len(GESTURES)
GESTURE_TO_IDX = {v:k for k,v in GESTURES.items()}
HOLD_SEC       = 0.8

# ── Linguistic prior ────────────────────────────────────────────────────────
# Weights reflect natural English word-pair tendencies for this vocabulary.
# Higher = more likely to follow the current gesture.
_RAW_PRIOR = {
    "Me":        {"Hello":2, "Please":3, "Sorry":4, "See":3, "You":2,
                  "Thank you":2, "Food":1, "House":1, "Time":1},
    "Hello":     {"You":5,  "Please":2, "See":3,  "Me":2,  "Thank you":2,
                  "Sorry":1, "Food":1,  "House":1, "Time":1},
    "Please":    {"Food":4, "See":3,   "You":3,  "Me":2,  "Sorry":1,
                  "Hello":1, "House":2, "Time":2, "Thank you":1},
    "You":       {"Please":3, "See":4, "Sorry":2, "Hello":2, "Me":2,
                  "Food":2,  "House":2, "Time":2, "Thank you":3},
    "Sorry":     {"Me":4,  "Please":3, "You":3,  "Thank you":2, "Hello":1,
                  "See":1, "Food":1,  "House":1, "Time":1},
    "House":     {"Me":3,  "You":3,   "Food":3,  "Time":2, "See":2,
                  "Please":2, "Thank you":2, "Hello":1, "Sorry":1},
    "Thank you": {"You":5, "Me":3,    "See":2,   "Please":1, "Hello":2,
                  "Sorry":1, "Food":1, "House":1, "Time":1},
    "Time":      {"Please":3, "Food":3, "See":2, "You":2,  "Me":2,
                  "Sorry":1, "Hello":1, "House":1, "Thank you":1},
    "Food":      {"Please":4, "Me":3,  "You":3,  "Thank you":2, "Time":2,
                  "See":2,   "Hello":1, "House":1, "Sorry":1},
    "See":       {"You":5, "Me":3,    "Hello":2,  "Thank you":2, "Please":2,
                  "Sorry":1, "Food":1, "House":1,  "Time":1},
}

def _build_prior():
    """Normalize _RAW_PRIOR rows to probabilities, keeping only known gestures."""
    prior = {}
    vocab = set(GESTURES.values())
    for src, targets in _RAW_PRIOR.items():
        if src not in vocab:
            continue
        row   = {k: v for k, v in targets.items() if k in vocab}
        total = sum(row.values())
        if total:
            prior[src] = {k: v / total for k, v in row.items()}
        else:
            prior[src] = {g: 1 / N for g in vocab}
    # Fill any missing source gesture with a uniform distribution
    for g in vocab:
        if g not in prior:
            prior[g] = {t: 1 / N for t in vocab}
    return prior

LINGUISTIC_PRIOR = _build_prior()

# α blend: how much session Markov data vs linguistic prior to trust.
# Grows 0 → MAX_ALPHA as session transitions accumulate.
MAX_ALPHA        = 0.75   # cap — always keep at least 25% linguistic prior
ALPHA_SATURATION = 40     # transitions needed to reach MAX_ALPHA


def _blended_next(gesture, markov_probs, session_transition_count):
    """
    Returns (predicted_next_gesture, blended_probability_pct).
    Blends session Markov probabilities with the static linguistic prior.
    Cold start  → mostly linguistic prior (useful from gesture #1).
    Warm session → mostly learned Markov (personalized to the signer).
    """
    ci = GESTURE_TO_IDX.get(gesture)
    if ci is None:
        return None, 0.0

    alpha      = min(MAX_ALPHA, session_transition_count / ALPHA_SATURATION * MAX_ALPHA)
    markov_row = markov_probs[ci]
    prior_row  = LINGUISTIC_PRIOR.get(gesture, {})

    blended = {}
    for idx, gname in GESTURES.items():
        m             = markov_row[idx]
        p             = prior_row.get(gname, 1 / N)
        blended[idx]  = alpha * m + (1 - alpha) * p

    best_idx  = max(blended, key=blended.get)
    best_prob = blended[best_idx]

    # Suppress self-prediction unless it strongly dominates
    if best_idx == ci and best_prob < 0.55:
        blended.pop(best_idx)
        if blended:
            best_idx  = max(blended, key=blended.get)
            best_prob = blended[best_idx]
        else:
            return None, 0.0

    return GESTURES[best_idx], round(best_prob * 100, 1)


# ── Shared state ────────────────────────────────────────────────────────────
lock  = threading.Lock()
state = {
    "current_gesture":      None,
    "confidence":           0.0,
    "sequence":             [],
    "freq_table":           {g: 0  for g in GESTURES.values()},
    "markov_counts":        [[0]   * N for _ in range(N)],
    "markov_probs":         [[0.0] * N for _ in range(N)],
    "next_prediction":      None,
    "next_prob":            0.0,
    "sentence":             "",
    "translating":          False,
    "distribution_entropy": 0.0,
    "confidence_history":   collections.deque(maxlen=60),
    "gesture_avg_conf":     {g: [] for g in GESTURES.values()},
    "session_start":        time.time(),
}


def _async_record_worker(gesture, conf):
    with lock:
        seq       = list(state["sequence"])
        counts    = [row[:] for row in state["markov_counts"]]
        probs     = [row[:] for row in state["markov_probs"]]
        freqs     = state["freq_table"].copy()
        avg_confs = {g: list(c) for g, c in state["gesture_avg_conf"].items()}

    entropy             = 0.0
    next_pred           = None
    next_pr             = 0.0
    session_transitions = sum(sum(row) for row in counts)

    if seq:
        pi = GESTURE_TO_IDX.get(seq[-1])
        ci = GESTURE_TO_IDX.get(gesture)
        if pi is not None and ci is not None:
            counts[pi][ci]     += 1
            session_transitions += 1
            row   = counts[pi]
            total = sum(row)
            if total:
                probs[pi] = [c / total for c in row]
                ps        = [p for p in probs[pi] if p > 0]
                entropy   = round(-sum(p * math.log2(p) for p in ps), 3)

    seq.append(gesture)
    freqs[gesture] += 1
    avg_confs[gesture].append(conf)

    # Blended prediction: Markov session data + linguistic prior
    next_pred, next_pr = _blended_next(gesture, probs, session_transitions)

    with lock:
        state["sequence"]             = seq
        state["markov_counts"]        = counts
        state["markov_probs"]         = probs
        state["freq_table"]           = freqs
        state["gesture_avg_conf"]     = avg_confs
        state["distribution_entropy"] = entropy
        state["next_prediction"]      = next_pred
        state["next_prob"]            = next_pr


def _record(gesture, conf):
    threading.Thread(target=_async_record_worker, args=(gesture, conf), daemon=True).start()


# ── Camera buffers ──────────────────────────────────────────────────────────
raw_frame  = None
raw_lock   = threading.Lock()
jpeg_frame = None
jpeg_lock  = threading.Lock()


def capture_loop():
    global raw_frame
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS,          30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01)
            continue
        frame = cv2.flip(frame, 1)
        with raw_lock:
            raw_frame = frame
        time.sleep(0.01)


def predict_loop():
    global jpeg_frame

    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    tf_model          = None
    tf_scaler         = None
    model_ok          = False
    expected_features = None

    try:
        tf_model = tf.keras.models.load_model("isl_gesture_model.keras")
        with open("scaler.pkl", "rb") as f:
            tf_scaler = pickle.load(f)

        expected_features = tf_scaler.n_features_in_

        @tf.function(jit_compile=True)
        def optimized_predict(tensor_input):
            return tf_model(tensor_input, training=False)

        dummy = tf.zeros((1, expected_features, 1), dtype=tf.float32)
        _     = optimized_predict(dummy).numpy()
        print(f"✅ Fast Inference Engine Active. ({expected_features} features)")
        model_ok = True

    except Exception as e:
        print(f"⚠️  Model load failed: {e}. Running in demo mode.")

    mp_h  = mp.solutions.hands
    mp_dr = mp.solutions.drawing_utils
    det   = mp_h.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    )

    hold = {"g": None, "t": 0.0}
    import random

    while True:
        with raw_lock:
            frame = raw_frame
        if frame is None:
            time.sleep(0.01)
            continue

        frame = frame.copy()
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res   = det.process(rgb)
        rgb.flags.writeable = True

        lm_flat  = []
        detected = 0

        if res.multi_hand_landmarks:
            detected = len(res.multi_hand_landmarks)
            for hl in res.multi_hand_landmarks:
                mp_dr.draw_landmarks(
                    frame, hl, mp_h.HAND_CONNECTIONS,
                    mp_dr.DrawingSpec(color=(180,130,255), thickness=2, circle_radius=3),
                    mp_dr.DrawingSpec(color=(255,200,120), thickness=2),
                )
                for lm in hl.landmark:
                    lm_flat.extend([lm.x, lm.y, lm.z])

        if detected == 1:
            lm_flat.extend([0.0] * 63)

        gesture_label = None
        conf          = 0.0

        if detected in [1, 2]:
            if model_ok:
                try:
                    stats    = extract_statistical_features(lm_flat)
                    full_row = lm_flat + stats

                    if len(full_row) != expected_features:
                        if len(full_row) > expected_features:
                            full_row = full_row[:expected_features]
                        else:
                            full_row += [0.0] * (expected_features - len(full_row))

                    arr       = np.array(full_row).reshape(1, -1)
                    scaled    = tf_scaler.transform(arr)
                    cnn_in    = scaled.reshape(scaled.shape[0], scaled.shape[1], 1)
                    tensor_in = tf.convert_to_tensor(cnn_in, dtype=tf.float32)

                    pred = optimized_predict(tensor_in).numpy()
                    cls  = int(np.argmax(pred))
                    conf = float(pred[0][cls])

                    if conf > 0.75:
                        gesture_label = GESTURES[cls]

                    time.sleep(0.001)

                except Exception as ex:
                    print(f"❌ Predict error: {ex}")
            else:
                if random.random() < 0.012:
                    gesture_label = random.choice(list(GESTURES.values()))
                    conf          = random.uniform(0.76, 0.99)

        if gesture_label:
            now = time.time()
            if hold["g"] == gesture_label:
                if now - hold["t"] >= HOLD_SEC:
                    _record(gesture_label, conf)
                    hold["t"] = now + 9999
            else:
                hold = {"g": gesture_label, "t": time.time()}
        else:
            hold = {"g": None, "t": 0.0}

        with lock:
            state["current_gesture"] = gesture_label
            state["confidence"]      = round(conf * 100, 1)
            state["confidence_history"].append(round(conf * 100, 1))

        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, h-46), (w, h), (8, 8, 12), -1)
        label = f"{gesture_label}  {conf*100:.0f}%" if gesture_label else "waiting..."
        color = (180, 255, 180) if gesture_label else (100, 100, 140)
        cv2.putText(frame, label, (14, h-14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

        ret, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
        if ret:
            with jpeg_lock:
                jpeg_frame = buf.tobytes()

        time.sleep(0.033)


threading.Thread(target=capture_loop, daemon=True).start()
threading.Thread(target=predict_loop, daemon=True).start()

app = Flask(__name__)


def gen_frames():
    while True:
        with jpeg_lock:
            f = jpeg_frame
        if f:
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + f + b'\r\n'
        time.sleep(0.033)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/state")
def api_state():
    with lock:
        avg = {g: round(sum(c) / len(c) * 100, 1) if c else 0.0
               for g, c in state["gesture_avg_conf"].items()}
        return jsonify({
            "current_gesture":      state["current_gesture"],
            "confidence":           state["confidence"],
            "sequence":             state["sequence"][-30:],
            "freq_table":           state["freq_table"],
            "next_prediction":      state["next_prediction"],
            "next_prob":            state["next_prob"],
            "sentence":             state["sentence"],
            "translating":          state["translating"],
            "distribution_entropy": state["distribution_entropy"],
            "confidence_history":   list(state["confidence_history"]),
            "avg_confidence":       avg,
            "session_duration":     round(time.time() - state["session_start"]),
            "total_gestures":       len(state["sequence"]),
            "markov_probs":         state["markov_probs"],
        })


@app.route("/api/translate", methods=["POST"])
def translate():
    if not GEMINI_API_KEY:
        return jsonify({"error": "No GEMINI_API_KEY in .env"}), 400
    with lock:
        seq = list(state["sequence"])
    if not seq:
        return jsonify({"error": "No gestures yet"}), 400
    with lock:
        state["translating"] = True

    def go():
        try:
            gemini = genai.GenerativeModel("gemini-2.5-flash")
            resp   = gemini.generate_content(
                f"These are Indian Sign Language gestures signed in order: [{', '.join(seq)}].\n"
                "Convert them into a single natural, grammatically correct English sentence. "
                "Reply with only the sentence, nothing else."
            )
            with lock:
                state["sentence"]    = resp.text.strip()
                state["translating"] = False
        except Exception as e:
            with lock:
                state["sentence"]    = f"[Error: {e}]"
                state["translating"] = False

    threading.Thread(target=go, daemon=True).start()
    return jsonify({"status": "translating"})


@app.route("/api/reset", methods=["POST"])
def reset():
    with lock:
        state["sequence"].clear()
        state["freq_table"]           = {g: 0  for g in GESTURES.values()}
        state["markov_counts"]        = [[0]   * N for _ in range(N)]
        state["markov_probs"]         = [[0.0] * N for _ in range(N)]
        state["next_prediction"]      = None
        state["next_prob"]            = 0.0
        state["sentence"]             = ""
        state["distribution_entropy"] = 0.0
        state["confidence_history"].clear()
        state["gesture_avg_conf"]     = {g: [] for g in GESTURES.values()}
        state["session_start"]        = time.time()
    return jsonify({"status": "reset"})


@app.route("/api/undo", methods=["POST"])
def undo():
    with lock:
        if state["sequence"]:
            r = state["sequence"].pop()
            if state["freq_table"][r] > 0:
                state["freq_table"][r] -= 1
    return jsonify({"status": "undone"})


@app.route("/api/clear_sequence", methods=["POST"])
def clear_sequence():
    with lock:
        state["sequence"].clear()
        state["sentence"]        = ""
        state["next_prediction"] = None
        state["next_prob"]       = 0.0
        # freq_table, markov_counts, markov_probs, gesture_avg_conf intentionally preserved
    return jsonify({"status": "sequence_cleared"})


@app.route("/api/delete_at", methods=["POST"])
def delete_at():
    data = request.get_json()
    idx  = data.get("index", -1)
    with lock:
        seq = state["sequence"]
        if 0 <= idx < len(seq):
            removed = seq.pop(idx)
            if state["freq_table"].get(removed, 0) > 0:
                state["freq_table"][removed] -= 1
            return jsonify({"status": "deleted", "removed": removed})
    return jsonify({"error": "invalid index"}), 400


if __name__ == "__main__":
    try:
        from waitress import serve
        print("\n  ISL Dashboard → http://localhost:5000\n")
        serve(app, host="0.0.0.0", port=5000, threads=8)
    except ImportError:
        app.run(debug=False, threaded=True, port=5000)