import cv2
import mediapipe as mp
import numpy as np
import pickle
import threading
import time
import collections
import math
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from flask import Flask, Response, render_template, jsonify, request
from anthropic import Anthropic

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

GESTURES = {
    0:"Me", 1:"Hello", 2:"Please", 3:"You", 4:"Sorry",
    5:"House", 6:"Thank you", 7:"Time", 8:"Food", 9:"See"
}
N              = len(GESTURES)
GESTURE_TO_IDX = {v:k for k,v in GESTURES.items()}
HOLD_SEC       = 0.8

# ── Shared App State ──────────────────────────────────────────────────────────
lock  = threading.Lock()
state = {
    "current_gesture":      None,
    "confidence":           0.0,
    "sequence":             [],
    "freq_table":           {g:0   for g in GESTURES.values()},
    "markov_counts":        [[0]  *N for _ in range(N)],
    "markov_probs":         [[0.0]*N for _ in range(N)],
    "next_prediction":      None,
    "next_prob":            0.0,
    "sentence":             "",
    "translating":          False,
    "distribution_entropy": 0.0,
    "confidence_history":   collections.deque(maxlen=60),
    "gesture_avg_conf":     {g:[] for g in GESTURES.values()},
    "session_start":        time.time(),
}

# FIXED: Heavy math calculation moved completely to a non-blocking background task
def _async_record_worker(gesture, conf):
    # 1. Do the heavy string and array lifting completely unlocked
    with lock:
        seq = list(state["sequence"])
        counts = [row[:] for row in state["markov_counts"]]
        probs = [row[:] for row in state["markov_probs"]]
        freqs = state["freq_table"].copy()
        avg_confs = {g: list(c) for g, c in state["gesture_avg_conf"].items()}

    # 2. Compute statistics updates asynchronously safely 
    entropy = 0.0
    next_pred = None
    next_pr = 0.0
    
    if seq:
        pi = GESTURE_TO_IDX.get(seq[-1])
        ci = GESTURE_TO_IDX.get(gesture)
        if pi is not None and ci is not None:
            counts[pi][ci] += 1
            row = counts[pi]
            total = sum(row)
            if total:
                probs[pi] = [c/total for c in row]
                ps = [p for p in probs[pi] if p > 0]
                entropy = round(-sum(p * math.log2(p) for p in ps), 3)
                
    seq.append(gesture)
    freqs[gesture] += 1
    avg_confs[gesture].append(conf)
    
    ci = GESTURE_TO_IDX.get(gesture)
    if ci is not None:
        row = probs[ci]
        if any(p > 0 for p in row):
            bi = int(np.argmax(row))
            next_pred = GESTURES[bi]
            next_pr = round(row[bi] * 100, 1)

    # 3. Quickly write the fully compiled result back to the main state thread
    with lock:
        state["sequence"] = seq
        state["markov_counts"] = counts
        state["markov_probs"] = probs
        state["freq_table"] = freqs
        state["gesture_avg_conf"] = avg_confs
        state["distribution_entropy"] = entropy
        state["next_prediction"] = next_pred
        state["next_prob"] = next_pr

def _record(gesture, conf):
    # Offload execution instantly to a pool worker thread so predict_loop never lags
    threading.Thread(target=_async_record_worker, args=(gesture, conf), daemon=True).start()

# ── Inter-Thread Double Buffers ───────────────────────────────────────────────
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

    tf_model  = None
    tf_scaler = None
    model_ok  = False
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
        _ = optimized_predict(dummy).numpy()
        print(f"✅ Fast Inference Engine Active.")
        model_ok = True

    except Exception as e:
        print(f"⚠️ Model load failed: {e}. Running in demo mode.")

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
                    print(f"❌ Predict runtime break: {ex}")
            else:
                if random.random() < 0.012:
                    gesture_label = random.choice(list(GESTURES.values()))
                    conf          = random.uniform(0.76, 0.99)

        # FIXED DEBOUNCE: Thread safety separation
        if gesture_label:
            now = time.time()
            if hold["g"] == gesture_label:
                if now - hold["t"] >= HOLD_SEC:
                    # Request async append without pausing operations
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

# ── Flask Server ──────────────────────────────────────────────────────────────
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
        avg = {g: round(sum(c)/len(c)*100, 1) if c else 0.0
               for g, c in state["gesture_avg_conf"].items()}
        return jsonify({
            "current_gesture":      state["current_gesture"],
            "confidence":           state["confidence"],
            "sequence":             state["sequence"][-20:],
            "freq_table":           state["freq_table"],
            "next_prediction":      state["next_prediction"],
            "next_prob":            state["next_prob"],
            "sentence":             state["sentence"],
            "translating":          state["translating"],
            "distribution_entropy": state["distribution_entropy"],
            "confidence_history":   list(state["confidence_history"]),
            "avg_confidence":       avg,
            "session_duration":     round(time.time()-state["session_start"]),
            "total_gestures":       len(state["sequence"]),
            "markov_probs":         state["markov_probs"],
        })

@app.route("/api/translate", methods=["POST"])
def translate():
    data = request.get_json()
    key  = data.get("api_key", "").strip()
    if not key: return jsonify({"error": "No API key"}), 400
    with lock:
        seq = list(state["sequence"])
    if not seq: return jsonify({"error": "No gestures yet"}), 400
    with lock:
        state["translating"] = True
    def go():
        try:
            msg = Anthropic(api_key=key).messages.create(
                model="claude-3-5-sonnet-20241022", max_tokens=200,
                messages=[{"role": "user", "content": (
                    f"ISL gestures in order: [{', '.join(seq)}].\n"
                    "Write one natural English sentence. Reply with only the sentence."
                )}])
            with lock:
                state["sentence"]    = msg.content[0].text.strip()
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
        state["markov_counts"]        = [[0]  *N for _ in range(N)]
        state["markov_probs"]         = [[0.0]*N for _ in range(N)]
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

if __name__ == "__main__":
    app.run(debug=False, threaded=True, port=5000)