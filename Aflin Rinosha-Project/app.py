"""
LPR Dashboard - Flask Backend (v14)
Changes over v13.2.1:
  Signal Jump Detection (v14):
  - Imports SignalJumpDetector from signal_jump.py
  - Global _signal_detector instance (one per server process)
  - detect_image route: calls detect_light() on full frame, then
    check_signal_jump(y2) per vehicle bounding box
  - process_video_thread: same integration — per-frame light update,
    per-vehicle jump check before saving violation
  - New violation type: "signal_jumping" (applies to car, bike, truck)
  - GET /signal_config  — returns current detector config as JSON
  - POST /signal_config — updates roi, stop_line_y, tolerance,
                          min_lit_pixels, enabled at runtime
  - RAG chatbot context updated to include signal_jumping violation stats
  Nothing else has been altered.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import cv2
import easyocr
import numpy as np
import base64
import os
import threading
import uuid
import json
import sqlite3
import re
import time
import traceback
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
from difflib import SequenceMatcher

from google import genai
from dotenv import load_dotenv

# ── NEW v14 ──────────────────────────────────────────────────
from signal_jump import SignalJumpDetector
# ─────────────────────────────────────────────────────────────

load_dotenv()

# ─────────────────────────────────────────────────────────────
# AUTO BASE DIR
# ─────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best.pt"
DB_PATH    = str(BASE_DIR / "lpr_database(4).db")
COCO_PATH  = str(BASE_DIR / "yolov8n.pt")

print(f"[INIT] Folder : {BASE_DIR}")
print(f"[INIT] DB     : {DB_PATH}")

# ─────────────────────────────────────────────────────────────
# API KEY SANITY CHECKS
# ─────────────────────────────────────────────────────────────
if not os.getenv("GEMINI_API_KEY"):
    print("[WARN] ⚠  GEMINI_API_KEY not set — plate correction will always fail!")
else:
    print("[INIT] GEMINI_API_KEY : OK")

app = Flask(__name__, static_folder=str(BASE_DIR))
CORS(app)

# ─────────────────────────────────────────────────────────────
# Load models
# ─────────────────────────────────────────────────────────────
print("[INIT] Loading plate model...")
plate_model = YOLO(str(MODEL_PATH))

print("[INIT] Loading COCO model...")
coco_model  = YOLO(COCO_PATH)

print("[INIT] Loading EasyOCR...")
reader = easyocr.Reader(['en'], gpu=False)

_test_img = np.zeros((480, 640, 3), dtype=np.uint8)
plate_model.predict(source=_test_img, conf=0.05, verbose=False)
print("[INIT] Plate model sanity test: OK")

# ─────────────────────────────────────────────────────────────
# LOCAL HELMET MODEL  (replaces Roboflow API)
# ─────────────────────────────────────────────────────────────
HELMET_MODEL_PATH = r"C:\Users\ADMIN\OneDrive\Desktop\New folder\helmet_yolov8n.pt"
print("[INIT] Loading helmet model...")
helmet_model = YOLO(HELMET_MODEL_PATH)
print(f"[INIT] Helmet model loaded — classes: {helmet_model.names}")

gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

COCO_CLASSES = {2: "car", 3: "bike", 5: "truck", 7: "truck"}

# ─────────────────────────────────────────────────────────────
# Indian plate constants
# ─────────────────────────────────────────────────────────────
INDIAN_STATE_CODES = {
    "AN","AP","AR","AS","BR","CG","CH","DD","DL","DN",
    "GA","GJ","HP","HR","JH","JK","KA","KL","LA","LD",
    "MH","ML","MN","MP","MZ","NL","OD","PB","PY","RJ",
    "SK","TN","TR","TS","UK","UP","WB"
}
PLATE_RE = re.compile(r'^([A-Z]{2})(\d{2})([A-Z]{1,3})(\d{4})$')

# ─────────────────────────────────────────────────────────────
# GEMINI MODEL CASCADE
# ─────────────────────────────────────────────────────────────
GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
]

def _gemini_retry(make_call):
    last_exc = None
    for model in GEMINI_MODELS:
        try:
            result = make_call(model)
            if model != GEMINI_MODELS[0]:
                print(f"[GEMINI] ✓ succeeded with fallback model: {model}")
            return result
        except Exception as e:
            err = str(e)
            if '429' in err or 'RESOURCE_EXHAUSTED' in err:
                print(f"[GEMINI] {model} quota hit → trying next model")
                last_exc = e
                continue
            raise
    raise last_exc or Exception("All Gemini models exhausted")

# ─────────────────────────────────────────────────────────────
# LOCAL PLATE CORRECTOR
# ─────────────────────────────────────────────────────────────
_OCR_SUBS = [
    ('O','0'), ('0','O'),
    ('I','1'), ('1','I'),
    ('B','8'), ('8','B'),
    ('S','5'), ('5','S'),
    ('Z','2'), ('2','Z'),
    ('G','6'), ('6','G'),
    ('Q','0'), ('D','0'),
    ('T','1'), ('L','1'),
    ('A','4'), ('H','4'),
    ('U','0'), ('V','0'),
]

def local_correct_plate(raw):
    c = re.sub(r'[^A-Z0-9]', '', raw.upper())
    if len(c) < 6:
        return None
    if is_valid_indian_plate(c):
        return c
    for idx in range(len(c)):
        for old, new in _OCR_SUBS:
            if c[idx] == old:
                variant = c[:idx] + new + c[idx+1:]
                if is_valid_indian_plate(variant):
                    print(f"[LOCAL CORRECT] '{c}' pos {idx} {old}→{new} = '{variant}'")
                    return variant
    return None

# ─────────────────────────────────────────────────────────────
# DB
# ─────────────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS crossings (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            plate         TEXT    NOT NULL,
            vehicle_type  TEXT    DEFAULT 'car',
            crossing_time TEXT    NOT NULL,
            source_path   TEXT,
            source_type   TEXT,
            session_id    TEXT,
            created_at    TEXT    DEFAULT (datetime('now','localtime'))
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS violations (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            plate          TEXT    NOT NULL,
            vehicle_type   TEXT    DEFAULT 'bike',
            violation_type TEXT    DEFAULT 'no_helmet',
            crossing_time  TEXT    NOT NULL,
            source_path    TEXT,
            source_type    TEXT,
            session_id     TEXT,
            created_at     TEXT    DEFAULT (datetime('now','localtime'))
        )
    """)
    conn.commit()
    crossings  = conn.execute("SELECT COUNT(*) FROM crossings").fetchone()[0]
    violations = conn.execute("SELECT COUNT(*) FROM violations").fetchone()[0]
    conn.close()
    print(f"[DB] Crossings: {crossings} | Violations: {violations}")

init_db()
video_jobs = {}

# ─────────────────────────────────────────────────────────────
# NEW v14 — GLOBAL SIGNAL JUMP DETECTOR INSTANCE
# Configure roi and stop_line_y to match your camera layout.
# These can be updated at runtime via POST /signal_config.
# ─────────────────────────────────────────────────────────────
_signal_detector = SignalJumpDetector(
    roi          = (1100, 20, 1260, 180),   # top-right traffic light box
    stop_line_y  = 400,                     # horizontal y of stop line
    tolerance    = 10,
    min_lit_pixels = 40,
    enabled      = True,
)
print(f"[INIT] SignalJumpDetector ready — stop_line_y={_signal_detector.stop_line_y}  roi={_signal_detector.roi}")

# ─────────────────────────────────────────────────────────────
# VALIDATORS
# ─────────────────────────────────────────────────────────────
def is_valid_indian_plate(text):
    cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
    m = PLATE_RE.match(cleaned)
    return bool(m) and m.group(1) in INDIAN_STATE_CODES

# ─────────────────────────────────────────────────────────────
# HELMET DETECTION
# ─────────────────────────────────────────────────────────────
def check_helmet(frame, bike_box):
    try:
        vx1, vy1, vx2, vy2 = bike_box
        fh, fw = frame.shape[:2]
        head_y2   = vy1 + int((vy2 - vy1) * 0.55)
        head_crop = frame[
            max(0, vy1):min(fh, head_y2),
            max(0, vx1):min(fw, vx2)
        ]
        if head_crop.size == 0:
            return "unknown"

        results = helmet_model.predict(source=head_crop, conf=0.30, verbose=False)

        if not results or len(results[0].boxes) == 0:
            return "no_helmet"

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            label  = helmet_model.names[cls_id].lower()
            print(f"  [HELMET] detected class='{label}'  conf={float(box.conf[0]):.2f}")
            if "helmet" in label and \
               "no"      not in label and \
               "without" not in label and \
               "non"     not in label:
                return "helmet"

        return "no_helmet"

    except Exception as e:
        print(f"[HELMET ERROR] {e}")
        return "unknown"

# ─────────────────────────────────────────────────────────────
# DETECT ALL VEHICLES + PLATES
# ─────────────────────────────────────────────────────────────
def detect_all_vehicles_and_plates(frame):
    fh, fw     = frame.shape[:2]
    found      = []
    seen_boxes = []

    vehicle_boxes = []
    try:
        coco_results = coco_model.predict(
            source=frame, conf=0.25, verbose=False,
            classes=list(COCO_CLASSES.keys())
        )
        for box in coco_results[0].boxes:
            cls_id = int(box.cls[0])
            if cls_id not in COCO_CLASSES:
                continue
            vx1, vy1, vx2, vy2 = map(int, map(float, box.xyxy[0]))
            vx1 = max(0, vx1); vy1 = max(0, vy1)
            vx2 = min(fw, vx2); vy2 = min(fh, vy2)
            vehicle_boxes.append((vx1, vy1, vx2, vy2, COCO_CLASSES[cls_id]))
        print(f"[COCO] {len(vehicle_boxes)} vehicle(s)")
    except Exception as e:
        print(f"[COCO ERROR] {e}")

    def _run_and_collect(src, offset_x, offset_y, conf, vtype, vbox):
        if src is None or src.size == 0:
            return
        try:
            res = plate_model.predict(source=src, conf=conf, verbose=False)
            for pb in res[0].boxes.xyxy:
                bx1, by1, bx2, by2 = map(int, map(float, pb))
                fx1 = max(0, offset_x + bx1); fy1 = max(0, offset_y + by1)
                fx2 = min(fw, offset_x + bx2); fy2 = min(fh, offset_y + by2)
                if (fx2 - fx1) > 0 and (fy2 - fy1) > 0:
                    is_dup = any(
                        abs(sb[0] - fx1) < 30 and abs(sb[1] - fy1) < 30
                        for sb in seen_boxes
                    )
                    if not is_dup:
                        found.append({"box": (fx1, fy1, fx2, fy2),
                                      "vtype": vtype, "vehicle_box": vbox})
                        seen_boxes.append((fx1, fy1, fx2, fy2))
        except Exception as e:
            print(f"[CROP ERROR] {e}")

    for (vx1, vy1, vx2, vy2, vtype) in vehicle_boxes:
        cx1 = max(0, vx1 - 20); cy1 = max(0, vy1 - 10)
        cx2 = min(fw, vx2 + 20); cy2 = min(fh, vy2 + 80)
        vehicle_crop = frame[cy1:cy2, cx1:cx2]
        if vehicle_crop.size == 0:
            continue
        vh           = cy2 - cy1
        strip_y1     = max(0, cy2 - int(vh * 0.45))
        bottom_strip = frame[strip_y1:cy2, cx1:cx2]
        _run_and_collect(vehicle_crop, cx1, cy1, 0.05, vtype, (vx1, vy1, vx2, vy2))
        _run_and_collect(bottom_strip, cx1, strip_y1, 0.03, vtype, (vx1, vy1, vx2, vy2))

    try:
        full_res = plate_model.predict(source=frame, conf=0.07, verbose=False)
        for pb in full_res[0].boxes.xyxy:
            px1, py1, px2, py2 = map(int, map(float, pb))
            px1 = max(0, px1); py1 = max(0, py1)
            px2 = min(fw, px2); py2 = min(fh, py2)
            if (px2 - px1) <= 0 or (py2 - py1) <= 0:
                continue
            is_dup = any(
                abs(sb[0] - px1) < 30 and abs(sb[1] - py1) < 30
                for sb in seen_boxes
            )
            if not is_dup:
                pcx = (px1 + px2) / 2
                pcy = (py1 + py2) / 2
                vtype = "car"
                vbox  = None
                for (vx1, vy1, vx2, vy2, vt) in vehicle_boxes:
                    if vx1 <= pcx <= vx2 and vy1 <= pcy <= vy2:
                        vtype = vt
                        vbox  = (vx1, vy1, vx2, vy2)
                        break
                found.append({"box": (px1, py1, px2, py2),
                              "vtype": vtype, "vehicle_box": vbox})
                seen_boxes.append((px1, py1, px2, py2))
    except Exception as e:
        print(f"[FULL FRAME ERROR] {e}")

    print(f"[DETECT] {len(found)} plate region(s) found")
    return found

# ─────────────────────────────────────────────────────────────
# SIMILARITY
# ─────────────────────────────────────────────────────────────
def are_similar(a, b, threshold=0.5):
    ac = "".join(c for c in a if c.isalnum()).upper()
    bc = "".join(c for c in b if c.isalnum()).upper()
    if len(ac) < 4 or len(bc) < 4:
        return ac == bc
    return SequenceMatcher(None, ac, bc).ratio() >= threshold

# ─────────────────────────────────────────────────────────────
# GEMINI — plate correction (with retry)
# ─────────────────────────────────────────────────────────────
def gemini_analyze_plate(raw_text):
    try:
        prompt = f"""You are an expert Indian license plate OCR corrector.

Indian plate format: XX00XX0000
Examples: TN01AB1234, KA02FY7410, MH12DE3456, DL3CAF1234

Valid state codes: AN,AP,AR,AS,BR,CG,CH,DD,DL,DN,GA,GJ,HP,HR,JH,JK,KA,KL,LA,LD,MH,ML,MN,MP,MZ,NL,OD,PB,PY,RJ,SK,TN,TR,TS,UK,UP,WB

OCR gave this text: {raw_text}

IMPORTANT — Try HARD to correct before rejecting:
- Fix OCR errors aggressively: O<>0, I<>1, B<>8, S<>5, Z<>2, H<>4, G<>6, R<>N, F<>T, D<>0
- Even if only part of the plate is readable, try to form a valid plate
- If you can make a valid Indian plate from this text, do it
- Only return REJECT if the text is completely random with no plate-like pattern at all
- Partial plates or slightly wrong formats — still try to fix them

Return ONLY this JSON, no markdown:
{{"plate":"CORRECTED OR REJECT","state":"full state name OR Unknown","valid":true/false}}"""

        res = _gemini_retry(
            lambda model: gemini_client.models.generate_content(
                model=model, contents=prompt)
        )
        clean = res.text.strip().replace("```json", "").replace("```", "").strip()
        data  = json.loads(clean)

        plate = data.get("plate", "REJECT").strip().upper()
        plate = re.sub(r'[^A-Z0-9]', '', plate)

        if plate == "REJECT" or len(plate) < 6:
            return None

        valid = data.get("valid", False)
        if isinstance(valid, str):
            valid = valid.strip().lower() == "true"
        if not valid:
            return None

        if not is_valid_indian_plate(plate):
            return None

        return {"plate": plate, "state": data.get("state", "Unknown")}

    except Exception as e:
        print(f"[GEMINI CORRECT ERROR] {e}")
        plate = local_correct_plate(raw_text)
        if plate:
            print(f"[LOCAL FALLBACK] '{raw_text}' → '{plate}'")
            return {"plate": plate, "state": "Unknown"}
        cleaned = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
        if is_valid_indian_plate(cleaned):
            return {"plate": cleaned, "state": "Unknown"}
        return None

# ─────────────────────────────────────────────────────────────
# GEMINI VISION (with retry)
# ─────────────────────────────────────────────────────────────
def gemini_vision_fallback(frame):
    try:
        _, buf = cv2.imencode('.jpg', frame)
        b64    = base64.b64encode(buf).decode('utf-8')

        contents = [{"parts": [
            {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
            {"text": (
                "List every Indian license plate you can see in this image. "
                "One per line, raw text exactly as it appears on the plate. "
                "If none are visible, reply: NONE"
            )}
        ]}]

        res = _gemini_retry(
            lambda model: gemini_client.models.generate_content(
                model=model, contents=contents)
        )
        text = res.text.strip()
        print(f"[GEMINI VISION RAW] {repr(text)}")

        if not text or text.upper() == "NONE":
            return []

        plates = []
        for line in text.splitlines():
            raw = line.strip().upper()
            if not raw or raw == "NONE":
                continue
            corrected = gemini_analyze_plate(raw)
            if corrected:
                plates.append(corrected["plate"])
            else:
                cleaned = re.sub(r'[^A-Z0-9]', '', raw)
                if is_valid_indian_plate(cleaned):
                    plates.append(cleaned)

        print(f"[GEMINI VISION] Corrected plates: {plates}")
        return plates

    except Exception as e:
        print(f"[GEMINI VISION ERROR] {e}")
        return []

# ─────────────────────────────────────────────────────────────
# MULTI-VARIANT PREPROCESSING
# ─────────────────────────────────────────────────────────────
def preprocess_variants(crop):
    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return []

    scale = min(max(4, int(120 / max(h, 1))), 10)
    up    = cv2.resize(crop, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_LANCZOS4)

    gray     = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=8)
    clahe    = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4)).apply(denoised)
    sharp    = cv2.filter2D(clahe, -1, np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]))

    _, otsu = cv2.threshold(sharp, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adapt   = cv2.adaptiveThreshold(sharp, 255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)

    return [
        gray,
        clahe,
        sharp,
        otsu,
        cv2.bitwise_not(otsu),
        adapt,
    ]

# ─────────────────────────────────────────────────────────────
# OCR PLATE — score by char_count × confidence
# ─────────────────────────────────────────────────────────────
def ocr_plate(crop):
    kw_normal = dict(
        allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
        detail=1, paragraph=False,
        width_ths=0.4, contrast_ths=0.05, adjust_contrast=0.7
    )
    kw_loose = dict(
        allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
        detail=1, paragraph=False,
        width_ths=0.2, contrast_ths=0.01, adjust_contrast=0.9,
        text_threshold=0.4, low_text=0.3
    )

    variants = preprocess_variants(crop)
    if not variants:
        return []

    best_result = []
    best_score  = -1

    for img in variants:
        for kw in [kw_normal, kw_loose]:
            try:
                result = reader.readtext(img, **kw)
                if not result:
                    continue
                text  = "".join(r[1] for r in result).upper()
                chars = len(re.sub(r'[^A-Z0-9]', '', text))
                avg_c = float(np.mean([r[2] for r in result]))
                score = chars * avg_c
                if score > best_score:
                    best_score  = score
                    best_result = result
            except Exception:
                continue

    if best_result:
        text = "".join(r[1] for r in best_result)
        print(f"  [OCR BEST] score={best_score:.2f}  text='{text}'")

    return best_result

# ─────────────────────────────────────────────────────────────
# DB HELPERS
# ─────────────────────────────────────────────────────────────
def save_crossing(plate, vtype, crossing_time,
                  source_path, source_type, session_id):
    try:
        conn = get_db()
        conn.execute("""
            INSERT INTO crossings
              (plate, vehicle_type, crossing_time,
               source_path, source_type, session_id)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (plate, vtype, crossing_time,
              source_path, source_type, session_id))
        conn.commit()
        conn.close()
        print(f"[SAVED] {plate} | {vtype} | {crossing_time}")
        return True
    except Exception as e:
        print(f"[DB ERROR] {e}")
        return False

def save_violation(plate, vtype, violation_type, crossing_time,
                   source_path, source_type, session_id):
    try:
        conn = get_db()
        conn.execute("""
            INSERT INTO violations
              (plate, vehicle_type, violation_type, crossing_time,
               source_path, source_type, session_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (plate, vtype, violation_type, crossing_time,
              source_path, source_type, session_id))
        conn.commit()
        conn.close()
        print(f"[VIOLATION] {plate} | {violation_type} | {crossing_time}")
        return True
    except Exception as e:
        print(f"[VIOLATION DB ERROR] {e}")
        return False

def already_saved(plate, session_id):
    conn = get_db()
    row  = conn.execute(
        "SELECT id FROM crossings WHERE plate=? AND session_id=?",
        (plate, session_id)
    ).fetchone()
    conn.close()
    return row is not None


# ─────────────────────────────────────────────────────────────
# RAG CHATBOT — v14 (signal_jumping stats added to context)
# ─────────────────────────────────────────────────────────────

def _rag_fetch_context(question: str) -> str:
    conn = get_db()
    today = datetime.now().strftime('%Y-%m-%d')
    q_up  = question.upper()
    facts  = []

    try:
        # ── Core summary stats ──
        total_c  = conn.execute("SELECT COUNT(*) FROM crossings").fetchone()[0]
        total_v  = conn.execute("SELECT COUNT(*) FROM violations").fetchone()[0]
        today_c  = conn.execute(
            "SELECT COUNT(*) FROM crossings WHERE created_at LIKE ?",
            (today + '%',)).fetchone()[0]
        today_v  = conn.execute(
            "SELECT COUNT(*) FROM violations WHERE created_at LIKE ?",
            (today + '%',)).fetchone()[0]
        vtype_rows = conn.execute(
            "SELECT vehicle_type, COUNT(*) as cnt FROM crossings GROUP BY vehicle_type"
        ).fetchall()
        vtype_summary = ", ".join(
            f"{r['vehicle_type']}:{r['cnt']}" for r in vtype_rows
        )

        facts.append(f"Total crossings all-time: {total_c}")
        facts.append(f"Today's crossings: {today_c}")
        facts.append(f"Total violations all-time: {total_v}")
        facts.append(f"Today's violations: {today_v}")
        facts.append(f"Vehicle type breakdown: {vtype_summary}")

        # ── Violation breakdown (includes signal_jumping) ──
        viol_types = conn.execute(
            "SELECT violation_type, COUNT(*) as cnt FROM violations GROUP BY violation_type"
        ).fetchall()
        for r in viol_types:
            facts.append(f"Violation type '{r['violation_type']}': {r['cnt']} total")

        today_viol_types = conn.execute(
            "SELECT violation_type, COUNT(*) as cnt FROM violations "
            "WHERE created_at LIKE ? GROUP BY violation_type",
            (today + '%',)
        ).fetchall()
        for r in today_viol_types:
            facts.append(f"Today's '{r['violation_type']}' violations: {r['cnt']}")

        # ── Signal jump specific query ──
        if any(w in q_up for w in ['SIGNAL', 'JUMP', 'RED LIGHT', 'TRAFFIC LIGHT']):
            sj_total = conn.execute(
                "SELECT COUNT(*) FROM violations WHERE violation_type='signal_jump'"
            ).fetchone()[0]
            sj_today = conn.execute(
                "SELECT COUNT(*) FROM violations WHERE violation_type='signal_jump' AND created_at LIKE ?",
                (today + '%',)
            ).fetchone()[0]
            sj_by_type = conn.execute(
                "SELECT vehicle_type, COUNT(*) as cnt FROM violations "
                "WHERE violation_type='signal_jump' GROUP BY vehicle_type"
            ).fetchall()
            facts.append(f"Signal jumping violations all-time: {sj_total}")
            facts.append(f"Signal jumping violations today: {sj_today}")
            for r in sj_by_type:
                facts.append(f"  Signal jump by {r['vehicle_type']}: {r['cnt']}")

        # ── Plate lookup ──
        plate_match = re.search(r'\b([A-Z]{2}\d{2}[A-Z]{1,3}\d{4})\b', q_up)
        if plate_match:
            plate = plate_match.group(1)
            crossings = conn.execute(
                "SELECT crossing_time, created_at, vehicle_type, source_type "
                "FROM crossings WHERE plate=? ORDER BY created_at DESC",
                (plate,)
            ).fetchall()
            if crossings:
                facts.append(f"Plate {plate} crossed {len(crossings)} time(s).")
                for c in crossings[:5]:
                    facts.append(
                        f"  → {c['created_at']} at {c['crossing_time']} "
                        f"({c['vehicle_type']}, via {c['source_type']})"
                    )
            else:
                facts.append(f"Plate {plate}: no crossing record found.")

            viol_rows = conn.execute(
                "SELECT violation_type, crossing_time, created_at "
                "FROM violations WHERE plate=? ORDER BY created_at DESC",
                (plate,)
            ).fetchall()
            if viol_rows:
                facts.append(f"Plate {plate} has {len(viol_rows)} violation(s):")
                for v in viol_rows[:5]:
                    facts.append(
                        f"  → {v['violation_type']} on {v['created_at']} "
                        f"at {v['crossing_time']}"
                    )
            else:
                facts.append(f"Plate {plate}: no violations on record.")

        # ── Time-range query ──
        time_match = re.search(
            r'\b(\d{1,2})(?::\d{2})?\s*(?:to|\u2013|-|and)\s*(\d{1,2})(?::\d{2})?\b',
            question,
            re.IGNORECASE
        )
        if time_match:
            h1 = max(0, min(23, int(time_match.group(1))))
            h2 = max(0, min(23, int(time_match.group(2))))
            t1 = f"{h1:02d}:00"
            t2 = f"{h2:02d}:59"
            rows = conn.execute(
                "SELECT vehicle_type, COUNT(*) as cnt FROM crossings "
                "WHERE crossing_time >= ? AND crossing_time <= ? "
                "GROUP BY vehicle_type",
                (t1, t2)
            ).fetchall()
            if rows:
                breakdown = ", ".join(f"{r['vehicle_type']}: {r['cnt']}" for r in rows)
                total_range = sum(r['cnt'] for r in rows)
                facts.append(
                    f"Crossings between {t1}–{t2}: {total_range} total ({breakdown})"
                )
            else:
                facts.append(f"No crossings recorded between {t1} and {t2}.")

        # ── Date-specific query ──
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', question)
        if date_match:
            qdate = date_match.group(1)
            dc = conn.execute(
                "SELECT COUNT(*) FROM crossings WHERE created_at LIKE ?",
                (qdate + '%',)
            ).fetchone()[0]
            dv = conn.execute(
                "SELECT COUNT(*) FROM violations WHERE created_at LIKE ?",
                (qdate + '%',)
            ).fetchone()[0]
            facts.append(f"On {qdate}: {dc} crossing(s), {dv} violation(s).")

        # ── Most frequent plates ──
        if any(w in q_up for w in ['FREQUENT', 'TOP', 'MOST', 'REPEAT', 'COMMON']):
            top = conn.execute(
                "SELECT plate, vehicle_type, COUNT(*) as cnt FROM crossings "
                "GROUP BY plate ORDER BY cnt DESC LIMIT 5"
            ).fetchall()
            for r in top:
                facts.append(
                    f"Plate {r['plate']} ({r['vehicle_type']}): {r['cnt']} crossing(s)"
                )

        # ── Latest / recent crossings ──
        if any(w in q_up for w in ['LATEST', 'RECENT', 'LAST', 'NEW']):
            recent = conn.execute(
                "SELECT plate, vehicle_type, crossing_time, created_at "
                "FROM crossings ORDER BY created_at DESC LIMIT 5"
            ).fetchall()
            facts.append("Recent crossings:")
            for r in recent:
                facts.append(
                    f"  {r['plate']} ({r['vehicle_type']}) at "
                    f"{r['crossing_time']} on {r['created_at']}"
                )

    except Exception as e:
        facts.append(f"[DB error: {e}]")
    finally:
        conn.close()

    return "\n".join(facts)


def _rag_answer(question: str) -> str:
    context = _rag_fetch_context(question)
    print(f"[RAG CTX]\n{context}\n")

    prompt = f"""You are a concise traffic-data assistant for an Indian LPR (License Plate Recognition) system.
Answer the user's question using ONLY the database facts provided below.
Rules:
- Reply in 2-3 short sentences maximum (50-60 words).
- Be direct and factual. No greetings, no filler words.
- If the data doesn't contain the answer, say "No data available for that."
- For plate queries, confirm yes/no and give crossing count + date(s).
- For time-range queries, give vehicle-type breakdown.
- For violation queries, state count and type.
- Violation types are: no_helmet (bikes without helmets) and signal_jumping (any vehicle that crossed the stop line on a red light).

DATABASE FACTS (live, pulled right now):
{context}

USER QUESTION: {question}

ANSWER (50-60 words max):"""

    try:
        res = _gemini_retry(
            lambda model: gemini_client.models.generate_content(
                model=model,
                contents=prompt,
                config={"max_output_tokens": 150, "temperature": 0.2}
            )
        )
        answer = res.text.strip()
        words = answer.split()
        if len(words) > 90:
            sentences = re.split(r'(?<=[.!?])\s+', answer)
            trimmed = ""
            for s in sentences:
                if len((trimmed + " " + s).split()) <= 80:
                    trimmed = (trimmed + " " + s).strip()
                else:
                    break
            answer = trimmed if trimmed else " ".join(words[:75]) + "…"
        return answer

    except Exception as e:
        print(f"[RAG ERROR] {e}")
        return _rag_local_fallback(question, context)


def _rag_local_fallback(question: str, context: str) -> str:
    q = question.upper()
    lines = context.splitlines()

    def find(keyword):
        for l in lines:
            if keyword in l.upper():
                return l
        return None

    if 'SIGNAL' in q or 'JUMP' in q:
        l = find("SIGNAL JUMPING VIOLATIONS")
        return l if l else "No signal jumping data available."
    if 'VIOLATION' in q and 'TODAY' in q:
        l = find("TODAY'S VIOLATIONS:")
        return l if l else "No violation data available for today."
    if 'TOTAL' in q and 'VIOLATION' in q:
        l = find("TOTAL VIOLATIONS ALL-TIME:")
        return l if l else "No violation data."
    if 'TODAY' in q and 'CROSSING' in q:
        l = find("TODAY'S CROSSINGS:")
        return l if l else "No crossing data for today."
    if 'TOTAL' in q:
        l = find("TOTAL CROSSINGS ALL-TIME:")
        return l if l else "No data."
    plate_match = re.search(r'\b([A-Z]{2}\d{2}[A-Z]{1,3}\d{4})\b', q)
    if plate_match:
        plate = plate_match.group(1)
        l = find(f"PLATE {plate} CROSSED")
        return l if l else f"No record found for plate {plate}."
    t_match = re.search(r'\b(\d{1,2})\s*(?:to|-|and)\s*(\d{1,2})\b', q)
    if t_match:
        h1 = int(t_match.group(1)); h2 = int(t_match.group(2))
        t1 = f"{h1:02d}:00"; t2 = f"{h2:02d}:59"
        l = find(f"CROSSINGS BETWEEN {t1}")
        return l if l else f"No crossing data found between {t1} and {t2}."
    if any(w in q for w in ['BIKE', 'CAR', 'TRUCK', 'COUNT', 'BREAKDOWN']):
        l = find("VEHICLE TYPE BREAKDOWN:")
        return l if l else "No vehicle breakdown data available."
    if any(w in q for w in ['FREQUENT', 'TOP', 'MOST', 'REPEAT']):
        lines_with_crossings = [l for l in lines if 'CROSSING(S)' in l.upper()]
        if lines_with_crossings:
            return "Top plates: " + "; ".join(lines_with_crossings[:3])
    return "No data found for that query. Please check the dashboard directly."


# ═════════════════════════════════════════════════════════════
# CHAT ROUTE (RAG)
# ═════════════════════════════════════════════════════════════
@app.route('/chat', methods=['POST'])
def chat():
    data     = request.get_json()
    question = (data.get('message') or '').strip()
    if not question:
        return jsonify({"error": "No message provided"}), 400
    try:
        answer = _rag_answer(question)
        return jsonify({"answer": answer})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ═════════════════════════════════════════════════════════════
# NEW v14 — SIGNAL CONFIG ROUTES
# GET  /signal_config  → returns current detector settings
# POST /signal_config  → updates detector settings at runtime
# ═════════════════════════════════════════════════════════════
@app.route('/signal_config', methods=['GET'])
def get_signal_config():
    return jsonify(_signal_detector.get_config())


@app.route('/signal_config', methods=['POST'])
def update_signal_config():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body"}), 400
    try:
        _signal_detector.update_config(data)
        print(f"[SIGNAL CONFIG] Updated: {_signal_detector.get_config()}")
        return jsonify({"ok": True, "config": _signal_detector.get_config()})
    except (ValueError, TypeError) as e:
        return jsonify({"error": str(e)}), 400


# ═════════════════════════════════════════════════════════════
# IMAGE ROUTE  (v14 — signal jump detection added)
# ═════════════════════════════════════════════════════════════
@app.route('/detect_image', methods=['POST'])
def detect_image():
    data          = request.get_json()
    image_path    = data.get('image_path', '').strip()
    crossing_time = data.get('crossing_time',
                             datetime.now().strftime('%H:%M')).strip()

    if not image_path or not os.path.exists(image_path):
        return jsonify({"error": f"File not found: {image_path}"}), 400

    try:
        img = cv2.imread(image_path)
        if img is None:
            return jsonify({"error": "Could not read image"}), 400

        h, w       = img.shape[:2]
        session_id = str(uuid.uuid4())
        saved      = []
        saved_set  = set()

        print(f"\n{'─'*60}")
        print(f"[IMAGE] {image_path}  {w}×{h}  session={session_id[:8]}")

        # ── v14: read traffic light state once for the whole frame ──
        light_color = _signal_detector.detect_light(img)
        print(f"[SIGNAL] Traffic light detected as: {light_color.upper()}")

        detections = detect_all_vehicles_and_plates(img)
        print(f"[IMAGE] Pipeline A: {len(detections)} YOLO plate region(s)")

        for det in detections:
            box   = det["box"]
            vtype = det["vtype"]
            vbox  = det["vehicle_box"]
            x1, y1, x2, y2 = box

            pad  = 20
            crop = img[max(0, y1-pad):min(h, y2+pad),
                       max(0, x1-pad):min(w, x2+pad)]
            if crop.size == 0:
                print(f"  [SKIP] Empty crop at {box}")
                continue

            ocr = ocr_plate(crop)
            raw = "".join([r[1] for r in ocr]).upper() if ocr else ""
            print(f"  [OCR RAW] box={box}  raw='{raw}'")

            result = None
            if raw and len(re.sub(r'[^A-Z0-9]', '', raw)) >= 4:
                result = gemini_analyze_plate(raw)
                if result:
                    print(f"  [GEMINI CORRECT] '{raw}' → {result}")
                else:
                    plate = local_correct_plate(raw)
                    if plate:
                        result = {"plate": plate, "state": "Unknown"}
                        print(f"  [LOCAL CORRECT] '{raw}' → '{plate}'")
            else:
                crop_plates = gemini_vision_fallback(crop)
                result = {"plate": crop_plates[0], "state": "Unknown"} \
                         if crop_plates else None
                print(f"  [CROP VISION FALLBACK] {result}")

            if not result:
                print(f"  [NO PLATE] could not identify plate from crop {box}")
                continue

            plate = result["plate"]
            if plate in saved_set:
                print(f"  [DEDUP] {plate}")
                continue
            saved_set.add(plate)

            save_crossing(plate, vtype, crossing_time,
                          image_path, "image", session_id)
            entry = {"plate": plate, "vehicle_type": vtype,
                     "crossing_time": crossing_time}

            # ── v14: signal jump check using vehicle bounding box y2 ──
            if vbox is not None:
                vehicle_y2 = vbox[3]   # bottom edge of vehicle YOLO box
                if _signal_detector.check_signal_jump(vehicle_y2):
                    save_violation(plate, vtype, "signal_jump",
               crossing_time, image_path, "image", session_id)
                    entry["violation"] = "signal_jump"
                    print(f"  [SIGNAL JUMP] {plate} ({vtype}) crossed on RED at y2={vehicle_y2}")

            # Helmet check for bikes (only if no signal jump already flagged)
            if vtype == "bike" and vbox is not None and "violation" not in entry:
                helmet_status = check_helmet(img, vbox)
                if helmet_status == "no_helmet":
                    save_violation(plate, vtype, "no_helmet",
                                   crossing_time, image_path, "image", session_id)
                    entry["violation"] = "no_helmet"

            saved.append(entry)
            print(f"  [SAVED-A] {plate} ({vtype})")

        print(f"[IMAGE] Pipeline B: Gemini Vision on full image...")
        vision_plates = gemini_vision_fallback(img)
        for pt in vision_plates:
            if pt in saved_set:
                print(f"  [DEDUP] {pt} (already found in Pipeline A)")
                continue
            saved_set.add(pt)
            save_crossing(pt, "car", crossing_time,
                          image_path, "image", session_id)
            saved.append({"plate": pt, "vehicle_type": "car",
                          "crossing_time": crossing_time})
            print(f"  [SAVED-B] {pt} (vision)")

        print(f"[IMAGE] Done — {len(saved)} saved: {[s['plate'] for s in saved]}")
        print('─'*60)

        return jsonify({"session_id": session_id,
                        "plates_saved": len(saved), "plates": saved})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ═════════════════════════════════════════════════════════════
# VIDEO THREAD  (v14 — signal jump detection added)
# ═════════════════════════════════════════════════════════════
def process_video_thread(job_id, video_path, crossing_time):
    job           = video_jobs[job_id]
    job["status"] = "processing"

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            job["status"] = "error"
            job["error"]  = "Cannot open video"
            return

        session_id   = job_id
        tracked      = {}
        saved_plates = set()
        plates_found = []
        frame_count  = 0
        saved_count  = 0
        MIN_FRAMES     = 2
        FALLBACK_EVERY = 10
        SKIP_FRAMES    = 3

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % SKIP_FRAMES != 0:
                frame_count += 1
                continue

            fh, fw   = frame.shape[:2]
            detected = []

            # ── v14: update traffic light state once per processed frame ──
            light_color = _signal_detector.detect_light(frame)

            detections = detect_all_vehicles_and_plates(frame)

            if not detections:
                if frame_count % FALLBACK_EVERY == 0:
                    for pt in gemini_vision_fallback(frame):
                        detected.append({"text": pt, "conf": 0.75,
                                         "vtype": "car", "vbox": None})
            else:
                for det in detections:
                    box   = det["box"]
                    vtype = det["vtype"]
                    vbox  = det["vehicle_box"]
                    x1, y1, x2, y2 = box
                    pad    = 20
                    region = frame[max(0, y1-pad):min(fh, y2+pad),
                                   max(0, x1-pad):min(fw, x2+pad)]
                    if region.size == 0:
                        continue

                    ocr = ocr_plate(region)
                    raw = "".join([r[1] for r in ocr]).upper() if ocr else ""

                    if not raw or len(re.sub(r'[^A-Z0-9]', '', raw)) < 4:
                        crop_plates = gemini_vision_fallback(region)
                        if crop_plates:
                            detected.append({"text": crop_plates[0], "conf": 0.75,
                                             "vtype": vtype, "vbox": vbox})
                        continue

                    result = gemini_analyze_plate(raw)
                    if not result:
                        plate = local_correct_plate(raw)
                        if plate:
                            result = {"plate": plate, "state": "Unknown"}

                    if not result:
                        continue

                    conf = float(np.mean([r[2] for r in ocr])) if ocr else 0.5
                    detected.append({"text": result["plate"], "conf": conf,
                                     "vtype": vtype, "vbox": vbox})

            for det in detected:
                plate = det["text"]
                conf  = det["conf"]
                vtype = det["vtype"]
                vbox  = det.get("vbox")
                match = None
                for tp in tracked:
                    if are_similar(plate, tp):
                        match = tp
                        break

                if match:
                    tracked[match]["frames"] += 1
                    tracked[match]["last"]    = frame_count
                    if conf > tracked[match]["conf"]:
                        tracked[match]["conf"]  = conf
                        tracked[match]["vtype"] = vtype
                        tracked[match]["vbox"]  = vbox

                    # ── v14: check signal jump when frame threshold hit ──
                    if tracked[match]["frames"] == MIN_FRAMES and match not in saved_plates:
                        save_crossing(match, tracked[match]["vtype"],
                                      crossing_time, video_path, "video", session_id)
                        saved_plates.add(match)
                        saved_count += 1
                        entry = {"plate": match,
                                 "vehicle_type": tracked[match]["vtype"],
                                 "crossing_time": crossing_time}

                        # Signal jump check — uses current vbox y2
                        if tracked[match]["vbox"] is not None:
                            vehicle_y2 = tracked[match]["vbox"][3]
                            if _signal_detector.check_signal_jump(vehicle_y2):
                                save_violation(match, tracked[match]["vtype"],
                                                 "signal_jump", crossing_time,
                                                  video_path, "video", session_id)
                                entry["violation"] = "signal_jump"
                                print(f"  [SIGNAL JUMP VIDEO] {match} ({tracked[match]['vtype']}) at y2={vehicle_y2}")

                        # Helmet check (only if no signal jump)
                        if (tracked[match]["vtype"] == "bike"
                                and tracked[match]["vbox"]
                                and "violation" not in entry):
                            hs = check_helmet(frame, tracked[match]["vbox"])
                            if hs == "no_helmet":
                                save_violation(match, "bike", "no_helmet",
                                               crossing_time, video_path,
                                               "video", session_id)
                                entry["violation"] = "no_helmet"

                        plates_found.append(entry)
                else:
                    tracked[plate] = {"conf": conf, "frames": 1,
                                      "last": frame_count,
                                      "vtype": vtype, "vbox": vbox}

            frame_count += 1

        for plate, d in tracked.items():
            if plate not in saved_plates:
                save_crossing(plate, d["vtype"], crossing_time,
                              video_path, "video", session_id)
                saved_plates.add(plate)
                saved_count += 1
                plates_found.append({"plate": plate,
                                     "vehicle_type": d["vtype"],
                                     "crossing_time": crossing_time})

        if saved_count < 2:
            cap2  = cv2.VideoCapture(video_path)
            total = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
            for pos in [int(total * .25), int(total * .5), int(total * .75)]:
                if saved_count >= 2:
                    break
                cap2.set(cv2.CAP_PROP_POS_FRAMES, pos)
                ret, frm = cap2.read()
                if not ret:
                    continue
                for pt in gemini_vision_fallback(frm):
                    if saved_count >= 2:
                        break
                    if pt in saved_plates:
                        continue
                    save_crossing(pt, "car", crossing_time,
                                  video_path, "video", session_id)
                    saved_plates.add(pt)
                    saved_count += 1
                    plates_found.append({"plate": pt, "vehicle_type": "car",
                                         "crossing_time": crossing_time})
            cap2.release()

        cap.release()
        job["status"]  = "done"
        job["results"] = plates_found
        job["stats"]   = {"total_frames": frame_count, "plates_saved": saved_count}

    except Exception as e:
        job["status"] = "error"
        job["error"]  = str(e)
        print(f"[VIDEO ERROR] {e}")
        traceback.print_exc()


# ═════════════════════════════════════════════════════════════
# ROUTES
# ═════════════════════════════════════════════════════════════
@app.route('/process_video', methods=['POST'])
def process_video():
    data          = request.get_json()
    video_path    = data.get('video_path', '').strip()
    crossing_time = data.get('crossing_time',
                             datetime.now().strftime('%H:%M')).strip()
    if not video_path or not os.path.exists(video_path):
        return jsonify({"error": f"File not found: {video_path}"}), 400
    job_id = str(uuid.uuid4())
    video_jobs[job_id] = {"status": "queued", "results": [], "stats": {}}
    threading.Thread(target=process_video_thread,
                     args=(job_id, video_path, crossing_time),
                     daemon=True).start()
    return jsonify({"job_id": job_id, "status": "queued"})

@app.route('/video_status/<job_id>', methods=['GET'])
def video_status(job_id):
    job = video_jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)

@app.route('/crossings', methods=['GET'])
def get_crossings():
    vtype  = request.args.get('vehicle_type', '')
    search = request.args.get('plate', '')
    conn   = get_db()
    q      = "SELECT * FROM crossings WHERE 1=1"
    p      = []
    if vtype:  q += " AND vehicle_type=?"; p.append(vtype)
    if search: q += " AND plate LIKE ?";   p.append(f"%{search}%")
    q += " ORDER BY created_at DESC"
    try:
        rows = conn.execute(q, p).fetchall()
        return jsonify({"crossings": [dict(r) for r in rows], "total": len(rows)})
    finally:
        conn.close()

@app.route('/crossings/stats', methods=['GET'])
def get_stats():
    conn  = get_db()
    today = datetime.now().strftime('%Y-%m-%d')
    try:
        total      = conn.execute("SELECT COUNT(*) FROM crossings").fetchone()[0]
        cars       = conn.execute("SELECT COUNT(*) FROM crossings WHERE vehicle_type='car'").fetchone()[0]
        bikes      = conn.execute("SELECT COUNT(*) FROM crossings WHERE vehicle_type='bike'").fetchone()[0]
        trucks     = conn.execute("SELECT COUNT(*) FROM crossings WHERE vehicle_type='truck'").fetchone()[0]
        td         = conn.execute("SELECT COUNT(*) FROM crossings WHERE created_at LIKE ?", (today+'%',)).fetchone()[0]
        viol_total = conn.execute("SELECT COUNT(*) FROM violations").fetchone()[0]
        viol_today = conn.execute("SELECT COUNT(*) FROM violations WHERE created_at LIKE ?", (today+'%',)).fetchone()[0]
        # ── v14: signal jump counts ──
        signal_total = conn.execute("SELECT COUNT(*) FROM violations WHERE violation_type='signal_jump'").fetchone()[0]
        signal_today = conn.execute(
            "SELECT COUNT(*) FROM violations WHERE violation_type='signal_jump' AND created_at LIKE ?",
            (today+'%',)
        ).fetchone()[0]
        return jsonify({
            "total": total, "today": td, "cars": cars,
            "bikes": bikes, "trucks": trucks,
            "violations_total": viol_total,
            "violations_today": viol_today,
            "signal_jump_total": signal_total,
            "signal_jump_today": signal_today,
        })
    finally:
        conn.close()

@app.route('/crossings/plate/<plate>', methods=['GET'])
def get_plate_history(plate):
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT * FROM crossings WHERE plate=? ORDER BY created_at DESC",
            (plate.upper(),)
        ).fetchall()
        return jsonify({"plate": plate.upper(),
                        "crossings": [dict(r) for r in rows], "total": len(rows)})
    finally:
        conn.close()

@app.route('/violations', methods=['GET'])
def get_violations():
    search = request.args.get('plate', '')
    vtype  = request.args.get('violation_type', '')
    conn   = get_db()
    q      = "SELECT * FROM violations WHERE 1=1"
    p      = []
    if search: q += " AND plate LIKE ?";       p.append(f"%{search}%")
    if vtype:  q += " AND violation_type=?";   p.append(vtype)
    q += " ORDER BY created_at DESC"
    try:
        rows = conn.execute(q, p).fetchall()
        return jsonify({"violations": [dict(r) for r in rows], "total": len(rows)})
    finally:
        conn.close()

@app.route('/')
def index():
    return send_from_directory(str(BASE_DIR), 'index.html')

if __name__ == '__main__':
    print(f"\n{'='*50}")
    print(f"  LPR Dashboard v14")
    print(f"  Folder : {BASE_DIR}")
    print(f"  DB     : {DB_PATH}")
    print(f"  Open   : http://localhost:5000")
    print(f"{'='*50}\n")
    app.run(debug=False, port=5000, threaded=True)