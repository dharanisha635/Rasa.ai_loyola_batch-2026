# 🚗 LPR Dashboard — License Plate Recognition System
> **v14** · Flask · YOLOv8 · EasyOCR · Gemini AI · SQLite

A full-stack Indian traffic surveillance system that detects vehicles, reads license plates, flags helmet violations, detects signal jumping, and answers natural-language queries about your traffic data via a RAG-powered chatbot.

---

## 📁 Project Structure

```
New folder/
│
├── app.py                  ← Flask backend (main entry point)
├── signal_jump.py          ← Signal jump detection module
├── index.html              ← Frontend dashboard (served by Flask)
│
├── best.pt                 ← Your trained YOLOv8 plate detection model
├── yolov8n.pt              ← COCO model for vehicle classification
├── helmet_yolov8n.pt       ← Helmet detection model
│
├── lpr_database.db         ← SQLite database (auto-created on first run)
├── .env                    ← API keys (create this yourself)
└── requirements.txt        ← Python dependencies
```

---

## ⚙️ Setup

### 1. Clone / copy the project files

Place all files in the same folder. The paths in `app.py` are resolved automatically relative to the script location — no hardcoding needed except for `helmet_yolov8n.pt` (see step 4).

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU users:** replace the `torch` line in `requirements.txt` with your CUDA build from [pytorch.org](https://pytorch.org/get-started/locally/) before installing.

### 4. Fix the helmet model path

Open `app.py` and update line 105 to point to where your helmet model lives:

```python
HELMET_MODEL_PATH = r"C:\path\to\your\helmet_yolov8n.pt"
```

### 5. Create the `.env` file

```
GEMINI_API_KEY=your_actual_key_here
```

Get a free key at [aistudio.google.com](https://aistudio.google.com).

### 6. Run

```bash
python app.py
```

Open your browser at **http://localhost:5000**

---

## 🧠 How It Works

### Detection Pipeline

```
Input (image or video)
        │
        ▼
COCO YOLOv8 (yolov8n.pt)
→ Detects vehicles: car / bike / truck
        │
        ▼
Plate YOLOv8 (best.pt)
→ Locates license plate regions
        │
        ▼
EasyOCR (multi-variant preprocessing)
→ Extracts raw plate text from 6 image variants
→ Scores by (char_count × confidence) — avoids partial reads
        │
        ▼
Gemini AI correction cascade
→ Tries gemini-2.5-flash → 2.0-flash → 1.5-flash on quota errors
→ Fixes OCR errors (O↔0, I↔1, B↔8, S↔5, …)
→ Falls back to local_correct_plate() if all Gemini quota exhausted
        │
        ▼
SQLite DB
→ Saves crossings + violations
```

### Violation Detection

| Violation | Applies To | Method |
|-----------|-----------|--------|
| `no_helmet` | bike only | Helmet YOLOv8 on top 55% of bike bounding box |
| `signal_jump` | car, bike, truck | HSV traffic light color + y2 stop line crossing |

### RAG Chatbot

```
User question
      │
      ▼
_rag_fetch_context()
→ Runs targeted SQLite queries:
    • Always: total/today crossings + violations, vehicle breakdown
    • If plate in question: full crossing + violation history
    • If time range in question: per-type breakdown for that hour window
    • If "frequent/top/latest": top plates or recent crossings
      │
      ▼
Gemini (max_output_tokens=150, temperature=0.2)
→ Answers in 50–60 words using only the DB facts
→ Falls back to local text parsing if Gemini is down
```

---

## 🔌 API Reference

All endpoints return JSON. Base URL: `http://localhost:5000`

### Image Detection

```
POST /detect_image
Body: { "image_path": "C:/path/to/image.jpg", "crossing_time": "14:30" }

Response:
{
  "session_id": "uuid",
  "plates_saved": 2,
  "plates": [
    { "plate": "TN01AB1234", "vehicle_type": "car", "crossing_time": "14:30" },
    { "plate": "KA02FY7410", "vehicle_type": "bike", "crossing_time": "14:30",
      "violation": "no_helmet" }
  ]
}
```

### Video Processing (standard)

```
POST /process_video
Body: { "video_path": "C:/path/to/video.mp4", "crossing_time": "14:30" }

Response: { "job_id": "uuid", "status": "queued" }
```

### Video Processing (with signal jump detection)

```
POST /process_signal_video
Body: {
  "video_path": "C:/path/to/video.mp4",
  "crossing_time": "14:30",
  "stop_line_y": 400,           ← optional, overrides global config
  "roi": [1100, 20, 1260, 180]  ← optional, overrides global config
}

Response: { "job_id": "uuid", "status": "queued" }
```

### Poll Video Job

```
GET /video_status/<job_id>

Response (done):
{
  "status": "done",
  "results": [ { "plate": "...", "vehicle_type": "...", "signal_jump": true } ],
  "stats": { "total_frames": 900, "plates_saved": 4, "signal_jumps": 1 }
}
```

### Crossings

```
GET /crossings                          ← all crossings
GET /crossings?vehicle_type=bike        ← filter by type (car/bike/truck)
GET /crossings?plate=TN01               ← search by partial plate
GET /crossings/plate/TN01AB1234         ← full history for one plate
```

### Stats

```
GET /crossings/stats

Response:
{
  "total": 142,
  "today": 18,
  "cars": 80,
  "bikes": 45,
  "trucks": 17,
  "violations_total": 23,
  "violations_today": 4,
  "signal_jump_total": 9,
  "signal_jump_today": 2,
  "helmet_total": 14
}
```

### Violations

```
GET /violations                              ← all violations
GET /violations?plate=KA04                   ← filter by plate
GET /violations?violation_type=signal_jump   ← filter by type
GET /violations?violation_type=no_helmet
```

### Signal Jump Configuration

```
GET /signal_config
Response:
{
  "enabled": true,
  "roi": [1100, 20, 1260, 180],
  "stop_line_y": 400,
  "tolerance": 10,
  "min_lit_pixels": 40,
  "current_light": "red"
}

POST /signal_config
Body (any subset of fields):
{
  "stop_line_y": 350,
  "roi": [900, 10, 1100, 160],
  "tolerance": 15,
  "min_lit_pixels": 60,
  "enabled": true
}
```

### RAG Chatbot

```
POST /chat
Body: { "message": "How many violations happened today?" }

Response: { "answer": "There were 4 violations today, all no_helmet type." }
```

Example questions the bot handles:
- *"How many vehicles crossed between 11 and 12?"*
- *"Did KA04MN9876 cross here before?"*
- *"How many signal jump violations today?"*
- *"Most frequent plates?"*
- *"Bike vs car vs truck count"*

---

## 🚦 Signal Jump Detection — Setup Guide

The `SignalJumpDetector` in `signal_jump.py` needs two values calibrated to your camera:

### 1. ROI (traffic light bounding box)

The ROI is `[x1, y1, x2, y2]` in pixels, pointing at the traffic light circle in your frame. Default is top-right corner `[1100, 20, 1260, 180]` for a 1280×720 frame.

**To find your ROI:**
```python
import cv2
img = cv2.imread("your_frame.jpg")
# Check frame dimensions
print(img.shape)  # (height, width, channels)
# Then crop and preview until it captures just the traffic light lamp
```

### 2. Stop Line Y coordinate

`stop_line_y` is the Y pixel where the stop line is drawn across the road. A vehicle triggers the violation when its bounding box bottom (`y2`) reaches this Y value.

**To find your stop line Y:**
```python
import cv2
img = cv2.imread("your_frame.jpg")
# Click on the stop line in any image viewer and note the Y pixel value
```

### 3. Apply without restarting Flask

```bash
curl -X POST http://localhost:5000/signal_config \
  -H "Content-Type: application/json" \
  -d '{"stop_line_y": 380, "roi": [900, 15, 1100, 170]}'
```

### HSV Tuning

If your traffic light uses unusual colours or your camera has a strong tint, adjust the HSV ranges at the top of `signal_jump.py`:

```python
SIGNAL_COLORS = {
    "red_lo1":  (  0, 120,  80),   # lower red hue band
    "red_hi1":  ( 10, 255, 255),
    "red_lo2":  (160, 120,  80),   # upper red hue band (wraps around)
    "red_hi2":  (180, 255, 255),
    "green_lo": ( 40,  60,  60),   # green hue band
    "green_hi": ( 90, 255, 255),
}
```

---

## 🗄️ Database Schema

**`crossings`**

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Auto PK |
| plate | TEXT | Indian plate number e.g. `TN01AB1234` |
| vehicle_type | TEXT | `car` / `bike` / `truck` |
| crossing_time | TEXT | HH:MM as entered by user |
| source_path | TEXT | Path to source image/video |
| source_type | TEXT | `image` or `video` |
| session_id | TEXT | UUID per processing session |
| created_at | TEXT | `datetime('now','localtime')` |

**`violations`**

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Auto PK |
| plate | TEXT | Plate that violated |
| vehicle_type | TEXT | `car` / `bike` / `truck` |
| violation_type | TEXT | `no_helmet` or `signal_jump` |
| crossing_time | TEXT | HH:MM |
| source_path | TEXT | Path to source |
| source_type | TEXT | `image` or `video` |
| session_id | TEXT | UUID |
| created_at | TEXT | Localtime timestamp |

---

## 🔑 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | Yes | Google Gemini API key. Without it, plate correction and the chatbot will not work. |

---

## 📌 Indian Plate Format

The system validates against the standard Indian plate format:

```
XX  00  XXX  0000
│   │   │    └── 4-digit serial number
│   │   └─────── 1–3 letter series code
│   └─────────── 2-digit district code
└─────────────── 2-letter state code (e.g. TN, KA, MH, DL …)
```

All 37 state/UT codes are supported: `AN AP AR AS BR CG CH DD DL DN GA GJ HP HR JH JK KA KL LA LD MH ML MN MP MZ NL OD PB PY RJ SK TN TR TS UK UP WB`

---

## 🐛 Common Issues

**`SyntaxError: invalid decimal literal` on startup**
→ You ran `index.html` instead of `app.py`. Make sure you're running `python app.py`.

**`GEMINI_API_KEY not set` warning**
→ Create a `.env` file in the same folder as `app.py` with `GEMINI_API_KEY=your_key`.

**Helmet model not loading**
→ Update `HELMET_MODEL_PATH` in `app.py` line 105 to the full path of your `helmet_yolov8n.pt`.

**Signal jump never fires**
→ Check `GET /signal_config` — if `current_light` is always `unknown`, your ROI does not capture the traffic light. Recalibrate `roi` for your camera resolution.

**Chatbot says "I couldn't retrieve an answer"**
→ Gemini quota is exhausted and the question didn't match the offline fallback patterns. Check your Gemini usage at [aistudio.google.com](https://aistudio.google.com) or try again after quota resets (typically midnight PST).

**OCR reads partial plates like "33"**
→ This is fixed in v13 by the `char_count × confidence` scorer. If it still happens, the plate crop is smaller than 60×55 px — try reducing `conf` threshold in `plate_model.predict()` or increasing camera resolution.

---

## 🗺️ Version History

| Version | What changed |
|---------|-------------|
| v13 | Multi-variant OCR preprocessing, char×confidence scorer, local plate corrector, video conf filter removed |
| v13.1 | Replaced Roboflow API with local `helmet_yolov8n.pt` |
| v13.2 | RAG chatbot (`/chat` route) with live SQLite context + Gemini |
| v13.2.1 | Fixed time-range regex (`11-12` format), raised token limit, better fallback |
| **v14** | Signal jump detection (`signal_jump.py`), new `signal_jump` violation type, `/signal_config` + `/process_signal_video` routes, signal jump stats in `/crossings/stats` |

---

## 📄 License

This project is for educational and research purposes.
