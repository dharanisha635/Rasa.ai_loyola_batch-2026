# 🌿 AgroStat AI
### AI-Powered Leaf Disease Detection, Statistical Analysis & LLM-Based Crop Advisory System

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Flask](https://img.shields.io/badge/Flask-Web%20Framework-green)
![Mistral](https://img.shields.io/badge/Mistral-Pixtral--12B-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 Overview

AgroStat AI is a web-based plant health monitoring platform that enables farmers and agronomists to detect leaf diseases instantly by uploading a single leaf image. The system uses **Pixtral-12B Vision AI** by Mistral to identify the plant, detect the disease, assess severity, and determine the cause — all in real time.

---

## ✨ Features

- 🔍 **AI Leaf Disease Detection** — Powered by Pixtral-12B multimodal vision model
- 📊 **7 Statistical Analytics** — Disease Frequency, Prevalence Rate, Severity Analysis, Accuracy, Precision/Recall/F1, Trend Analysis, Correlation Analysis
- 🧪 **Pixel-Level Analysis** — OpenCV HSV segmentation for real affected area measurement
- 📄 **AI Diagnostic Report** — LLM-generated 7-section professional report, downloadable as PDF
- 🧠 **Correction Memory System** — MD5 image hash fingerprinting to remember and apply user corrections
- 💬 **User Feedback System** — Collects correct/incorrect responses to compute real Precision, Recall and F1
- 📈 **Analytics Dashboard** — Historical statistics from SQLite database with real aggregate data
- 🌿 **20+ Plant Species** — Supports mango, tomato, potato, rice, banana, citrus, apple, grape, cotton, papaya, cassava, soybean, coffee, guava, sugarcane, okra, sunflower, neem, groundnut, brinjal and more

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Flask, Python 3.11 |
| AI Vision & Report | Pixtral-12B via Mistral API |
| Pixel Analysis | OpenCV, NumPy |
| Statistics | SciPy, Python statistics module |
| Database | SQLite (sqlite3) |
| Frontend | HTML5, CSS3, JavaScript |
| Charts | Chart.js 4.4.1 |
| Environment | python-dotenv |

---

## 📁 Project Structure

```
AgroStat-AI/
├── app/
│   ├── __init__.py
│   ├── routes.py
│   ├── predictor.py
│   ├── database.py
│   ├── pixel_analysis.py
│   ├── statistics_engine.py
│   ├── static/
│   │   ├── style.css
│   │   └── uploads/
│   └── templates/
│       ├── index.html
│       └── analytics.html
├── main.py
├── requirements.txt
├── .env
└── README.md
```

---

## ⚙️ Installation & Setup

**1. Clone the repository:**
```bash
git clone https://github.com/lakhs0128-tech/AgroStat-AI.git
cd AgroStat-AI
```

**2. Create and activate virtual environment:**
```bash
python -m venv venv
venv\Scripts\activate
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Create a `.env` file and add your Mistral API key:**
```
MISTRAL_API_KEY=your_mistral_api_key_here
```

**5. Run the application:**
```bash
py main.py
```

**6. Open in browser:**
```
http://127.0.0.1:5000
```

---

## 🚀 How It Works

1. User uploads a leaf image (JPG, PNG, WEBP)
2. Image is encoded in base64 and sent to Pixtral-12B Vision AI
3. AI returns structured diagnosis — plant, disease, severity, cause, remedies, precautions
4. OpenCV independently measures the real affected area using pixel analysis
5. 7 statistical analytics are computed from the AI response values
6. Results are saved to SQLite database
7. User can generate a downloadable PDF diagnostic report
8. User can submit feedback — correct or incorrect — to improve real F1 metrics
9. Correction memory system applies past corrections to future scans

---

## 📊 Statistical Analytics

| Analytics | Description |
|---|---|
| Disease Frequency Distribution | Diseased vs healthy tissue breakdown |
| Disease Prevalence Rate | Percentage share of each disease from total scans |
| Disease Severity Analysis | Affected area, spread risk, risk score, severity score |
| Accuracy Analysis | Radar chart of confidence, precision, recall, F1, recovery |
| Precision / Recall / F1 | Classification quality metrics from user feedback |
| Disease Trend Analysis | 8-week severity projection — untreated vs treated |
| Correlation Analysis | Temperature and humidity vs disease severity |

---

## 🌐 Pages

| Page | URL | Description |
|---|---|---|
| Scanner | `http://127.0.0.1:5000` | Upload leaf image and get diagnosis |
| Analytics Dashboard | `http://127.0.0.1:5000/analytics` | Historical statistics from all scans |

---

## 📋 Requirements

```
flask
python-dotenv
mistralai
opencv-python
numpy
scipy
```

---

## ⚠️ Disclaimer

AgroStat AI is intended for agricultural advisory use only and is not a substitute for professional phytopathological diagnosis. API rate limits on the free Mistral tier may cause temporary service interruptions.

---

## 👨‍💻 Developed By

**[R. LAKSHMI NARAYANAN]**  
[LOYOLA COLLEGE - CHENNAI]  
[M.Sc.STATISTICS]  
[2026]

---

## 📄 License

This project is for educational and agricultural advisory purposes only.
"# AgroStat-AI" 
