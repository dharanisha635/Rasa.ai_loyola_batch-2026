
# AI-Powered Real-Time Scene Understanding and Intelligent Vision Analytics

> **Binary street cleanliness classification using CNN, Bayesian Statistical Modelling, and Generative AI — deployed as a full-stack web application.**

---

## Project Overview

This project builds an end-to-end system that classifies street images as **Clean** or **Dirty**, quantifies prediction reliability using Bayesian uncertainty estimation, validates performance with formal statistical hypothesis tests, and auto-generates professional municipal cleanliness reports using a Large Language Model.

Built as an internship project by **Dharanisha R | 25PST032 | MSc Statistics**

---

## Live Demo

| Tab | What it does |
|---|---|
| **Predict** | Upload any street image → get label, confidence, uncertainty, entropy in ~2 seconds |
| **Statistics** | Interactive dashboard of all Phase 1 statistical analysis with hover tooltips |
| **Report** | One-click LLM-generated 4-sentence professional cleanliness assessment |

---

## Key Results

| Metric | Value |
|---|---|
| Accuracy | **85.29%** (29/34 test images) |
| ROC-AUC | **0.8997** (near-excellent) |
| Dirty F1 | **85.71%** |
| Clean F1 | **84.85%** |
| Binomial test p-value | **0.000019** (p < 0.001) |
| McNemar test p-value | **0.006** |
| Bootstrap 95% CI | **[73.53% – 97.06%]** |
| MC Dropout ambiguous predictions | **0 / 34** |
| Mean uncertainty (std) | **0.0946** |

---

## Architecture

```
Street Image (uploaded by user)
        │
        ▼
┌─────────────────┐
│   React Frontend │  ← Predict tab · Statistics tab · Report tab
└────────┬────────┘
         │  POST /predict
         ▼
┌─────────────────┐
│  Flask REST API  │  ← app.py
└────────┬────────┘
         │
         ├──── CNN Model (TensorFlow/Keras)
         │         └── MC Dropout × 50 passes
         │               ├── label
         │               ├── confidence
         │               ├── uncertainty (std)
         │               ├── predictive entropy
         │               └── is_ambiguous flag
         │
         └──── GET /stats ──── JSON files (pre-computed Phase 1 analysis)

Report Tab ──── Groq API (Llama 3 8B) ──── 4-sentence professional report
```

---

## Three Pillars

### 1. Computer Vision — CNN Classifier
- 4 convolutional blocks (32 → 64 → 128 filters)
- 128×128 RGB input, sigmoid binary output
- Dropout(0.5) — dual purpose: regularisation + Bayesian inference
- Trained on 155 balanced images with data augmentation

### 2. Statistical Modelling
- **Confidence distribution** — histogram, mean/std, Shapiro-Wilk normality test
- **Calibration analysis** — confidence bins vs actual accuracy
- **Confusion matrix** — ROC curve (AUC=0.8997), PR curve (AUC=0.8707)
- **Threshold sensitivity** — identifies t=0.4 as optimal (100% dirty recall)
- **Binomial test** — performance significantly above chance (p=0.000019)
- **McNemar's test** — significantly better than naive baseline (p=0.006)
- **Bootstrap CI** — 95% confidence interval via 10,000 resamples
- **MC Dropout** — Bayesian uncertainty via 50 stochastic passes
- **Predictive entropy** — information-theoretic uncertainty measure
- **Spearman correlation** — uncertainty vs error relationship (non-parametric)

### 3. Generative AI — LLM Report Generation
- Groq API (Llama 3 8B) via zero-shot prompting
- Receives: label, confidence, uncertainty std, predictive entropy
- Outputs: 4-sentence professional cleanliness assessment for municipal officers

---

## Project Structure

```
street-cleanliness/
│
├── street.py/                          # Python backend
│   ├── app.py                          # Flask REST API (4 endpoints)
│   ├── train.py                        # CNN architecture + training
│   ├── evaluate.py                     # Test set evaluation + MC Dropout
│   ├── statistical_analysis.py         # Distribution + calibration plots
│   ├── confusion_matrix_analysis.py    # ROC, PR, threshold sensitivity
│   ├── statistical_tests.py            # Binomial, McNemar, Bootstrap CI
│   ├── uncertainty_analysis.py         # Entropy, reliability, MC plots
│   ├── requirements.txt                # Python dependencies
│   ├── stats_summary.json              # 34 test predictions (generated)
│   ├── test_results.json               # Hypothesis test results (generated)
│   ├── uncertainty_results.json        # Per-image uncertainty (generated)
│   └── street_model.h5                 # Trained CNN weights (not in repo — see below)
│
└── street-frontend/                    # React frontend
    ├── public/
    │   └── plots/                      # Analysis PNG files for Statistics tab
    │       ├── confidence_analysis.png
    │       ├── confusion_matrix_analysis.png
    │       └── uncertainty_analysis.png
    ├── src/
    │   ├── App.js                      # Tab routing + state
    │   ├── App.css                     # Global dark theme styles
    │   ├── api.js                      # All API calls (Flask + Groq)
    │   └── components/
    │       ├── Navbar.jsx              # Tab navigation
    │       ├── PredictTab.jsx          # Image upload + result card
    │       ├── StatsTab.jsx            # Interactive statistics dashboard
    │       └── ReportTab.jsx           # LLM report generation
    ├── package.json
    └── .env.example                    # Environment variable template
```

---

## Setup and Installation

### Prerequisites
- Python 3.10+
- Node.js 18+
- A free [Groq API key](https://console.groq.com) (for LLM report generation)

---

### Step 1 — Clone the repository

```bash
git clone https://github.com/your-username/street-cleanliness-detector.git
cd street-cleanliness-detector
```

---

### Step 2 — Download the trained model

The model file `street_model.h5` is not included in the repository due to file size.

Download it from: **[Google Drive link — add your link here]**

Place it in the `street.py/` folder:
```
street.py/
└── street_model.h5   ← place here
```

---

### Step 3 — Set up the Python backend

```bash
cd street.py

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

---

### Step 4 — Set up the React frontend

```bash
cd street-frontend

# Install dependencies
npm install

# Create your environment file
copy .env.example .env        # Windows
# cp .env.example .env        # Mac/Linux
```

Open `.env` and add your Groq API key:
```
REACT_APP_GROQ_API_KEY=gsk_your_key_here
REACT_APP_API_URL=http://localhost:5000
```

---

### Step 5 — Run the application

Open **two terminals**:

**Terminal 1 — Flask API:**
```bash
cd street.py
python app.py
```
You should see: `Running on http://127.0.0.1:5000`

**Terminal 2 — React frontend:**
```bash
cd street-frontend
npm start
```
Dashboard opens at: `http://localhost:3000`

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/predict` | Upload image → returns label, confidence, uncertainty, entropy, is_ambiguous |
| `GET` | `/stats` | Returns all three statistical JSON files for the dashboard |
| `GET` | `/health` | API status and current threshold configuration |

### Example `/predict` response

```json
{
  "label": "Dirty",
  "confidence": 0.7940,
  "raw_score": 0.7940,
  "uncertainty": 0.0597,
  "entropy": 0.7335,
  "is_ambiguous": false,
  "threshold_used": 0.5
}
```

---

## Regenerating Statistical Analysis

If you retrain the model or want to rerun the analysis:

```bash
cd street.py

# Step 1 — Generate test predictions with MC Dropout
python evaluate.py

# Step 2 — Confidence distribution and calibration
python statistical_analysis.py

# Step 3 — Confusion matrix, ROC curve, threshold sensitivity
python confusion_matrix_analysis.py

# Step 4 — Binomial test, McNemar's test, Bootstrap CI
pip install statsmodels --break-system-packages
python statistical_tests.py

# Step 5 — MC Dropout uncertainty analysis
python uncertainty_analysis.py
```

Copy the output PNG files to `street-frontend/public/plots/`:
```bash
copy confidence_analysis.png street-frontend\public\plots\
copy confusion_matrix_analysis.png street-frontend\public\plots\
copy uncertainty_analysis.png street-frontend\public\plots\
```

---

## Statistical Methods Used

| Method | Purpose | Result |
|---|---|---|
| Shapiro-Wilk test | Normality check on confidence scores | W=0.923, p=0.019 — non-normal |
| Calibration analysis | Confidence vs actual accuracy per bin | Model is slightly underconfident |
| Binomial test | Performance vs random chance | p=0.000019 — highly significant |
| McNemar's test | CNN vs naive baseline comparison | p=0.006 — CNN significantly better |
| Bootstrap CI (N=10,000) | Empirical accuracy confidence interval | [73.53%, 97.06%] at 95% |
| MC Dropout (N=50) | Bayesian epistemic uncertainty | 0/34 ambiguous predictions |
| Predictive entropy | Information-theoretic uncertainty | Mean = 0.785 bits (normalised) |
| Spearman correlation | Uncertainty vs error relationship | r=-0.089, p=0.617 (non-parametric) |
| Threshold sensitivity | Optimal classification threshold | t=0.4 → 100% dirty recall |

---

## Technology Stack

| Component | Technology |
|---|---|
| Deep learning | TensorFlow / Keras |
| Statistical analysis | NumPy, SciPy, scikit-learn, statsmodels |
| Visualisation | Matplotlib |
| API backend | Flask + Flask-CORS |
| Frontend | React (create-react-app) |
| LLM report generation | Groq API — Llama 3 8B |
| Image processing | Pillow (PIL) |

---

## Important Notes

**Threshold:** The default classification threshold is 0.5. Statistical analysis identified t=0.4 as optimal for deployment (achieves 100% dirty recall). Change `THRESHOLD = 0.4` in `app.py` for zero false negatives.

**Model file:** `street_model.h5` is excluded from this repository. Download from the Google Drive link above and place in `street.py/`.

**API key:** Never commit your `.env` file. It is listed in `.gitignore`. Use `.env.example` as the template.

**Test set:** Statistical analysis was conducted on a balanced test set of 34 images (17 Clean, 17 Dirty) from a total dataset of 155 images.

---

## Author

**Dharanisha R**
MSc Statistics 
Street Cleanliness Detector — Internship Project 2026

