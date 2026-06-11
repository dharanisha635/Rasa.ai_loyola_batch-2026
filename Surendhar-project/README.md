\# 🏥 MedBill AI — AI-Powered Medical Billing Analytics System



An end-to-end AI pipeline that transcribes clinical audio dictations, extracts structured billing fields using Generative AI, and predicts insurance claim outcomes using Machine Learning.



\---



\## 📌 Project Title

AI-Powered Medical Billing Analytics System Using Voice-to-Text, Generative AI, and Statistical Modeling for Claim Optimization



\---



\## 🎯 Problem Statement

15-20% of medical claims are denied by insurance companies due to manual coding errors, incomplete documentation, and billing mistakes. This system automates the entire process — from audio dictation to claim prediction — helping hospitals identify high-risk claims before submission.



\---



\## 🚀 Features

\- 🎙️ Audio transcription using OpenAI Whisper (runs locally, offline)

\- 🤖 Structured field extraction using Groq Llama 3.3-70B

\- 🔍 Medical coding accuracy review (CPT + ICD-10 validation)

\- 💬 AI-generated denial explanation for high-risk claims

\- 🌲 Random Forest denial prediction with 88% Recall

\- 📊 Analytics Dashboard with 6 interactive charts

\- 🧪 Statistical Analysis — ANOVA, Chi-Square, T-Test, Pareto

\- 📋 3-page Flask web application



\---



\## 🔄 Pipeline

```

Audio (.mp3) → Whisper → Raw Transcript → Groq LLM → Structured Fields → MySQL → Random Forest → Prediction

```



\---



\## 🏗️ Project Structure

```

voice-to-text/

│

├── main.py                  ← Flask web application

├── random\_forest.py         ← Model training and saving

├── step1\_transcribe.py      ← Whisper transcription

├── step2\_extract.py         ← Groq field extraction

├── step3\_database.py        ← MySQL data storage

├── step4\_analysis.py        ← Statistical analysis

├── update\_claim\_status.py   ← Assign Paid/Denied/Pending

├── generate\_scripts.py      ← Synthetic audio generation

│

├── models/

│   ├── random\_forest.pkl    ← Trained model

│   ├── encoders.pkl         ← Label encoders

│   ├── features.pkl         ← Feature list

│   └── threshold.pkl        ← Decision threshold (0.35)

│

├── audio\_files/             ← 400 MP3 audio files

├── transcripts/             ← Whisper transcripts

├── extracted/               ← Groq extracted JSON files

├── charts/                  ← Generated analysis charts

├── results/                 ← Pending claim predictions

│

├── templates/               ← Flask HTML templates

├── static/                  ← CSS, JS, assets

│

├── .env                     ← API keys (not committed)

├── requirements.txt         ← Python dependencies

└── README.md

```



\---



\## 📊 Dataset

\- Total Records: 400 synthetic clinical encounters

\- Departments: Orthopedics, General Medicine, Dermatology, Cardiology, Pediatrics, Neurology

\- Diagnoses: 12 common outpatient conditions

\- Claim Status: 238 Paid (59.5%) | 43 Denied (10.8%) | 119 Pending (29.8%)

\- Avg Billed Amount: $218.61



\---



\## 🤖 Machine Learning Model

\- Algorithm: Random Forest Classifier

\- Training Set: 281 records (Paid + Denied)

\- Imbalance Fix: SMOTE + class\_weight='balanced'

\- Threshold: 35% (recall-optimised)

\- Tuning: RandomizedSearchCV (20 iterations, 5-fold CV)



\### Features Used

1\. billed\_amount

2\. patient\_age

3\. is\_rushed

4\. department (encoded)

5\. cpt\_code (encoded)

6\. icd10\_code (encoded)

7\. diagnosis (encoded)

8\. patient\_gender (encoded)

9\. month\_num



\### Model Performance

| Metric    | Score  |

|-----------|--------|

| Recall    | 88%    |

| Accuracy  | 47%    |

| Precision | 22%    |

| F1 Score  | 35%    |



\### Model Comparison

| Model               | Accuracy | Recall | F1    |

|---------------------|----------|--------|-------|

| Logistic Regression | 68.4%    | 33.3%  | 25.0% |

| Decision Tree       | 61.4%    | 55.6%  | 31.3% |

| Random Forest ⭐    | 47.4%    | 88.9%  | 34.8% |



Random Forest selected — highest recall to minimise missed denials.



\---



\## 🧪 Statistical Analysis

| Test          | Result                         | Finding                                  |

|---------------|--------------------------------|------------------------------------------|

| ANOVA         | F=63.94, p≈0.0 (Significant)  | Billed amount differs across departments |

| Chi-Square    | χ²=14.84, p=0.138 (Not Sig.)  | No strong dept-claim status association  |

| T-Test        | Rushed +4.8pp denial risk      | Rushed dictations increase denial risk   |

| Wilson 95% CI | 54.6% – 64.2%                 | True approval rate confidence interval   |



\---



\## 🔬 Medical Coding Accuracy (Groq AI)

\- Accuracy: 100%

\- CPT Match Rate: 82.5% (330/400 matched)

\- Mismatch Rate: 17.5% (70 flagged for review)

\- Avg Confidence: 89.8%

\- High Confidence (>80%): 400 records



\---



\## 🌐 Web Application (Flask)



\### Page 1 — Upload \& Predict

\- Upload clinical audio file (MP3, WAV, M4A, FLAC)

\- Whisper transcribes locally

\- Groq extracts 13 structured fields

\- Random Forest predicts claim outcome

\- Shows 4-level risk banner + AI explanation + recommended action



\### Page 2 — Analytics Dashboard

\- KPI cards: total records, denial rate, pending revenue, avg denied amount

\- Charts: claim status breakdown, denial by department, denial by diagnosis,

&#x20; rushed dictation impact, denial by age group, paid vs denied amounts



\### Page 3 — Statistical Analysis

\- ANOVA, Chi-Square, T-Test results

\- Model comparison chart and table

\- Medical coding accuracy metrics

\- Pareto analysis of top diagnoses

\- Approval rate by department



\---



\## ⚙️ Installation \& Setup



\### Prerequisites

\- Python 3.10+

\- MySQL Server

\- Git



\### 1. Clone the repository

```

git clone https://github.com/yourusername/medbill-ai.git

cd medbill-ai

```



\### 2. Create virtual environment

```

python -m venv venv

venv\\Scripts\\activate       # Windows

source venv/bin/activate    # Mac/Linux

```



\### 3. Install dependencies

```

pip install -r requirements.txt

```



\### 4. Set up .env file

```

DB\_HOST=localhost

DB\_USER=root

DB\_PASSWORD=your\_password

DB\_NAME=medical\_billing

GROQ\_API\_KEY\_1=your\_key\_1

GROQ\_API\_KEY\_2=your\_key\_2

GROQ\_API\_KEY\_3=your\_key\_3

```



\### 5. Set up MySQL database

```sql

CREATE DATABASE medical\_billing;

```



\### 6. Run the pipeline

```

\# Transcribe audio

python step1\_transcribe.py --folder audio\_files



\# Extract fields

python step2\_extract.py --folder transcripts/



\# Save to database

python step3\_database.py --file extracted/all\_records.json



\# Update claim status

python update\_claim\_status.py



\# Train and save model

python random\_forest.py

```



\### 7. Run the web application

```

python main.py

```

Open: http://127.0.0.1:5000



\---



\## 📦 Dependencies

```

\# Voice \& Transcription

faster-whisper



\# Generative AI

groq

python-dotenv



\# Machine Learning

scikit-learn

imbalanced-learn

joblib



\# Statistical Analysis

scipy

numpy



\# Data Handling

pandas

openpyxl

mysql-connector-python



\# Web Application

flask



\# Audio Generation

gtts

```



\---



\## 🔮 Future Enhancements

\- Train on real anonymised hospital data for production accuracy

\- Add XGBoost model for higher F1 score

\- Real-time API integration with hospital billing systems

\- Multi-language support for non-English dictations

\- Automated claim resubmission with corrected codes

\- Dashboard alerts for high-risk claims queue

\- Mobile-friendly responsive design



\---



\## 👨‍💻 Author

Surendhar B



\---



\## 📄 License

This project is for educational and research purposes only.

Patient data used is entirely synthetic and does not represent real individuals.

