# CDSS — Diabetes Clinical Decision Support System

## Project Structure

```
cdss/
├── app_new.py               ← Main Flask application
├── rag_knowledge.py         ← RAG prescription engine
├── train_new_model.py       ← Train the ANN model
├── statistics_analysis.py   ← Generate dataset statistics
├── requirements.txt
├── templates/
│   ├── index_new.html       ← Patient input form
│   ├── result_new.html      ← Prediction results
│   ├── doctor.html          ← Doctor portal (list view)
│   ├── doctor_review.html   ← Doctor review (single record)
│   └── statistics.html      ← Statistics page
└── static/
    └── stats_summary.json   ← (auto-generated)
```

## Setup Instructions

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download the dataset
Download `diabetes_prediction_dataset.csv` from Kaggle:
https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset

Place it in the project root folder.

### 3. Train the model
```bash
python train_new_model.py
```
This generates:
- `diabetes_new_model.h5`
- `new_scaler.pkl`
- `model_config.json`

### 4. Generate statistics (optional)
```bash
python statistics_analysis.py
```
This generates `static/stats_summary.json`.

### 5. Run the app
```bash
python app_new.py
```
Open http://127.0.0.1:5000 in your browser.

## Workflow
1. Enter patient data on the home page
2. AI predicts diabetes risk with confidence interval
3. RAG engine generates evidence-based prescription (ADA 2024)
4. 3-layer hallucination check validates the prescription
5. Doctor reviews at /doctor and approves/rejects/modifies
6. Approved prescriptions can be downloaded as PDF

## Features
- ANN model trained on 100K patient records
- Bootstrap confidence intervals (n=100)
- Feature importance (Z-score × ADA clinical weights)
- RAG knowledge base (ADA Standards 2024 + WHO Guidelines)
- Multi-layer hallucination check
- Doctor validation workflow
- PDF prescription generation
- Doctor feedback loop for AI improvement
