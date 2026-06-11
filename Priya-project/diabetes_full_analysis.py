# ============================================================
# CLINICAL DECISION SUPPORT SYSTEM FOR DIABETES PREDICTION
# FINAL BALANCED HEALTHCARE VERSION
# ============================================================

# ============================================================
# SUPPRESS WARNINGS
# ============================================================

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings('ignore')

# ============================================================
# IMPORT LIBRARIES
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

from imblearn.over_sampling import SMOTE

import statsmodels.api as sm

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv("diabetes_prediction_dataset.csv")

print("\nFIRST 5 ROWS")
print(df.head())

print("\nDATASET SHAPE")
print(df.shape)

# ============================================================
# REMOVE DUPLICATES
# ============================================================

df.drop_duplicates(inplace=True)

print("\nSHAPE AFTER REMOVING DUPLICATES")
print(df.shape)

# ============================================================
# MISSING VALUES
# ============================================================

print("\nMISSING VALUES")
print(df.isnull().sum())

# ============================================================
# ENCODE CATEGORICAL FEATURES
# ============================================================

le_gender = LabelEncoder()
le_smoking = LabelEncoder()

df['gender'] = le_gender.fit_transform(df['gender'])
df['smoking_history'] = le_smoking.fit_transform(df['smoking_history'])

# ============================================================
# FEATURES AND TARGET
# ============================================================

X = df.drop("diabetes", axis=1)
y = df["diabetes"]

# ============================================================
# FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTRAIN SIZE:", X_train.shape)
print("TEST SIZE :", X_test.shape)

# ============================================================
# CLASS DISTRIBUTION BEFORE SMOTE
# ============================================================

print("\nBEFORE SMOTE")
print(pd.Series(y_train).value_counts())

# ============================================================
# APPLY SMOTE
# ============================================================

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print("\nAFTER SMOTE")
print(pd.Series(y_train_smote).value_counts())

# ============================================================
# ============================================================
# LOGISTIC REGRESSION
# ============================================================
# ============================================================

print("\n==============================")
print("LOGISTIC REGRESSION")
print("==============================")

lr_model = LogisticRegression()

lr_model.fit(X_train_smote, y_train_smote)

# ============================================================
# LOGISTIC PREDICTIONS
# ============================================================

lr_prob = lr_model.predict_proba(X_test)[:, 1]

lr_pred = (lr_prob > 0.6).astype(int)

# ============================================================
# LOGISTIC METRICS
# ============================================================

lr_acc = accuracy_score(y_test, lr_pred)

lr_precision = precision_score(y_test, lr_pred)

lr_recall = recall_score(y_test, lr_pred)

lr_f1 = f1_score(y_test, lr_pred)

lr_auc = roc_auc_score(y_test, lr_prob)

print("\nLOGISTIC REGRESSION RESULTS")

print("Accuracy :", lr_acc)
print("Precision:", lr_precision)
print("Recall   :", lr_recall)
print("F1 Score :", lr_f1)
print("ROC AUC  :", lr_auc)

# ============================================================
# LOGISTIC CONFUSION MATRIX
# ============================================================

lr_cm = confusion_matrix(y_test, lr_pred)

plt.figure(figsize=(6,5))

sns.heatmap(
    lr_cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Logistic Regression Confusion Matrix")

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# ============================================================
# LOGISTIC CLASSIFICATION REPORT
# ============================================================

print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, lr_pred))

# ============================================================
# ============================================================
# STATISTICAL INFERENCE
# ============================================================
# ============================================================

print("\n==============================")
print("STATISTICAL INFERENCE")
print("==============================")

sample_size = 5000

X_sample = X_train_smote[:sample_size]
y_sample = y_train_smote[:sample_size]

X_sm = sm.add_constant(X_sample)

sm_model = sm.Logit(y_sample, X_sm)

result = sm_model.fit(disp=False)

print(result.summary())

# ============================================================
# ============================================================
# ANN MODEL
# ============================================================
# ============================================================

print("\n==============================")
print("ANN MODEL")
print("==============================")

ann_model = Sequential()

# ============================================================
# INPUT LAYER
# ============================================================

ann_model.add(Dense(
    64,
    activation='relu',
    input_dim=X_train_smote.shape[1]
))

ann_model.add(Dropout(0.3))

# ============================================================
# HIDDEN LAYER
# ============================================================

ann_model.add(Dense(
    32,
    activation='relu'
))

ann_model.add(Dropout(0.2))

# ============================================================
# SECOND HIDDEN LAYER
# ============================================================

ann_model.add(Dense(
    16,
    activation='relu'
))

ann_model.add(Dropout(0.2))

# ============================================================
# OUTPUT LAYER
# ============================================================

ann_model.add(Dense(
    1,
    activation='sigmoid'
))

# ============================================================
# COMPILE MODEL
# ============================================================

ann_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ============================================================
# EARLY STOPPING
# ============================================================

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# ============================================================
# TRAIN MODEL
# ============================================================

history = ann_model.fit(
    X_train_smote,
    y_train_smote,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# ============================================================
# OVERFITTING ANALYSIS
# ============================================================

plt.figure(figsize=(12,5))

# ============================================================
# ACCURACY GRAPH
# ============================================================

plt.subplot(1,2,1)

plt.plot(
    history.history['accuracy'],
    label='Training Accuracy'
)

plt.plot(
    history.history['val_accuracy'],
    label='Validation Accuracy'
)

plt.title("Training vs Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend()

# ============================================================
# LOSS GRAPH
# ============================================================

plt.subplot(1,2,2)

plt.plot(
    history.history['loss'],
    label='Training Loss'
)

plt.plot(
    history.history['val_loss'],
    label='Validation Loss'
)

plt.title("Training vs Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()

plt.show()

# ============================================================
# ANN PREDICTION PROBABILITY
# ============================================================

ann_prob = ann_model.predict(X_test)

# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

print("\n==============================")
print("THRESHOLD ANALYSIS")
print("==============================")

thresholds = np.arange(0.1, 1.0, 0.1)

results = []

for t in thresholds:

    y_pred = (ann_prob > t).astype(int)

    acc = accuracy_score(y_test, y_pred)

    prec = precision_score(y_test, y_pred)

    rec = recall_score(y_test, y_pred)

    f1 = f1_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)

    tn, fp, fn, tp = cm.ravel()

    results.append([
        t,
        acc,
        prec,
        rec,
        f1,
        fp,
        fn
    ])

    print("\n--------------------------------")
    print(f"Threshold: {t:.1f}")
    print("--------------------------------")

    print("Accuracy :", acc)
    print("Precision:", prec)
    print("Recall   :", rec)
    print("F1 Score :", f1)

    print("False Positives:", fp)
    print("False Negatives:", fn)

# ============================================================
# THRESHOLD TABLE
# ============================================================

results_df = pd.DataFrame(
    results,
    columns=[
        "Threshold",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "False Positives",
        "False Negatives"
    ]
)

print("\nTHRESHOLD ANALYSIS TABLE")
print(results_df)

# ============================================================
# MANUALLY SELECTED BALANCED THRESHOLD
# ============================================================

best_threshold = 0.6

print("\n==============================")
print("SELECTED THRESHOLD")
print("==============================")

print("Best Threshold:", best_threshold)

# ============================================================
# FINAL ANN PREDICTIONS
# ============================================================

final_pred = (ann_prob > best_threshold).astype(int)

# ============================================================
# FINAL ANN METRICS
# ============================================================

ann_acc = accuracy_score(y_test, final_pred)

ann_precision = precision_score(y_test, final_pred)

ann_recall = recall_score(y_test, final_pred)

ann_f1 = f1_score(y_test, final_pred)

ann_auc = roc_auc_score(y_test, ann_prob)

print("\n==============================")
print("FINAL ANN RESULTS")
print("==============================")

print("Accuracy :", ann_acc)
print("Precision:", ann_precision)
print("Recall   :", ann_recall)
print("F1 Score :", ann_f1)
print("ROC AUC  :", ann_auc)

# ============================================================
# FINAL CONFUSION MATRIX
# ============================================================

ann_cm = confusion_matrix(y_test, final_pred)

plt.figure(figsize=(6,5))

sns.heatmap(
    ann_cm,
    annot=True,
    fmt='d',
    cmap='Greens'
)

plt.title("ANN Confusion Matrix")

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, final_pred))

# ============================================================
# ROC CURVE
# ============================================================

fpr, tpr, thresholds = roc_curve(y_test, ann_prob)

plt.figure(figsize=(7,6))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {ann_auc:.4f}"
)

plt.plot(
    [0,1],
    [0,1],
    linestyle='--'
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.show()

# ============================================================
# MODEL COMPARISON
# ============================================================

comparison = pd.DataFrame({

    "Model": [
        "Logistic Regression",
        "ANN"
    ],

    "Accuracy": [
        lr_acc,
        ann_acc
    ],

    "Precision": [
        lr_precision,
        ann_precision
    ],

    "Recall": [
        lr_recall,
        ann_recall
    ],

    "F1 Score": [
        lr_f1,
        ann_f1
    ],

    "ROC AUC": [
        lr_auc,
        ann_auc
    ]
})

print("\n==============================")
print("MODEL COMPARISON")
print("==============================")

print(comparison)

# ============================================================
# SAVE MODEL
# ============================================================

ann_model.save("diabetes_model.keras")

print("\nMODEL SAVED SUCCESSFULLY")

