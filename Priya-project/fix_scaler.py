# paste this as fix_scaler.py
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE

df = pd.read_csv("diabetes_prediction_dataset.csv")
df.drop_duplicates(inplace=True)

le_gender  = LabelEncoder()
le_smoking = LabelEncoder()
df['gender']          = le_gender.fit_transform(df['gender'])
df['smoking_history'] = le_smoking.fit_transform(df['smoking_history'])

X = df.drop("diabetes", axis=1)
y = df["diabetes"]

# Scale BEFORE SMOTE exactly like your training code
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

with open("correct_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("le_gender.pkl", "wb") as f:
    pickle.dump(le_gender, f)

with open("le_smoking.pkl", "wb") as f:
    pickle.dump(le_smoking, f)

print("Feature order:", list(X.columns))
print("Scaler mean:", scaler.mean_.round(3))
print("Done!")