# save_correct_scaler.py
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("diabetes_prediction_dataset.csv")
df.drop_duplicates(inplace=True)

le_gender  = LabelEncoder()
le_smoking = LabelEncoder()
df['gender']          = le_gender.fit_transform(df['gender'])
df['smoking_history'] = le_smoking.fit_transform(df['smoking_history'])

X = df.drop("diabetes", axis=1)

scaler = StandardScaler()
scaler.fit(X)

with open("correct_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Save label encoders too
with open("le_gender.pkl", "wb") as f:
    pickle.dump(le_gender, f)
with open("le_smoking.pkl", "wb") as f:
    pickle.dump(le_smoking, f)

print("Saved correctly!")
print("Feature order:", list(X.columns))
print("Feature count:", X.shape[1])