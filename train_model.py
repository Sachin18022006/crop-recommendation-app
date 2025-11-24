# train_model.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the dataset
try:
    df = pd.read_csv('Crop_recommendation_with_soil.csv')
except FileNotFoundError:
    print("FATAL ERROR: 'Crop_recommendation_with_soil.csv' not found.")
    exit()

# --- 1. CROP PREDICTION MODEL TRAINING ---

print("1. Training Crop Prediction Model...")
df_encoded = pd.get_dummies(df, columns=['soil_type'], drop_first=True)
X = df_encoded.drop(['label'], axis=1)
y = df_encoded['label']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

joblib.dump(model, 'crop_recommendation_model.pkl')
joblib.dump(X.columns.tolist(), 'model_features.pkl')
print("✅ Crop Model and features saved.")

# --- 2. FERTILIZER RATIO CALCULATION ---

print("\n2. Calculating Fertilizer Ratios (Mean NPK per crop)...")
# Calculate the mean N, P, K for each crop label
fert_df = df.groupby('label')[['N', 'P', 'K']].mean().reset_index()

# Convert to a dictionary for fast lookup in the API
# The key is the crop label (e.g., 'rice'), and the value is a dictionary of N, P, K means
fert_dict = fert_df.set_index('label').to_dict('index')

# Save the dictionary
joblib.dump(fert_dict, 'fertilizer_ratios.pkl')
print("✅ Fertilizer Ratios saved.")