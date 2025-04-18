import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

os.makedirs(os.path.join(parent_dir, "visualisations"), exist_ok=True)

print("="*80)
print("POLITICAL ORIENTATION PREDICTION FOR 2027")
print("="*80)

try:
    print("\nLoading model and encoders...")
    model_dir = os.path.join(parent_dir, "models")
    model = joblib.load(os.path.join(model_dir, 'orientation_model.pkl'))
    label_encoder = joblib.load(os.path.join(model_dir, 'orientation_encoder.pkl'))
    scaler = joblib.load(os.path.join(model_dir, 'orientation_scaler.pkl'))
    features = joblib.load(os.path.join(model_dir, 'orientation_features.pkl'))
except FileNotFoundError:
    print("ERROR: Model not found.")
    print("Please run src/model/random_forest.py first")
    sys.exit(1)

orientations_encoding = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("\nAvailable political orientations:")
for orientation, code in orientations_encoding.items():
    print(f"{orientation}: {code}")

data_path = os.path.join(os.path.dirname(current_dir), "data", "FINAL_FINAL.csv")
df = pd.read_csv(data_path)

latest_year = df['annee'].max()
print(f"Latest available year in data: {latest_year}")

def predict_for_2027(df, features, model, label_encoder, scaler):
    latest_data = df[df['annee'] == latest_year].copy()
    latest_data['annee'] = 2027
    
    X_future = latest_data[features].fillna(latest_data[features].mean()).values.reshape(1, -1)
    X_future_scaled = scaler.transform(X_future)
    
    y_pred_index = model.predict(X_future_scaled)[0]
    predicted_orientation = label_encoder.inverse_transform([y_pred_index])[0]
    
    probabilities = model.predict_proba(X_future_scaled)[0]
    orientations = label_encoder.classes_
    proba_dict = {orientation: prob for orientation, prob in zip(orientations, probabilities)}
    
    return predicted_orientation, proba_dict

predicted_orientation, probabilities = predict_for_2027(
    df, features, model, label_encoder, scaler
)

current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

print("\n" + "="*80)
print(f"PREDICTION RESULTS ({current_datetime})")
print("="*80)

print(f"\nPredicted political orientation for 2027: {predicted_orientation}")
print("\nProbabilities by orientation:")

sorted_probas = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
for orientation, prob in sorted_probas:
    print(f"{orientation}: {prob:.4f} ({prob*100:.1f}%)")

plt.figure(figsize=(10, 6))
sns.barplot(x=[p[0] for p in sorted_probas], y=[p[1] for p in sorted_probas])
plt.title('Political Orientation Probabilities for 2027')
plt.ylabel('Probability')
plt.xlabel('Political Orientation')
plt.xticks(rotation=45)
plt.ylim(0, 1)

plt.text(0.95, 0.05, f"Generated: {current_datetime}", 
         ha='right', va='bottom', transform=plt.gca().transAxes, 
         fontsize=8, color='gray')

plt.tight_layout()
prediction_path = os.path.join(parent_dir, "visualisations", "prediction_2027.png")
plt.savefig(prediction_path)

print(f"\nPrediction visualization saved to '{prediction_path}'")

print("\nNote: This prediction is based on a model trained on limited historical data (5 elections).")