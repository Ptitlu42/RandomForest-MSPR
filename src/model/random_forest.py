import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

os.makedirs(os.path.join(parent_dir, "models"), exist_ok=True)
os.makedirs(os.path.join(parent_dir, "visualisations"), exist_ok=True)

print("="*80)
print("POLITICAL ORIENTATION PREDICTION MODEL")
print("="*80)

data_path = os.path.join(os.path.dirname(current_dir), "data", "FINAL_FINAL.csv")

df = pd.read_csv(data_path)

election_years = [2002, 2007, 2012, 2017, 2022]
df_elections = df[df['annee'].isin(election_years)].copy()

print(f"Filtered data: {len(df_elections)} election years")

print("\nPolitical orientations of winners:")
orientations = df_elections['orientation_gagnant'].dropna().unique()
print(orientations)

label_encoder = LabelEncoder()
df_elections['orientation_gagnant_encoded'] = label_encoder.fit_transform(df_elections['orientation_gagnant'])

orientations_encoding = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("\nPolitical orientation encoding:")
for orientation, code in orientations_encoding.items():
    print(f"{orientation}: {code}")

features_to_exclude = ['annee', 'parti_gagnant', 'parti_perdant', 'orientation_gagnant', 
                       'orientation_perdant', 'orientation_gagnant_encoded', 
                       'Voix vaincu', 'Voix vainqueur', 'Inscrits', 'Votants']

numeric_cols = df_elections.select_dtypes(include=['number']).columns
features = [col for col in numeric_cols if not any(excl in col for excl in features_to_exclude)]

print(f"\nNumber of features used: {len(features)}")

X = df_elections[features].copy()
y = df_elections['orientation_gagnant_encoded']

X = X.fillna(X.mean())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\nWARNING: Dataset is very small ({len(X)} observations).")

print("Training Random Forest model...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    bootstrap=True,
    class_weight='balanced',
    random_state=42
)
rf_model.fit(X_scaled, y)

print("Model trained successfully.")

y_pred = rf_model.predict(X_scaled)
accuracy = accuracy_score(y, y_pred)
print(f"Accuracy on training data: {accuracy:.4f}")

feature_importance = pd.DataFrame(
    {'feature': features, 'importance': rf_model.feature_importances_}
).sort_values('importance', ascending=False)

print("\nTop 10 most important features:")
print(feature_importance.head(10))

plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importance.head(15))
plt.title('Feature Importance for Political Orientation Prediction')
plt.tight_layout()
visualisation_path = os.path.join(parent_dir, "visualisations", "feature_importance.png")
plt.savefig(visualisation_path)
print(f"Feature importance visualization saved to '{visualisation_path}'")

print("\nSaving model and encoders...")
model_dir = os.path.join(parent_dir, "models")
joblib.dump(rf_model, os.path.join(model_dir, 'orientation_model.pkl'))
joblib.dump(label_encoder, os.path.join(model_dir, 'orientation_encoder.pkl'))
joblib.dump(scaler, os.path.join(model_dir, 'orientation_scaler.pkl'))
joblib.dump(features, os.path.join(model_dir, 'orientation_features.pkl'))

print("Model and encoders saved successfully.")