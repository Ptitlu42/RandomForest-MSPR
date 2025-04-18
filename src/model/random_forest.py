import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import sys
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.pipeline import Pipeline

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

os.makedirs(os.path.join(parent_dir, "models"), exist_ok=True)
os.makedirs(os.path.join(parent_dir, "visualisations"), exist_ok=True)

print("="*80)
print("IMPROVED POLITICAL ORIENTATION PREDICTION MODEL")
print("="*80)

data_path = os.path.join(os.path.dirname(current_dir), "data", "FINAL_FINAL.csv")

df = pd.read_csv(data_path)

election_years = [2002, 2007, 2012, 2017, 2022]
df_elections = df[df['annee'].isin(election_years)].copy()

print(f"Filtered data: {len(df_elections)} election years")

orientations = df_elections['orientation_gagnant'].dropna().unique()
print(f"\nPolitical orientations of winners: {orientations}")

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

print(f"\nWARNING: Dataset is very small ({len(X)} observations).")
print("Using advanced techniques for small datasets.")

print("\n1. Feature selection for dimensionality reduction...")
base_rf = RandomForestClassifier(n_estimators=100, random_state=42)
base_rf.fit(X.fillna(X.mean()), y)

feature_importance = pd.DataFrame(
    {'feature': features, 'importance': base_rf.feature_importances_}
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

selector = SelectFromModel(base_rf, threshold="mean", prefit=True)
X_selected = selector.transform(X.fillna(X.mean()))
selected_features = [features[i] for i in range(len(features)) if selector.get_support()[i]]
print(f"\nReduced feature set: {len(selected_features)} features")

print("\n2. Handling class imbalance...")
class_counts = pd.Series(y).value_counts()
print("Class distribution:")
for orientation, count in class_counts.items():
    print(f"{label_encoder.inverse_transform([orientation])[0]}: {count}")

print("Using RandomOverSampler instead of SMOTE for very small dataset...")
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_selected, y)
print(f"Data dimensions after oversampling: {X_resampled.shape}")

print("\n3. Creating an ensemble model...")
rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    bootstrap=True,
    class_weight='balanced',
    random_state=42
)

hist_gb = HistGradientBoostingClassifier(
    max_iter=200,
    learning_rate=0.05,
    max_depth=3,
    min_samples_leaf=1,
    random_state=42,
    early_stopping=False,
)

print("IMPORTANT: Using HistGradientBoostingClassifier which can handle NaN values natively")

ensemble = VotingClassifier(
    estimators=[('rf', rf), ('hist_gb', hist_gb)],
    voting='soft'
)

print("\n4. Cross-validation with Leave-One-Out...")
loo = LeaveOneOut()
cv_scores = cross_val_score(ensemble, X_resampled, y_resampled, cv=loo, scoring='accuracy')
print(f"Leave-One-Out CV accuracy: {cv_scores.mean():.4f}")

print("\n5. Training final model on all data...")
ensemble.fit(X_resampled, y_resampled)

y_pred = ensemble.predict(X_selected)
accuracy = accuracy_score(y, y_pred)
print(f"Final model accuracy on training data: {accuracy:.4f}")
print("\nClassification report:")
print(classification_report(y, y_pred, target_names=label_encoder.classes_))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

final_pipeline = Pipeline([
    ('feature_selection', selector),
    ('model', ensemble)
])

final_pipeline.fit(X_scaled, y)

print("\nSaving improved model and encoders...")
model_dir = os.path.join(parent_dir, "models")
joblib.dump(final_pipeline, os.path.join(model_dir, 'orientation_model.pkl'))
joblib.dump(label_encoder, os.path.join(model_dir, 'orientation_encoder.pkl'))
joblib.dump(scaler, os.path.join(model_dir, 'orientation_scaler.pkl'))
joblib.dump(features, os.path.join(model_dir, 'orientation_features.pkl'))
joblib.dump(selected_features, os.path.join(model_dir, 'orientation_selected_features.pkl'))

print("Enhanced model and encoders saved successfully.")