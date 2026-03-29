# =============================================================
# File: src/model.py
# Purpose: Train a machine learning model to predict disease risk
# =============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from xgboost import XGBClassifier
import os

print("=" * 55)
print("  STEP 6: MODEL DEVELOPMENT")
print("=" * 55)

# ── Step 1: Load merged dataset ──────────────────────────────
print("\n[1] Loading processed dataset...")
df = pd.read_csv("dataset/processed_data/merged_dataset.csv")
print(f"    Dataset shape: {df.shape}")

# ── Step 2: Prepare features and target ──────────────────────
print("\n[2] Preparing features and target variable...")

# Features (X) = climate columns we use to predict disease
feature_cols = [
    "temp_avg_monthly",
    "humidity_avg_monthly",
    "rainfall_total",
    "wind_avg",
    "high_humidity_days",
    "heavy_rain_days",
    "month"
]

X = df[feature_cols]

# Target (y) = what we want to predict: risk level
# We convert Low/Medium/High to numbers: Low=0, Medium=1, High=2
le = LabelEncoder()
y  = le.fit_transform(df["risk_level"])

print(f"    Features used : {feature_cols}")
print(f"    Target classes: {list(le.classes_)}")
print(f"    X shape: {X.shape}, y shape: {y.shape}")

# ── Step 3: Split into train and test sets ───────────────────
print("\n[3] Splitting data into train (80%) and test (20%)...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"    Training samples : {len(X_train)}")
print(f"    Testing samples  : {len(X_test)}")

# ── Step 4: Train the XGBoost model ─────────────────────────
print("\n[4] Training XGBoost classifier...")
print("    (This predicts Low / Medium / High disease risk)")

model = XGBClassifier(
    n_estimators  = 100,
    max_depth     = 4,
    learning_rate = 0.1,
    random_state  = 42,
    eval_metric   = "mlogloss",
    verbosity     = 0
)

model.fit(X_train, y_train)
print("    Model training complete!")

# ── Step 5: Evaluate the model ───────────────────────────────
print("\n[5] Evaluating model performance...")

y_pred    = model.predict(X_test)
accuracy  = accuracy_score(y_test, y_pred)

print(f"\n    Accuracy: {accuracy * 100:.2f}%")
print("\n    Detailed Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ── Step 6: Feature importance ───────────────────────────────
print("\n[6] Feature importance (which climate factor matters most):")
importance = pd.DataFrame({
    "feature"   : feature_cols,
    "importance": model.feature_importances_.round(4)
}).sort_values("importance", ascending=False)

for _, row in importance.iterrows():
    bar = "█" * int(row["importance"] * 40)
    print(f"    {row['feature']:<25} {bar} {row['importance']}")

# ── Step 7: Make a sample prediction ─────────────────────────
print("\n[7] Sample prediction:")
sample = pd.DataFrame([{
    "temp_avg_monthly"    : 29.5,
    "humidity_avg_monthly": 85.0,
    "rainfall_total"      : 120.0,
    "wind_avg"            : 15.0,
    "high_humidity_days"  : 22,
    "heavy_rain_days"     : 8,
    "month"               : 8
}])

pred       = model.predict(sample)
pred_label = le.inverse_transform(pred)[0]
print(f"    Input  : Temp=29.5°C, Humidity=85%, Rainfall=120mm, Month=August")
print(f"    Prediction → Disease Risk Level: {pred_label}")

# ── Step 8: Save results ─────────────────────────────────────
print("\n[8] Saving results...")
os.makedirs("outputs/results", exist_ok=True)

# Save predictions on test set
results_df = X_test.copy()
results_df["actual_risk"]    = le.inverse_transform(y_test)
results_df["predicted_risk"] = le.inverse_transform(y_pred)
results_df["correct"]        = results_df["actual_risk"] == results_df["predicted_risk"]
results_df.to_csv("outputs/results/model_predictions.csv", index=False)

# Save feature importance
importance.to_csv("outputs/results/feature_importance.csv", index=False)

print("    Saved: outputs/results/model_predictions.csv")
print("    Saved: outputs/results/feature_importance.csv")

print("\n" + "=" * 55)
print("  MODEL DEVELOPMENT COMPLETE!")
print(f"  Final Accuracy: {accuracy * 100:.2f}%")
print("=" * 55)
