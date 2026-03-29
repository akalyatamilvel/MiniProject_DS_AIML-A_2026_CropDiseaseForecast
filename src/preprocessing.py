# =============================================================
# File: src/preprocessing.py
# Purpose: Clean and prepare raw climate + disease data
# =============================================================

import pandas as pd
import numpy as np
import os

print("=" * 55)
print("  STEP 3: DATA CLEANING & PREPROCESSING")
print("=" * 55)

# ── Step 1: Load the raw data ────────────────────────────────
print("\n[1] Loading raw data...")

climate_path = "dataset/raw_data/climate_data.csv"
disease_path = "dataset/raw_data/disease_records.csv"

df_climate = pd.read_csv(climate_path, parse_dates=["date"])
df_disease = pd.read_csv(disease_path)

print(f"    Climate data  : {df_climate.shape[0]} rows, {df_climate.shape[1]} columns")
print(f"    Disease data  : {df_disease.shape[0]} rows, {df_disease.shape[1]} columns")

# ── Step 2: Check for missing values ────────────────────────
print("\n[2] Checking for missing values...")
climate_missing = df_climate.isnull().sum()
disease_missing = df_disease.isnull().sum()

print("    Climate missing values:")
print(climate_missing[climate_missing > 0] if climate_missing.sum() > 0 else "    None found!")
print("    Disease missing values:")
print(disease_missing[disease_missing.sum() > 0] if disease_missing.sum() > 0 else "    None found!")

# Fill missing values with column mean
df_climate.fillna(df_climate.mean(numeric_only=True), inplace=True)
print("    Missing values filled with column averages.")

# ── Step 3: Remove outliers ──────────────────────────────────
print("\n[3] Removing outliers...")

# Temperature should be between 10 and 50 degrees for Tamil Nadu
before = len(df_climate)
df_climate = df_climate[
    (df_climate["temp_max"] >= 10) & (df_climate["temp_max"] <= 50) &
    (df_climate["temp_min"] >= 10) & (df_climate["temp_min"] <= 50)
]
after = len(df_climate)
print(f"    Rows removed as outliers: {before - after}")
print(f"    Rows remaining          : {after}")

# ── Step 4: Add useful new features ─────────────────────────
print("\n[4] Creating new features...")

# Month and year columns (useful for seasonal analysis)
df_climate["month"] = df_climate["date"].dt.month
df_climate["year"]  = df_climate["date"].dt.year

# Season column based on Indian seasons
def get_season(month):
    if month in [6, 7, 8, 9]:
        return "Monsoon"
    elif month in [10, 11]:
        return "Post-Monsoon"
    elif month in [12, 1, 2]:
        return "Winter"
    else:
        return "Summer"

df_climate["season"] = df_climate["month"].apply(get_season)

# Temperature range (difference between max and min)
df_climate["temp_range"] = df_climate["temp_max"] - df_climate["temp_min"]

# High humidity flag (humidity > 80% is risky for disease)
df_climate["high_humidity"] = (df_climate["humidity_avg"] > 80).astype(int)

# Heavy rainfall flag (rainfall > 10mm in a day)
df_climate["heavy_rain"] = (df_climate["rainfall_mm"] > 10).astype(int)

print("    New columns added: month, year, season, temp_range, high_humidity, heavy_rain")

# ── Step 5: Create monthly summary for merging with disease ──
print("\n[5] Creating monthly climate summary...")

monthly_climate = df_climate.groupby(["year", "month"]).agg(
    temp_avg_monthly    = ("temp_avg",      "mean"),
    humidity_avg_monthly= ("humidity_avg",  "mean"),
    rainfall_total      = ("rainfall_mm",   "sum"),
    wind_avg            = ("wind_speed",    "mean"),
    high_humidity_days  = ("high_humidity", "sum"),
    heavy_rain_days     = ("heavy_rain",    "sum"),
).reset_index()

# Round all numbers to 2 decimal places
monthly_climate = monthly_climate.round(2)

print(f"    Monthly summary: {monthly_climate.shape[0]} rows")
print(monthly_climate.head())

# ── Step 6: Merge climate + disease data ────────────────────
print("\n[6] Merging climate and disease data...")

df_disease["year"]      = df_disease["year"].astype(int)
df_disease["month_num"] = df_disease["month_num"].astype(int)

merged = pd.merge(
    monthly_climate,
    df_disease[["year", "month_num", "rice_blast", "brown_spot", "risk_level"]],
    left_on  = ["year", "month"],
    right_on = ["year", "month_num"],
    how      = "inner"
)

merged.drop(columns=["month_num"], inplace=True)
print(f"    Merged dataset: {merged.shape[0]} rows, {merged.shape[1]} columns")
print("\n    Risk level distribution:")
print(merged["risk_level"].value_counts())

# ── Step 7: Save cleaned data ────────────────────────────────
print("\n[7] Saving cleaned data...")

os.makedirs("dataset/processed_data", exist_ok=True)

# Save daily cleaned climate data
daily_path = "dataset/processed_data/climate_cleaned.csv"
df_climate.to_csv(daily_path, index=False)
print(f"    Saved: {daily_path}")

# Save merged monthly dataset (used for ML model)
merged_path = "dataset/processed_data/merged_dataset.csv"
merged.to_csv(merged_path, index=False)
print(f"    Saved: {merged_path}")

print("\n" + "=" * 55)
print("  PREPROCESSING COMPLETE!")
print("  Files saved in dataset/processed_data/")
print("=" * 55)
