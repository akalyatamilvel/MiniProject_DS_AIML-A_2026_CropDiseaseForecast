# =============================================================
# File: src/analysis.py
# Purpose: Perform statistical analysis on cleaned data
# =============================================================

import pandas as pd
import numpy as np
import os

print("=" * 55)
print("  STEP 4: EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 55)

# ── Step 1: Load cleaned data ────────────────────────────────
print("\n[1] Loading cleaned data...")

daily   = pd.read_csv("dataset/processed_data/climate_cleaned.csv", parse_dates=["date"])
monthly = pd.read_csv("dataset/processed_data/merged_dataset.csv")

print(f"    Daily data  : {daily.shape[0]} rows")
print(f"    Monthly data: {monthly.shape[0]} rows")

# ── Step 2: Basic statistics ─────────────────────────────────
print("\n[2] Basic statistics of climate data:")
stats = daily[["temp_avg", "humidity_avg", "rainfall_mm", "wind_speed"]].describe().round(2)
print(stats)

# ── Step 3: Season-wise analysis ─────────────────────────────
print("\n[3] Average climate values by season:")
season_stats = daily.groupby("season")[
    ["temp_avg", "humidity_avg", "rainfall_mm"]
].mean().round(2)
print(season_stats)

# ── Step 4: Disease outbreak analysis ───────────────────────
print("\n[4] Disease outbreak analysis:")
print(f"    Total months analysed : {len(monthly)}")
print(f"    Rice Blast outbreaks  : {monthly['rice_blast'].sum()} months")
print(f"    Brown Spot outbreaks  : {monthly['brown_spot'].sum()} months")
print(f"\n    Risk level counts:")
print(monthly["risk_level"].value_counts())

# ── Step 5: Correlation analysis ────────────────────────────
print("\n[5] Correlation with disease outbreaks:")
numeric_cols = ["temp_avg_monthly", "humidity_avg_monthly",
                "rainfall_total", "wind_avg",
                "high_humidity_days", "heavy_rain_days"]

for col in numeric_cols:
    corr_blast = monthly[col].corr(monthly["rice_blast"]).round(3)
    corr_spot  = monthly[col].corr(monthly["brown_spot"]).round(3)
    print(f"    {col:<25} → Rice Blast: {corr_blast:>6}  |  Brown Spot: {corr_spot:>6}")

# ── Step 6: High risk months ─────────────────────────────────
print("\n[6] High risk months (both diseases present):")
high_risk = monthly[monthly["risk_level"] == "High"][
    ["year", "month", "humidity_avg_monthly", "rainfall_total", "risk_level"]
]
print(high_risk.to_string(index=False))

# ── Step 7: Monthly disease frequency ───────────────────────
print("\n[7] Which months have most outbreaks?")
monthly_freq = monthly.groupby("month")[["rice_blast", "brown_spot"]].sum()
monthly_freq.index = ["Jan","Feb","Mar","Apr","May","Jun",
                      "Jul","Aug","Sep","Oct","Nov","Dec"]
print(monthly_freq)

# ── Save summary ─────────────────────────────────────────────
os.makedirs("outputs/results", exist_ok=True)
monthly.to_csv("outputs/results/analysis_summary.csv", index=False)
print("\n    Analysis summary saved to outputs/results/analysis_summary.csv")

print("\n" + "=" * 55)
print("  EDA COMPLETE!")
print("=" * 55)
