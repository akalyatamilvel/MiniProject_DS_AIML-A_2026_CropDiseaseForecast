import requests
import pandas as pd
import os

# We are collecting data for Tamil Nadu (Thanjavur - major rice region)
LATITUDE  = 10.787
LONGITUDE = 79.138
START_DATE = "2020-01-01"
END_DATE   = "2023-12-31"

print("Starting data collection...")
print(f"Location : Thanjavur, Tamil Nadu ({LATITUDE}, {LONGITUDE})")
print(f"Period   : {START_DATE} to {END_DATE}")
print("-" * 50)

# Open-Meteo historical weather API 
url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude"              : LATITUDE,
    "longitude"             : LONGITUDE,
    "start_date"            : START_DATE,
    "end_date"              : END_DATE,
    "daily"                 : [
        "temperature_2m_max",       # Max temperature (°C)
        "temperature_2m_min",       # Min temperature (°C)
        "precipitation_sum",        # Rainfall (mm)
        "windspeed_10m_max",        # Max wind speed (km/h)
        "relative_humidity_2m_max", # Max humidity (%)
        "relative_humidity_2m_min", # Min humidity (%)
    ],
    "timezone": "Asia/Kolkata"
}

print("Connecting to Open-Meteo API...")

try:
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()  
    data = response.json()
    print("Data downloaded successfully!")

except requests.exceptions.ConnectionError:
    print("ERROR: No internet connection. Please check your connection and try again.")
    exit()

except requests.exceptions.Timeout:
    print("ERROR: Request timed out. Try again.")
    exit()

except Exception as e:
    print(f"ERROR: {e}")
    exit()

daily = data["daily"]

df_climate = pd.DataFrame({
    "date"         : daily["time"],
    "temp_max"     : daily["temperature_2m_max"],
    "temp_min"     : daily["temperature_2m_min"],
    "rainfall_mm"  : daily["precipitation_sum"],
    "wind_speed"   : daily["windspeed_10m_max"],
    "humidity_max" : daily["relative_humidity_2m_max"],
    "humidity_min" : daily["relative_humidity_2m_min"],
})

# Calculate average temperature and average humidity
df_climate["temp_avg"]     = (df_climate["temp_max"] + df_climate["temp_min"]) / 2
df_climate["humidity_avg"] = (df_climate["humidity_max"] + df_climate["humidity_min"]) / 2

# Convert date column to proper date format
df_climate["date"] = pd.to_datetime(df_climate["date"])

# Fill any missing values with the column average
df_climate.fillna(df_climate.mean(numeric_only=True), inplace=True)

print(f"\nTotal rows collected : {len(df_climate)} days")
print(f"Columns              : {list(df_climate.columns)}")
print("\nFirst 5 rows of climate data:")
print(df_climate.head())

os.makedirs("dataset/raw_data", exist_ok=True)
climate_path = "dataset/raw_data/climate_data.csv"
df_climate.to_csv(climate_path, index=False)
print(f"\nClimate data saved to: {climate_path}")

# This dataset is based on real patterns from agricultural research:
# Rice Blast and Rice Brown Spot are most common in Tamil Nadu
# They occur when humidity > 80% and temperature is between 20-30°C

print("\nCreating disease outbreak dataset...")

import numpy as np
np.random.seed(42)  


months = pd.date_range(start="2020-01-01", end="2023-12-31", freq="MS")

disease_records = []
for month in months:
    month_num = month.month
    is_monsoon = 1 if month_num in [6, 7, 8, 9] else 0
    is_postmonsoon = 1 if month_num in [10, 11] else 0

    rice_blast_prob = 0.7 if is_monsoon else (0.3 if is_postmonsoon else 0.1)
    brown_spot_prob = 0.5 if is_monsoon else (0.2 if is_postmonsoon else 0.05)

    disease_records.append({
        "month"           : month.strftime("%Y-%m"),
        "year"            : month.year,
        "month_num"       : month_num,
        "rice_blast"      : int(np.random.random() < rice_blast_prob),
        "brown_spot"      : int(np.random.random() < brown_spot_prob),
        "is_monsoon"      : is_monsoon,
        "is_postmonsoon"  : is_postmonsoon,
    })

df_disease = pd.DataFrame(disease_records)

def risk_label(row):
    total = row["rice_blast"] + row["brown_spot"]
    if total == 2:
        return "High"
    elif total == 1:
        return "Medium"
    else:
        return "Low"

df_disease["risk_level"] = df_disease.apply(risk_label, axis=1)

print(f"Total months recorded : {len(df_disease)}")
print("\nDisease outbreak counts:")
print(df_disease["risk_level"].value_counts())
print("\nFirst 5 rows of disease data:")
print(df_disease.head())

disease_path = "dataset/raw_data/disease_records.csv"
df_disease.to_csv(disease_path, index=False)
print(f"\nDisease data saved to: {disease_path}")

print("\n" + "=" * 50)
print("Day 2 COMPLETE! Both datasets saved in dataset/raw_data/")
print("Files created:")
print("  - dataset/raw_data/climate_data.csv")
print("  - dataset/raw_data/disease_records.csv")
print("=" * 50)
