# CropDisease Forecast: Predicting Crop Disease Outbreaks Using Climate Data

## Abstract

Crop diseases cause massive agricultural losses every year, especially in tropical regions like India where climate conditions rapidly change. This project aims to predict the likelihood of crop disease outbreaks by analyzing climate data such as temperature, humidity, rainfall, and wind speed alongside historical disease records. Using data science techniques including data cleaning, exploratory data analysis, visualization, and machine learning, we build a risk classification model that can help farmers and agricultural departments take early preventive action. The dataset is sourced from free and open platforms including Open-Meteo and ICRISAT. Our approach demonstrates how data-driven insights can directly support real-world agricultural decision-making and reduce crop losses.

---

## Problem Statement

Farmers in India and other agricultural regions face unpredictable crop disease outbreaks that lead to severe yield loss and economic damage. Traditional disease detection methods are reactive — farmers notice symptoms only after significant damage has occurred. This project addresses the problem of **early prediction of crop disease outbreaks** using historical climate variables (temperature, humidity, rainfall) and disease occurrence records. By identifying the climate patterns that precede outbreaks, we aim to provide a risk forecast that enables proactive intervention.

---

## Dataset Source

| Dataset | Source | Description |
|---|---|---|
| Climate data (temperature, humidity, rainfall, wind) | [Open-Meteo API](https://open-meteo.com/) | Free, no API key required |
| Historical crop disease records | [ICRISAT Open Data](https://www.icrisat.org/icrisat-open-data/) | Free historical disease data |
| Supplementary weather data | [IMD / data.gov.in](https://data.gov.in/) | India meteorological data |

---

## Methodology

1. **Problem Identification** — Define the target crop diseases and geography (focus: rice and wheat, Tamil Nadu region)
2. **Dataset Collection** — Fetch climate data via Open-Meteo API and disease records from ICRISAT
3. **Data Cleaning / Preprocessing** — Handle missing values, remove outliers, normalize features
4. **Exploratory Data Analysis (EDA)** — Understand distributions, correlations, seasonal patterns
5. **Data Visualization** — Plot climate trends, disease frequency, correlation heatmaps
6. **Model Development** — Train XGBoost classifier to predict disease risk level (Low / Medium / High)
7. **Result Interpretation** — Evaluate accuracy, identify key climate drivers of disease outbreaks

---

## Tools Used

| Tool | Purpose |
|---|---|
| Python 3.x | Core programming language |
| Pandas | Data manipulation and cleaning |
| NumPy | Numerical computations |
| Matplotlib & Seaborn | Data visualization |
| Scikit-learn | Machine learning utilities |
| XGBoost | Disease risk classification model |
| Jupyter Notebook | Interactive code and analysis |
| Git & GitHub | Version control and collaboration |

---

## Results 

- Temperature and relative humidity were found to be the strongest predictors of disease outbreak risk
- Disease outbreaks showed a strong seasonal pattern, peaking during high-humidity monsoon periods
- The XGBoost classifier achieved classification of crop disease risk into Low / Medium / High categories
- Visualizations revealed clear correlations between 7-day average humidity > 80% and outbreak occurrence
- Early warning is possible 5–7 days in advance using available forecast climate data
- Monsoon months (June–September) consistently showed the highest disease risk across all 4 years
- Humidity and high_humidity_days were identified as the top 2 most important features by the model


---

## Team Members

| Name | Role | GitHub |
|---|---|---|
| Akalya Tamilvel Senbakam | Data collection, EDA, Visualization | [@akalyatamilvel](https://github.com/) |
| Sachitha Ravichandran | Preprocessing, Model development, Report | [@sachitha07](https://github.com/) |

---

## Repository Structure

```
MiniProject/
├── README.md
├── requirements.txt
├── docs/
│   ├── abstract.pdf
│   ├── problem_statement.pdf
│   └── presentation.pptx
├── dataset/
│   ├── raw_data/
│   └── processed_data/
├── notebooks/
│   ├── data_understanding.ipynb
│   ├── preprocessing.ipynb
│   └── visualization.ipynb
├── src/
│   ├── preprocessing.py
│   ├── analysis.py
│   └── model.py
├── outputs/
│   ├── graphs/
│   └── results/
└── report/
    └── mini_project_report.pdf
```

---

## How to Run

1. Clone the repository:
```bash
git clone https://github.com/akalyatamilvel/MiniProject_DS_AIML-A_2026_CropDiseaseForecast.git
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run notebooks in order:
   - `notebooks/data_understanding.ipynb`
   - `notebooks/preprocessing.ipynb`
   - `notebooks/visualization.ipynb`

4. Run source files:
```bash
python src/preprocessing.py
python src/analysis.py
python src/model.py
```

---

*Project submitted for DS Mini Project — 2026*
