# 🌆 UrbanPulse AI

UrbanPulse AI is an end-to-end air quality and weather analytics platform featuring machine learning forecasting, statistical anomaly detection, and an interactive Streamlit & Plotly dashboard.

---

## 🛠️ Team Ownership & Roles

* **Developer 1:** Data Ingestion (OpenAQ / Open-Meteo), Database Storage, Data Normalization, FastAPI Backend Endpoints.
* **Developer 2:** Exploratory Data Analysis (EDA), Feature Engineering, ML Baseline & Forecasting Models, Anomaly Detection, Streamlit Dashboard.

---

## 📁 Repository Architecture

```text
urban-pulse-ai/
├── machine_learning/
│   ├── notebooks/          # EDA & experimental notebooks
│   ├── artifacts/          # Trained model binaries & metadata
│   ├── preprocessing.py    # Time-series resampling & cleaning
│   ├── feature_engineering.py # Lag, rolling, & temporal features
│   ├── train.py            # Model training & baseline evaluation
│   ├── predict.py          # Forecast output generators
│   └── anomaly_detection.py# Isolation Forest implementation
├── dashboard/
│   ├── app.py              # Streamlit dashboard entry point
│   ├── pages/              # Overview, City Detail, Analytics
│   └── components/         # API clients, metric cards, Plotly charts
├── tests/                  # Unit and integration tests
├── .gitignore
├── requirements.txt
└── README.md
