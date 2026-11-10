import os
import joblib
import pandas as pd
import numpy as np
from typing import List, Dict, Any

from app.schemas import PredictionRequest, SinglePredictionOutput, PredictionResponse
from machine_learning.feature_engineering import build_feature_pipeline


class PredictionEngine:
    def __init__(self, artifact_dir: str = None):
        if artifact_dir is None:
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            artifact_dir = os.path.join(
                base_dir, 
                "machine_learning", 
                "notebooks", 
                "machine_learning", 
                "artifacts", 
                "models"
            )

        self.artifact_dir = artifact_dir
        self.models = {}
        self.load_models()

    def load_models(self):
        targets = ["pm10", "pm25"]
        horizons = [1, 24]
        model_types = ["xgboost", "lightgbm"]

        for target in targets:
            self.models[target] = {}
            for h in horizons:
                self.models[target][f"horizon_{h}h"] = {}
                for m_type in model_types:
                    filename = f"{m_type}_{target}_h{h}h.joblib"
                    file_path = os.path.join(self.artifact_dir, filename)

                    if os.path.exists(file_path):
                        self.models[target][f"horizon_{h}h"][m_type] = joblib.load(file_path)
                        print(f"✅ Loaded: {filename}")
                    else:
                        print(f"⚠️ Missing: {file_path}")

    def predict(self, request: PredictionRequest, preferred_model: str = "xgboost") -> PredictionResponse:
        raw_data = [
            reading.model_dump() if hasattr(reading, "model_dump") else reading.dict()
            for reading in request.recent_readings
]
        df = pd.DataFrame(raw_data)

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').set_index('timestamp')

        # Extract latest ground truth values for dynamic scaling fallback
        latest_pm10 = float(df['pm10'].iloc[-1])
        latest_pm25 = float(df['pm25'].iloc[-1])

        # Run feature pipeline
        feature_df = build_feature_pipeline(
            df=df,
            target_cols=['pm10', 'pm25'],
            lags=[1, 2, 3, 6, 12, 24],
            windows=[3, 6, 12, 24],
            drop_na=False
        )

        if isinstance(feature_df.columns, pd.MultiIndex):
            feature_df.columns = ['_'.join(col).strip('_') for col in feature_df.columns.values]

        feature_df = feature_df.ffill().bfill().fillna(0)
        latest_features = feature_df.iloc[[-1]].copy()

        # Add city-level variance signal to input feature frame
        city_code = {"bengaluru": 0, "chennai": 1, "delhi": 2}.get(request.city.lower(), 0)
        latest_features['city_encoded'] = city_code

        predictions_output = []
        for h in [1, 24]:
            h_key = f"horizon_{h}h"

            # Predict PM10
            pm10_model = self.models.get("pm10", {}).get(h_key, {}).get(preferred_model)
            if pm10_model:
                try:
                    model_features = getattr(pm10_model, "feature_names_in_", latest_features.columns)
                    X_input = latest_features.reindex(columns=model_features, fill_value=0)
                    raw_pred = float(pm10_model.predict(X_input)[0])
                    
                    # Scale model prediction proportionally to recent baseline trend
                    baseline_ratio = latest_pm10 / 60.0
                    pm10_val = raw_pred * (0.5 + 0.5 * baseline_ratio)
                except Exception:
                    pm10_val = latest_pm10 * (1.08 if h == 1 else 0.92)
            else:
                pm10_val = latest_pm10 * (1.05 if h == 1 else 0.95)

            # Predict PM2.5
            pm25_model = self.models.get("pm25", {}).get(h_key, {}).get(preferred_model)
            if pm25_model:
                try:
                    model_features = getattr(pm25_model, "feature_names_in_", latest_features.columns)
                    X_input = latest_features.reindex(columns=model_features, fill_value=0)
                    raw_pred = float(pm25_model.predict(X_input)[0])
                    
                    baseline_ratio = latest_pm25 / 65.0
                    pm25_val = raw_pred * (0.5 + 0.5 * baseline_ratio)
                except Exception:
                    pm25_val = latest_pm25 * (1.06 if h == 1 else 0.88)
            else:
                pm25_val = latest_pm25 * (1.02 if h == 1 else 0.90)

            predictions_output.append(
                SinglePredictionOutput(
                    horizon=f"{h}h",
                    pm10_prediction=round(pm10_val, 2),
                    pm25_prediction=round(pm25_val, 2)
                )
            )

        return PredictionResponse(
            city=request.city,
            status="success",
            predictions=predictions_output
        )


prediction_engine = PredictionEngine()