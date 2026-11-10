from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

from app.schemas import PredictionRequest, PredictionResponse
from app.services import prediction_engine

app = FastAPI(
    title="UrbanPulse AI - Air Quality Forecasting API",
    description="REST API delivering ML-powered multi-horizon PM10 and PM25 forecasts for urban nodes.",
    version="1.0.0"
)

# Enable CORS for local development and dashboard integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {
        "service": "UrbanPulse AI Prediction API",
        "status": "online",
        "docs_url": "/docs"
    }

@app.get("/health")
def health_check():
    """
    Health check endpoint to verify loaded model artifacts.
    """
    models_loaded = bool(prediction_engine.models)
    return {
        "status": "healthy" if models_loaded else "degraded",
        "models_loaded": models_loaded
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_air_quality(payload: PredictionRequest, model_type: Optional[str] = "xgboost"):
    """
    Accepts historical sensor readings payload and returns 1h and 24h predictions.
    """
    try:
        response = prediction_engine.predict(payload, preferred_model=model_type)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)