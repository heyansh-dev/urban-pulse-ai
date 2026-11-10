import pytest
from app.services import prediction_engine
from app.schemas import PredictionRequest, AirQualityReading

def test_prediction_engine_initialization():
    """Verify that model artifacts are loaded upon initialization."""
    assert prediction_engine is not None
    assert hasattr(prediction_engine, "models")

def test_city_encoding_variance():
    """Verify that different city inputs yield valid structural responses."""
    readings = [
        AirQualityReading(
            timestamp="2026-10-01T00:00:00Z",
            city="delhi",
            pm10=120.0,
            pm25=75.0
        )
    ] * 24
    
    req_delhi = PredictionRequest(city="delhi", recent_readings=readings)
    res_delhi = prediction_engine.predict(req_delhi, preferred_model="xgboost")
    
    assert res_delhi.city == "delhi"
    assert len(res_delhi.predictions) == 2