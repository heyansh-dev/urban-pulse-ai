from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root_endpoint():
    """Verify that the API root endpoint returns a 200 OK status."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"

def test_health_check():
    """Verify the API health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_endpoint_xgboost():
    """Test POST /predict using XGBoost model architecture."""
    payload = {
        "city": "bengaluru",
        "recent_readings": [
            {
                "timestamp": "2026-10-01T00:00:00Z",
                "city": "bengaluru",
                "pm10": 85.0,
                "pm25": 45.0
            }
        ] * 24
    }
    response = client.post("/predict?model_type=xgboost", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["city"] == "bengaluru"
    assert len(data["predictions"]) == 2