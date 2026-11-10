from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class SensorReading(BaseModel):
    timestamp: str = Field(..., example="2026-11-10T05:00:00Z")
    city: str = Field(..., example="bengaluru")
    pm10: float = Field(..., example=105.4)
    pm25: float = Field(..., example=62.1)

class PredictionRequest(BaseModel):
    city: str = Field(..., example="bengaluru")
    recent_readings: List[SensorReading] = Field(
        ..., 
        description="Historical sensor readings (minimum 24 hours required for rolling features)"
    )

class SinglePredictionOutput(BaseModel):
    horizon: str = Field(..., example="1h")
    pm10_prediction: float = Field(..., example=102.3)
    pm25_prediction: float = Field(..., example=58.6)

class PredictionResponse(BaseModel):
    city: str
    status: str = "success"
    predictions: List[SinglePredictionOutput]