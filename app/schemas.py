from pydantic import BaseModel, Field
from typing import List

class AirQualityReading(BaseModel):
    timestamp: str = Field(..., json_schema_extra={"example": "2026-11-10T05:00:00Z"})
    city: str = Field(..., json_schema_extra={"example": "bengaluru"})
    pm10: float = Field(..., json_schema_extra={"example": 105.4})
    pm25: float = Field(..., json_schema_extra={"example": 62.1})

class PredictionRequest(BaseModel):
    city: str = Field(..., json_schema_extra={"example": "bengaluru"})
    recent_readings: List[AirQualityReading]

class SinglePredictionOutput(BaseModel):
    horizon: str = Field(..., json_schema_extra={"example": "1h"})
    pm10_prediction: float = Field(..., json_schema_extra={"example": 102.3})
    pm25_prediction: float = Field(..., json_schema_extra={"example": 58.6})

class PredictionResponse(BaseModel):
    city: str
    status: str
    predictions: List[SinglePredictionOutput]