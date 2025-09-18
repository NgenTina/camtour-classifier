from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class PredictionRequest(BaseModel):
    prompt: str = Field(..., description="The text prompt to classify")
    candidate_labels: List[str] = Field(
        default=["tourism", "not_tourism"],
        description="Candidate labels for classification"
    )
    model_type: Optional[str] = Field(
        default=None,
        description="Optional model type override (zero_shot/fine_tuned)"
    )

class PredictionResponse(BaseModel):
    sequence: str
    labels: List[str]
    scores: List[float]
    prediction: str
    confidence: float
    is_tourism: bool
    model_info: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_info: Dict[str, Any]

class ErrorResponse(BaseModel):
    detail: str
    error_type: str