from fastapi import APIRouter
from ...core.model_manager import model_manager
from ..models import HealthResponse

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and model status"""
    model_info = model_manager.get_model_info()
    
    return HealthResponse(
        status="healthy",
        model_loaded=model_info["model_loaded"],
        model_info=model_info
    )