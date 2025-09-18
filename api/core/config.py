import os
from pydantic_settings import BaseSettings
from enum import Enum

class ModelType(str, Enum):
    ZERO_SHOT = "zero_shot"
    FINE_TUNED = "fine_tuned"

class Settings(BaseSettings):
    PROJECT_NAME: str = "CAMTOUR-CLASSIFIER-API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Model settings
    MODEL_TYPE: ModelType = ModelType.ZERO_SHOT
    ZERO_SHOT_MODEL: str = "facebook/bart-large-mnli"
    FINE_TUNED_MODEL_PATH: str = "models/finetuned/best_model.pth"
    
    # Device settings
    DEVICE: str = "cuda:0"  # Will auto-detect later
    
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings()