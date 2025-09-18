from transformers import pipeline
import torch
import logging
from typing import Dict, Any, List, Optional
from .config import settings, ModelType
import os

logger = logging.getLogger(__name__)

class ModelManager:
    _instance = None
    _model = None
    _model_type = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance
    
    def initialize_model(self, model_type: Optional[ModelType] = None):
        """Initialize the appropriate model based on configuration"""
        if model_type is None:
            model_type = settings.MODEL_TYPE
        
        self._model_type = model_type
        
        try:
            if model_type == ModelType.ZERO_SHOT:
                self._load_zero_shot_model()
            elif model_type == ModelType.FINE_TUNED:
                self._load_fine_tuned_model()
            else:
                raise ValueError(f"Unknown model type: {model_type}")
                
            logger.info(f"Successfully loaded {model_type.value} model")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            # Fallback to a smaller model if the main one fails
            self._load_fallback_model()
    
    def _load_zero_shot_model(self):
        """Load the zero-shot classification model"""
        device = 0 if torch.cuda.is_available() and "cuda" in settings.DEVICE else -1
        
        # Use token if provided
        model_kwargs = {}
        if settings.HF_TOKEN:
            model_kwargs["token"] = settings.HF_TOKEN
        
        self._model = pipeline(
            "zero-shot-classification",
            model=settings.ZERO_SHOT_MODEL,
            device=device,
            **model_kwargs
        )
    
    def _load_fallback_model(self):
        """Load a fallback model if the primary model fails"""
        logger.warning("Loading fallback model: typeform/distilbert-base-uncased-mnli")
        device = 0 if torch.cuda.is_available() and "cuda" in settings.DEVICE else -1
        
        self._model = pipeline(
            "zero-shot-classification",
            model="typeform/distilbert-base-uncased-mnli",
            device=device
        )
        self._model_type = ModelType.ZERO_SHOT
    
    def _load_fine_tuned_model(self):
        """Load a fine-tuned model (to be implemented later)"""
        # Placeholder for fine-tuned model loading
        raise NotImplementedError("Fine-tuned model loading not implemented yet")
    
    def predict(self, prompt: str, candidate_labels: List[str]) -> Dict[str, Any]:
        """Make a prediction using the loaded model"""
        if self._model is None:
            self.initialize_model()
        
        if self._model_type == ModelType.ZERO_SHOT:
            return self._predict_zero_shot(prompt, candidate_labels)
        elif self._model_type == ModelType.FINE_TUNED:
            return self._predict_fine_tuned(prompt, candidate_labels)
        else:
            raise ValueError(f"Unknown model type: {self._model_type}")
    
    def _predict_zero_shot(self, prompt: str, candidate_labels: List[str]) -> Dict[str, Any]:
        """Zero-shot prediction using transformers pipeline"""
        result = self._model(prompt, candidate_labels, multi_label=False)
        
        return {
            "sequence": result["sequence"],
            "labels": result["labels"],
            "scores": [float(score) for score in result["scores"]],  # Convert to float for JSON
            "prediction": result["labels"][0],  # Top prediction
            "confidence": float(result["scores"][0]),  # Top confidence as float
            "is_tourism": result["labels"][0] == "tourism",
            "model_type": self._model_type.value if self._model_type else "unknown",
            "model_name": settings.ZERO_SHOT_MODEL if self._model_type == ModelType.ZERO_SHOT else "fallback"
        }
    
    def _predict_fine_tuned(self, prompt: str, candidate_labels: List[str]) -> Dict[str, Any]:
        """Fine-tuned model prediction (to be implemented later)"""
        raise NotImplementedError("Fine-tuned model prediction not implemented yet")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        model_name = settings.ZERO_SHOT_MODEL if self._model_type == ModelType.ZERO_SHOT else "fallback"
        
        return {
            "model_type": self._model_type.value if self._model_type else None,
            "model_name": model_name,
            "model_loaded": self._model is not None,
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        }

# Global model manager instance
model_manager = ModelManager()