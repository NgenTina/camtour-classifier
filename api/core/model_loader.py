from transformers import pipeline
import torch
import logging
from typing import Dict, Any, List, Optional
from .config import settings, ModelType

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
            raise
    
    def _load_zero_shot_model(self):
        """Load the zero-shot classification model"""
        device = 0 if torch.cuda.is_available() else -1
        self._model = pipeline(
            "zero-shot-classification",
            model=settings.ZERO_SHOT_MODEL,
            device=device,
            clean_up_tokenization_spaces=True
        )
    
    def _load_fine_tuned_model(self):
        """Load a fine-tuned model (to be implemented later)"""
        # Placeholder for fine-tuned model loading
        # This will be implemented when you have fine-tuned models
        raise NotImplementedError("Fine-tuned model loading not implemented yet")
    
    def predict(self, prompt: str, candidate_labels: List[str]) -> Dict[str, Any]:
        """Make a prediction using the loaded model"""
        if self._model is None:
            raise ValueError("Model not initialized. Call initialize_model() first.")
        
        if self._model_type == ModelType.ZERO_SHOT:
            return self._predict_zero_shot(prompt, candidate_labels)
        elif self._model_type == ModelType.FINE_TUNED:
            return self._predict_fine_tuned(prompt, candidate_labels)
        else:
            raise ValueError(f"Unknown model type: {self._model_type}")
    
    def _predict_zero_shot(self, prompt: str, candidate_labels: List[str]) -> Dict[str, Any]:
        """Zero-shot prediction using transformers pipeline"""
        result = self._model(prompt, candidate_labels)
        
        return {
            "sequence": result["sequence"],
            "labels": result["labels"],
            "scores": result["scores"],
            "prediction": result["labels"][0],  # Top prediction
            "confidence": result["scores"][0],  # Top confidence
            "is_tourism": result["labels"][0] == "tourism",
            "model_type": "zero_shot"
        }
    
    def _predict_fine_tuned(self, prompt: str, candidate_labels: List[str]) -> Dict[str, Any]:
        """Fine-tuned model prediction (to be implemented later)"""
        # Placeholder for fine-tuned model prediction
        raise NotImplementedError("Fine-tuned model prediction not implemented yet")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            "model_type": self._model_type.value if self._model_type else None,
            "model_loaded": self._model is not None,
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        }

# Global model manager instance
model_manager = ModelManager()