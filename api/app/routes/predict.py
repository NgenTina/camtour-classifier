from fastapi import APIRouter, HTTPException
from typing import List
import logging

from ...core.model_manager import model_manager
from ...core.config import ModelType
from ..models import PredictionRequest, PredictionResponse, ErrorResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def predict(request: PredictionRequest):
    """
    Classify a text prompt as tourism or not_tourism using zero-shot classification.

    Example:
    - Prompt: "Can I find my soulmate if I go to Siem Reap?"
    - Returns: tourism with confidence score
    """
    try:
        # Initialize model if not already loaded (with optional override)
        if model_manager._model is None:
            model_type = ModelType(
                request.model_type) if request.model_type else None
            model_manager.initialize_model(model_type)

        # Make prediction
        result = model_manager.predict(
            request.prompt, request.candidate_labels)

        # Add model info
        result["model_info"] = model_manager.get_model_info()

        return PredictionResponse(**result)

    except NotImplementedError as e:
        logger.error(f"Model not implemented: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail={
                "detail": str(e),
                "error_type": "model_not_implemented"
            }
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "detail": f"Error making prediction: {str(e)}",
                "error_type": "prediction_error"
            }
        )


@router.post(
    "/predict/batch",
    response_model=List[PredictionResponse],
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def predict_batch(requests: List[PredictionRequest]):
    """
    Classify multiple text prompts in batch.
    """
    try:
        results = []
        for request in requests:
            result = model_manager.predict(
                request.prompt, request.candidate_labels)
            result["model_info"] = model_manager.get_model_info()
            results.append(PredictionResponse(**result))

        return results

    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "detail": f"Error making batch prediction: {str(e)}",
                "error_type": "batch_prediction_error"
            }
        )
