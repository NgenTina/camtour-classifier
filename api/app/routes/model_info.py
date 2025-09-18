from fastapi import APIRouter
from api.core.model_manager import model_manager

router = APIRouter()


@router.get("/model-info", summary="Get model information", description="Returns the name of the model currently in use.")
async def get_model_info():
    try:
        model_info = model_manager.get_model_info()
        return {"model_name": model_info}
    except Exception as e:
        return {"error": str(e)}
