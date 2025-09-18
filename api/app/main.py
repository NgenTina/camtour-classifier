from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import predict, health, model_info
from api.core.config import settings
from api.core.model_manager import model_manager


def create_application() -> FastAPI:
    application = FastAPI(
        title=settings.PROJECT_NAME,
        description="API for zero-shot classification of tourism content",
        version=settings.VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Configure CORS
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    application.include_router(health.router, tags=["health"])
    application.include_router(
        predict.router, prefix=settings.API_V1_STR, tags=["predict"])
    application.include_router(
        model_info.router, prefix="/api", tags=["model-info"])

    # Initialize model on startup
    @application.on_event("startup")
    async def startup_event():
        try:
            model_manager.initialize_model()
            print(
                f"\033[95mSUCCESS\033[0m: Model loaded successfully: {model_manager.get_model_info()}")
        except Exception as e:
            print(f"\033[91mFAIL: Failed to load model: {e}\033[0m")

    return application


app = create_application()
