"""
Sentiment AI Platform - Main application entry point.
Professional sentiment analysis platform with customizable neural models.
"""

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import time
from datetime import datetime
from loguru import logger
import sys
from pathlib import Path

# Configure logging for production
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)
logger.add(
    "logs/app.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="DEBUG",
    rotation="100 MB",
    retention="30 days"
)

from app.config.settings import settings
from app.api.routes import router
from app.api.models import ErrorResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle manager"""
    
    # Startup
    logger.info("Starting Sentiment AI Platform")
    
    # Create required directories
    Path(settings.MODELS_PATH).mkdir(parents=True, exist_ok=True)
    Path(settings.DATA_PATH).mkdir(parents=True, exist_ok=True)
    Path(settings.LOGS_PATH).mkdir(parents=True, exist_ok=True)
    
    # Verify dependencies
    try:
        import torch
        logger.info(f"PyTorch {torch.__version__} - CUDA: {torch.cuda.is_available()}")
        
        import nltk
        logger.info("NLTK available")
        
        import spacy
        logger.info("spaCy available")
        
        from langdetect import detect
        logger.info("Language detection available")
        
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        raise
    
    logger.info("All dependencies ready")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Sentiment AI Platform")


# Create FastAPI application
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="""
    # Sentiment AI Platform
    
    Professional multilingual sentiment analysis platform with customizable neural models.
    
    ## Key Features
    
    * **Custom Neural Models** - LSTM, CNN, Transformer, Hybrid architectures
    * **Multilingual Support** - Automatic language detection and preprocessing
    * **Fine-grained Analysis** - 5-level sentiment classification
    * **Client-specific Training** - Each client trains their own model
    * **Real-time Monitoring** - Training progress via WebSocket
    * **Complete REST API** - Simple and powerful interface
    
    ## Typical Workflow
    
    1. **Data Upload** - Annotated training data
    2. **Model Configuration** - Architecture and hyperparameters
    3. **Training** - Real-time progress monitoring
    4. **Prediction** - Inference on new texts
    5. **Performance Monitoring** - Metrics and analytics
    
    ## Available Architectures
    
    * **LSTM** - Bidirectional recurrent networks with attention
    * **CNN** - 1D convolutions with multiple kernel sizes
    * **Transformer** - Pure attention architecture
    * **Hybrid** - Combined CNN + LSTM approach
    """,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure according to your needs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Compression middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Request logging middleware
@app.middleware("http")
async def log_requests(request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.3f}s"
    )
    
    return response

# Global exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=f"HTTP_{exc.status_code}",
            message=str(exc.detail),
            timestamp=datetime.now().isoformat(),
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="INTERNAL_SERVER_ERROR", 
            message="An internal error occurred",
            timestamp=datetime.now().isoformat(),
        ).dict()
    )

# Include routes
app.include_router(router, tags=["Sentiment Analysis"])

# Base routes
@app.get("/", tags=["Root"])
async def root():
    """API home page"""
    return {
        "message": "Welcome to Sentiment AI Platform",
        "version": settings.VERSION,
        "docs": "/docs",
        "status": "running",
        "features": [
            "Custom neural models",
            "Multilingual support", 
            "5-level sentiment analysis",
            "Personalized training",
            "Complete REST API"
        ]
    }

@app.get("/info", tags=["Info"])
async def get_info():
    """Detailed platform information"""
    import torch
    
    return {
        "platform": {
            "name": settings.PROJECT_NAME,
            "version": settings.VERSION,
            "debug": settings.DEBUG
        },
        "supported_features": {
            "architectures": settings.AVAILABLE_ARCHITECTURES,
            "languages": settings.SUPPORTED_LANGUAGES,
            "sentiment_levels": f"{settings.MIN_SENTIMENT_LEVELS}-{settings.MAX_SENTIMENT_LEVELS}",
            "file_formats": settings.ALLOWED_FILE_TYPES
        },
        "system": {
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_devices": torch.cuda.device_count() if torch.cuda.is_available() else 0
        },
        "limits": {
            "max_upload_size_mb": settings.MAX_UPLOAD_SIZE_MB,
            "max_training_time_hours": settings.MAX_TRAINING_TIME_HOURS,
            "min_training_samples": settings.MIN_TRAINING_SAMPLES
        }
    }


# Development entry point
if __name__ == "__main__":
    logger.info(f"Starting server on {settings.HOST}:{settings.PORT}")
    
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=1,  # Important for in-memory storage
        log_level="info"
    )