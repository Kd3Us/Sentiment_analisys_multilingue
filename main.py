"""
Point d'entrée principal de l'application Sentiment AI Platform.
Lance le serveur FastAPI avec toutes les configurations nécessaires.
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

# Configuration des logs
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


# Gestionnaire de cycle de vie de l'application
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestionnaire du cycle de vie de l'application"""
    
    # Startup
    logger.info("🚀 Démarrage de Sentiment AI Platform")
    
    # Créer les dossiers nécessaires
    Path(settings.MODELS_PATH).mkdir(parents=True, exist_ok=True)
    Path(settings.DATA_PATH).mkdir(parents=True, exist_ok=True)
    Path(settings.LOGS_PATH).mkdir(parents=True, exist_ok=True)
    
    # Vérifier les dépendances
    try:
        import torch
        logger.info(f"✓ PyTorch {torch.__version__} - CUDA: {torch.cuda.is_available()}")
        
        import nltk
        logger.info("✓ NLTK disponible")
        
        import spacy
        logger.info("✓ spaCy disponible")
        
        from langdetect import detect
        logger.info("✓ Détection de langue disponible")
        
    except ImportError as e:
        logger.error(f"❌ Dépendance manquante: {e}")
        raise
    
    logger.info("✅ Toutes les dépendances sont prêtes")
    
    yield
    
    # Shutdown
    logger.info("🛑 Arrêt de Sentiment AI Platform")


# Créer l'application FastAPI
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="""
    # Sentiment AI Platform
    
    Plateforme d'analyse de sentiment multilingue avec modèles personnalisables.
    
    ## Fonctionnalités principales
    
    * 🧠 **Modèles from scratch** - Architectures LSTM, CNN, Transformer, Hybride
    * 🌍 **Multilingue** - Support automatique de multiples langues
    * 📊 **5 nuances** - Classification fine du sentiment
    * 🎯 **Personnalisable** - Chaque client entraîne son propre modèle
    * ⚡ **Temps réel** - Suivi de l'entraînement via WebSocket
    * 🔄 **API REST** - Interface simple et complète
    
    ## Workflow typique
    
    1. **Upload** des données d'entraînement annotées
    2. **Configuration** du modèle (architecture, hyperparamètres)
    3. **Entraînement** avec suivi temps réel
    4. **Prédiction** sur nouveaux textes
    5. **Monitoring** des performances
    
    ## Architectures disponibles
    
    * **LSTM** - Réseaux récurrents bidirectionnels avec attention
    * **CNN** - Convolutions 1D avec multiple kernel sizes
    * **Transformer** - Architecture d'attention pure
    * **Hybrid** - Combinaison CNN + LSTM
    """,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # À configurer selon vos besoins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware de compression
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Middleware de logging des requêtes
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

# Gestionnaire d'erreurs global
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
    logger.error(f"Erreur non gérée: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="INTERNAL_SERVER_ERROR", 
            message="Une erreur interne s'est produite",
            timestamp=datetime.now().isoformat(),
        ).dict()
    )

# Inclure les routes
app.include_router(router, tags=["Sentiment Analysis"])

# Routes de base
@app.get("/", tags=["Root"])
async def root():
    """Page d'accueil de l'API"""
    return {
        "message": "Bienvenue sur Sentiment AI Platform! 🚀",
        "version": settings.VERSION,
        "docs": "/docs",
        "status": "running",
        "features": [
            "Modèles from scratch",
            "Support multilingue", 
            "5 niveaux de sentiment",
            "Entraînement personnalisé",
            "API REST complète"
        ]
    }

@app.get("/info", tags=["Info"])
async def get_info():
    """Informations détaillées sur la plateforme"""
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


# Point d'entrée pour le développement
if __name__ == "__main__":
    logger.info(f"🎯 Lancement du serveur sur {settings.HOST}:{settings.PORT}")
    
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=1,  # Important pour le stockage en mémoire
        log_level="info"
    )