from pydantic_settings import BaseSettings
from typing import List, Optional
import os


class Settings(BaseSettings):
    
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Sentiment AI Platform"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    DATABASE_URL: str = "postgresql://user:password@localhost/sentiment_db"
    REDIS_URL: str = "redis://localhost:6379"
    POSTGRES_PASSWORD: str = "default_password"
    
    DEFAULT_EMBEDDING_DIM: int = 300
    DEFAULT_HIDDEN_DIM: int = 256
    MAX_SEQUENCE_LENGTH: int = 512
    DEFAULT_BATCH_SIZE: int = 32
    DEFAULT_EPOCHS: int = 50
    
    SUPPORTED_LANGUAGES: List[str] = [
        "fr", "en", "es", "de", "it", "pt", "nl", "ru", "zh", "ja"
    ]
    
    AVAILABLE_ARCHITECTURES: List[str] = ["lstm", "cnn", "transformer", "hybrid"]
    
    MIN_SENTIMENT_LEVELS: int = 2
    MAX_SENTIMENT_LEVELS: int = 10
    DEFAULT_SENTIMENT_LEVELS: int = 5
    
    MAX_TRAINING_TIME_HOURS: int = 24
    MIN_TRAINING_SAMPLES: int = 100
    VALIDATION_SPLIT: float = 0.2
    
    MAX_UPLOAD_SIZE_MB: int = 100
    ALLOWED_FILE_TYPES: List[str] = [".csv", ".json", ".xlsx", ".txt"]
    
    SECRET_KEY: str = "your-secret-key-here"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_REGION: str = "us-east-1"
    S3_BUCKET_NAME: Optional[str] = None
    
    ENABLE_METRICS: bool = True
    LOG_LEVEL: str = "INFO"
    
    MODELS_PATH: str = "models"
    DATA_PATH: str = "data"
    LOGS_PATH: str = "logs"
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "allow"


class ModelConfig:
    
    LSTM_CONFIG = {
        "hidden_dim": 256,
        "num_layers": 2,
        "bidirectional": True,
        "dropout": 0.3
    }
    
    CNN_CONFIG = {
        "filters": [100, 100, 100],
        "kernel_sizes": [3, 4, 5],
        "dropout": 0.5
    }
    
    TRANSFORMER_CONFIG = {
        "num_heads": 8,
        "num_layers": 6,
        "dim_feedforward": 2048,
        "dropout": 0.1
    }
    
    OPTIMIZER_CONFIG = {
        "lr": 0.001,
        "weight_decay": 1e-5,
        "betas": (0.9, 0.999)
    }


settings = Settings()
model_config = ModelConfig()


def get_settings() -> Settings:
    return settings


def get_model_config() -> ModelConfig:
    return model_config