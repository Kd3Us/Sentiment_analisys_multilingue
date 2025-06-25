"""
Pydantic models for the API.
These models define data schemas for requests and responses.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from enum import Enum


class ArchitectureType(str, Enum):
    """Supported architecture types"""
    LSTM = "lstm"
    CNN = "cnn"
    TRANSFORMER = "transformer"
    HYBRID = "hybrid"


class SentimentLevel(int, Enum):
    """Predefined sentiment levels"""
    VERY_NEGATIVE = -2
    NEGATIVE = -1
    NEUTRAL = 0
    POSITIVE = 1
    VERY_POSITIVE = 2


class TrainingDataItem(BaseModel):
    """Training data item"""
    text: str = Field(..., description="Text to analyze", min_length=1, max_length=1000)
    sentiment: int = Field(..., description="Sentiment level", ge=-10, le=10)
    language: Optional[str] = Field(None, description="Language code (e.g., 'fr', 'en')")
    domain: Optional[str] = Field(None, description="Business domain")
    confidence: Optional[float] = Field(None, description="Annotation confidence", ge=0.0, le=1.0)
    
    @validator('text')
    def validate_text(cls, v):
        if not v or v.isspace():
            raise ValueError('Text cannot be empty')
        return v.strip()


class ModelConfiguration(BaseModel):
    """Client model configuration"""
    architecture: ArchitectureType = Field(ArchitectureType.LSTM, description="Architecture type")
    sentiment_levels: int = Field(5, description="Number of sentiment levels", ge=2, le=10)
    languages: List[str] = Field(default=["fr", "en"], description="Supported languages")
    domain: Optional[str] = Field(None, description="Application domain")
    
    # Hyperparameters
    batch_size: int = Field(32, description="Batch size", ge=1, le=256)
    learning_rate: float = Field(0.001, description="Learning rate", gt=0.0, le=1.0)
    epochs: int = Field(50, description="Number of epochs", ge=1, le=1000)
    validation_split: float = Field(0.2, description="Validation proportion", gt=0.0, lt=1.0)
    early_stopping_patience: int = Field(10, description="Early stopping patience", ge=1)
    
    # Architecture specific
    embed_dim: int = Field(300, description="Embedding dimension", ge=50, le=1000)
    hidden_dim: int = Field(256, description="Hidden dimension", ge=50, le=2048)
    dropout: float = Field(0.3, description="Dropout rate", ge=0.0, le=0.9)


class TrainingRequest(BaseModel):
    """Training request"""
    data: List[TrainingDataItem] = Field(..., description="Training data", min_items=10)
    config: ModelConfiguration = Field(..., description="Model configuration")
    
    @validator('data')
    def validate_data_distribution(cls, v):
        # Check sufficient data per class
        sentiment_counts = {}
        for item in v:
            sentiment_counts[item.sentiment] = sentiment_counts.get(item.sentiment, 0) + 1
        
        if len(sentiment_counts) < 2:
            raise ValueError('At least 2 different sentiment classes required')
        
        # Check minimum 5 examples per class
        for sentiment, count in sentiment_counts.items():
            if count < 5:
                raise ValueError(f'At least 5 examples required for sentiment {sentiment}')
        
        return v


class PredictionRequest(BaseModel):
    """Prediction request"""
    text: str = Field(..., description="Text to analyze", min_length=1, max_length=1000)
    language: Optional[str] = Field(None, description="Text language (auto-detection if None)")
    return_probabilities: bool = Field(False, description="Return probabilities")
    
    @validator('text')
    def validate_text(cls, v):
        if not v or v.isspace():
            raise ValueError('Text cannot be empty')
        return v.strip()


class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    texts: List[str] = Field(..., description="Texts to analyze", min_items=1, max_items=1000)
    languages: Optional[List[str]] = Field(None, description="Text languages")
    return_probabilities: bool = Field(False, description="Return probabilities")
    
    @validator('texts')
    def validate_texts(cls, v):
        cleaned_texts = []
        for text in v:
            if not text or text.isspace():
                raise ValueError('All texts must be non-empty')
            cleaned_texts.append(text.strip())
        return cleaned_texts
    
    @validator('languages')
    def validate_languages_length(cls, v, values):
        if v is not None and 'texts' in values:
            if len(v) != len(values['texts']):
                raise ValueError('Number of languages must match number of texts')
        return v


class PredictionResult(BaseModel):
    """Prediction result"""
    text: str = Field(..., description="Analyzed text")
    sentiment: int = Field(..., description="Predicted sentiment")
    confidence: float = Field(..., description="Prediction confidence")
    language_detected: str = Field(..., description="Detected language")
    probabilities: Optional[Dict[str, float]] = Field(None, description="Class probabilities")
    processing_time: float = Field(..., description="Processing time in seconds")


class BatchPredictionResult(BaseModel):
    """Batch prediction result"""
    results: List[PredictionResult] = Field(..., description="Individual results")
    total_processing_time: float = Field(..., description="Total processing time")
    average_confidence: float = Field(..., description="Average confidence")


class TrainingStatus(BaseModel):
    """Training status"""
    client_id: str = Field(..., description="Client ID")
    status: str = Field(..., description="Current status")
    progress: float = Field(..., description="Progress (0.0 to 1.0)")
    current_epoch: Optional[int] = Field(None, description="Current epoch")
    total_epochs: Optional[int] = Field(None, description="Total epochs")
    train_loss: Optional[float] = Field(None, description="Training loss")
    val_loss: Optional[float] = Field(None, description="Validation loss")
    val_accuracy: Optional[float] = Field(None, description="Validation accuracy")
    estimated_time_remaining: Optional[int] = Field(None, description="Estimated remaining time (seconds)")
    message: Optional[str] = Field(None, description="Informative message")


class TrainingResult(BaseModel):
    """Training result"""
    client_id: str = Field(..., description="Client ID")
    success: bool = Field(..., description="Training success")
    training_time: float = Field(..., description="Total training time")
    total_epochs: int = Field(..., description="Number of epochs executed")
    vocab_size: int = Field(..., description="Vocabulary size")
    model_parameters: Dict[str, int] = Field(..., description="Number of parameters")
    best_val_accuracy: float = Field(..., description="Best validation accuracy")
    best_val_loss: float = Field(..., description="Best validation loss")
    best_epoch: int = Field(..., description="Best epoch")
    model_path: str = Field(..., description="Saved model path")
    vocab_info: Dict[str, Any] = Field(..., description="Vocabulary information")
    language_distribution: Dict[str, int] = Field(..., description="Language distribution")
    error_message: Optional[str] = Field(None, description="Error message if failed")


class ModelInfo(BaseModel):
    """Client model information"""
    client_id: str = Field(..., description="Client ID")
    model_name: Optional[str] = Field(None, description="Model name")
    architecture: str = Field(..., description="Architecture used")
    sentiment_levels: int = Field(..., description="Number of sentiment levels")
    languages: List[str] = Field(..., description="Supported languages")
    vocab_size: int = Field(..., description="Vocabulary size")
    training_date: str = Field(..., description="Training date")
    accuracy: float = Field(..., description="Model accuracy")
    status: str = Field(..., description="Model status (active, training, error)")
    version: str = Field("1.0", description="Model version")


class DataValidationResult(BaseModel):
    """Data validation result"""
    is_valid: bool = Field(..., description="Data validity")
    total_samples: int = Field(..., description="Total number of samples")
    language_distribution: Dict[str, int] = Field(..., description="Language distribution")
    sentiment_distribution: Dict[str, int] = Field(..., description="Sentiment distribution")
    avg_text_length: float = Field(..., description="Average text length")
    warnings: List[str] = Field(default=[], description="Warnings")
    errors: List[str] = Field(default=[], description="Detected errors")
    recommendations: List[str] = Field(default=[], description="Recommendations")


class ErrorResponse(BaseModel):
    """Standardized error response"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional details")
    timestamp: str = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request ID")


class HealthCheck(BaseModel):
    """API health status"""
    status: str = Field("healthy", description="General status")
    version: str = Field(..., description="API version")
    uptime: float = Field(..., description="Uptime in seconds")
    models_loaded: int = Field(..., description="Number of loaded models")
    system_info: Dict[str, Any] = Field(..., description="System information")


class ClientStats(BaseModel):
    """Client statistics"""
    client_id: str = Field(..., description="Client ID")
    models_count: int = Field(..., description="Number of models")
    total_predictions: int = Field(..., description="Total predictions")
    avg_accuracy: float = Field(..., description="Average accuracy")
    languages_used: List[str] = Field(..., description="Languages used")
    last_activity: str = Field(..., description="Last activity")
    storage_used: float = Field(..., description="Storage used (MB)")


# WebSocket models
class WSMessage(BaseModel):
    """Generic WebSocket message"""
    type: str = Field(..., description="Message type")
    client_id: str = Field(..., description="Client ID")
    data: Dict[str, Any] = Field(..., description="Message data")
    timestamp: str = Field(..., description="Timestamp")


class TrainingProgress(BaseModel):
    """Training progress for WebSocket"""
    epoch: int = Field(..., description="Current epoch")
    total_epochs: int = Field(..., description="Total epochs")
    train_loss: float = Field(..., description="Training loss")
    val_loss: float = Field(..., description="Validation loss")
    val_accuracy: float = Field(..., description="Validation accuracy")
    learning_rate: float = Field(..., description="Current learning rate")
    time_elapsed: float = Field(..., description="Elapsed time")
    eta: Optional[float] = Field(None, description="Estimated remaining time")


# Advanced configuration models
class AdvancedModelConfig(BaseModel):
    """Advanced configuration for expert users"""
    # Custom optimizer
    optimizer_type: str = Field("adamw", description="Optimizer type")
    weight_decay: float = Field(1e-5, description="Weight decay")
    beta1: float = Field(0.9, description="Beta1 for Adam")
    beta2: float = Field(0.999, description="Beta2 for Adam")
    
    # Scheduler
    scheduler_type: str = Field("plateau", description="Scheduler type")
    scheduler_factor: float = Field(0.7, description="LR reduction factor")
    scheduler_patience: int = Field(5, description="Scheduler patience")
    
    # Data augmentation
    data_augmentation: bool = Field(False, description="Enable augmentation")
    augmentation_prob: float = Field(0.1, description="Augmentation probability")
    
    # Regularization
    gradient_clip_norm: float = Field(1.0, description="Gradient clipping")
    label_smoothing: float = Field(0.0, description="Label smoothing")
    
    # Architecture specific
    lstm_layers: Optional[int] = Field(None, description="Number of LSTM layers")
    attention_heads: Optional[int] = Field(None, description="Attention heads")
    transformer_layers: Optional[int] = Field(None, description="Transformer layers")


class ExportRequest(BaseModel):
    """Model export request"""
    format: str = Field("pytorch", description="Export format (pytorch, onnx, tflite)")
    include_processor: bool = Field(True, description="Include processor")
    optimize: bool = Field(False, description="Optimize for production")
    quantize: bool = Field(False, description="Model quantization")


class ExportResult(BaseModel):
    """Export result"""
    success: bool = Field(..., description="Export success")
    file_path: str = Field(..., description="Exported file path")
    file_size: int = Field(..., description="File size in bytes")
    format: str = Field(..., description="Export format")
    metadata: Dict[str, Any] = Field(..., description="Model metadata")


# Custom validation for sentiment ranges
def validate_sentiment_range(sentiment_levels: int, sentiment_value: int) -> bool:
    """Validates that sentiment value is in correct range"""
    if sentiment_levels == 5:
        return sentiment_value in [-2, -1, 0, 1, 2]
    elif sentiment_levels == 3:
        return sentiment_value in [-1, 0, 1]
    else:
        # For custom number of levels
        half_range = sentiment_levels // 2
        if sentiment_levels % 2 == 1:  # Odd, with neutral class
            return -half_range <= sentiment_value <= half_range
        else:  # Even, without explicit neutral class
            return -(sentiment_levels//2) <= sentiment_value < (sentiment_levels//2)


# Configuration for API documentation examples
class Config:
    schema_extra = {
        "example": {
            "text": "This product is absolutely fantastic!",
            "sentiment": 2,
            "language": "en",
            "confidence": 0.95
        }
    }