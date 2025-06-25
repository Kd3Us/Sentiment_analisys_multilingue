"""
API routes for the sentiment analysis platform.
This module defines all REST API endpoints.
"""

import asyncio
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from loguru import logger
import torch

from .models import *
from ..services.trainer import ModelTrainer, TrainingConfig
from ..core.data_processor import MultilingualProcessor
from ..config.settings import settings

# Main router
router = APIRouter(prefix=settings.API_V1_STR)

# In-memory storage for active models and training status
active_models: Dict[str, ModelTrainer] = {}
training_status: Dict[str, TrainingStatus] = {}
client_stats: Dict[str, ClientStats] = {}

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Client {client_id} connected via WebSocket")

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"Client {client_id} disconnected")

    async def send_message(self, client_id: str, message: dict):
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_json(message)
            except Exception as e:
                logger.error(f"WebSocket message error for {client_id}: {e}")
                self.disconnect(client_id)

manager = ConnectionManager()


# Utility functions
def get_current_timestamp() -> str:
    """Returns current timestamp"""
    return datetime.now().isoformat()


def generate_request_id() -> str:
    """Generates unique request ID"""
    return str(uuid.uuid4())


async def validate_client_data(data: List[TrainingDataItem]) -> DataValidationResult:
    """Validates client training data"""
    warnings = []
    errors = []
    recommendations = []
    
    # Basic statistics
    total_samples = len(data)
    language_dist = {}
    sentiment_dist = {}
    text_lengths = []
    
    for item in data:
        # Language distribution
        lang = item.language or "unknown"
        language_dist[lang] = language_dist.get(lang, 0) + 1
        
        # Sentiment distribution (convert keys to strings)
        sentiment_str = str(item.sentiment)
        sentiment_dist[sentiment_str] = sentiment_dist.get(sentiment_str, 0) + 1
        
        # Text lengths
        text_lengths.append(len(item.text))
    
    avg_text_length = sum(text_lengths) / len(text_lengths)
    
    # Validations
    if total_samples < 100:
        warnings.append(f"Limited data ({total_samples}). Recommended: >500 samples")
    
    if len(sentiment_dist) < 3:
        warnings.append("Few sentiment classes. Consider diversifying your data.")
    
    # Check class balance
    min_count = min(sentiment_dist.values())
    max_count = max(sentiment_dist.values())
    if max_count / min_count > 5:
        warnings.append("Imbalanced classes detected")
        recommendations.append("Balance your data or use resampling techniques")
    
    # Check text lengths
    if avg_text_length < 10:
        warnings.append("Very short texts detected")
    elif avg_text_length > 500:
        warnings.append("Very long texts detected")
        recommendations.append("Consider truncating long texts")
    
    # Multiple languages
    if len(language_dist) > 1:
        recommendations.append("Multilingual data detected - ensure sufficient examples per language")
    
    is_valid = len(errors) == 0
    
    return DataValidationResult(
        is_valid=is_valid,
        total_samples=total_samples,
        language_distribution=language_dist,
        sentiment_distribution=sentiment_dist,
        avg_text_length=avg_text_length,
        warnings=warnings,
        errors=errors,
        recommendations=recommendations
    )


# ENDPOINTS

@router.get("/health", response_model=HealthCheck)
async def health_check():
    """API health check"""
    return HealthCheck(
        status="healthy",
        version=settings.VERSION,
        uptime=time.time(),  # Simplified for example
        models_loaded=len(active_models),
        system_info={
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "pytorch_version": torch.__version__
        }
    )


@router.post("/clients/{client_id}/validate-data", response_model=DataValidationResult)
async def validate_training_data(client_id: str, data: List[TrainingDataItem]):
    """Validates training data before launching training"""
    try:
        result = await validate_client_data(data)
        logger.info(f"Data validation for client {client_id}: {result.total_samples} samples")
        return result
    except Exception as e:
        logger.error(f"Data validation error for {client_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Validation error: {str(e)}")


@router.post("/clients/{client_id}/train", response_model=dict)
async def start_training(
    client_id: str, 
    request: TrainingRequest, 
    background_tasks: BackgroundTasks
):
    """Starts model training for a client"""
    
    # Check if training is already in progress
    if client_id in training_status and training_status[client_id].status == "training":
        raise HTTPException(
            status_code=409, 
            detail="Training already in progress for this client"
        )
    
    try:
        # Validate data
        validation_result = await validate_client_data(request.data)
        if not validation_result.is_valid:
            raise HTTPException(
                status_code=400,
                detail={"validation_errors": validation_result.errors}
            )
        
        # Create training configuration
        training_config = TrainingConfig(
            client_id=client_id,
            architecture=request.config.architecture.value,
            sentiment_levels=request.config.sentiment_levels,
            languages=request.config.languages,
            batch_size=request.config.batch_size,
            learning_rate=request.config.learning_rate,
            epochs=request.config.epochs,
            validation_split=request.config.validation_split,
            early_stopping_patience=request.config.early_stopping_patience,
            embed_dim=request.config.embed_dim,
            hidden_dim=request.config.hidden_dim,
            dropout=request.config.dropout
        )
        
        # Initialize status
        training_status[client_id] = TrainingStatus(
            client_id=client_id,
            status="initializing",
            progress=0.0,
            message="Initializing training..."
        )
        
        # Launch background training
        background_tasks.add_task(
            run_training_background,
            client_id,
            training_config,
            [item.dict() for item in request.data]
        )
        
        logger.info(f"Training started for client {client_id}")
        
        return {
            "message": "Training started successfully",
            "client_id": client_id,
            "status": "training",
            "estimated_time": request.config.epochs * 2
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Training start error for {client_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


async def run_training_background(client_id: str, config: TrainingConfig, data: List[Dict]):
    """Executes training in background with real-time monitoring"""
    try:
        # Update status
        training_status[client_id].status = "training"
        training_status[client_id].message = "Training in progress..."
        
        # Create trainer
        trainer = ModelTrainer(config)
        
        # Start training
        result = trainer.train(data)
        
        # Save active model
        active_models[client_id] = trainer
        
        # Update final status
        training_status[client_id].status = "completed"
        training_status[client_id].progress = 1.0
        training_status[client_id].message = "Training completed successfully"
        
        # Notify via WebSocket
        await manager.send_message(client_id, {
            "type": "training_completed",
            "data": result
        })
        
        logger.info(f"Training completed for {client_id}")
        
    except Exception as e:
        logger.error(f"Background training error for {client_id}: {str(e)}")
        training_status[client_id].status = "error"
        training_status[client_id].message = f"Error: {str(e)}"
        
        await manager.send_message(client_id, {
            "type": "training_error",
            "data": {"error": str(e)}
        })


@router.get("/clients/{client_id}/training-status", response_model=TrainingStatus)
async def get_training_status(client_id: str):
    """Retrieves training status for a client"""
    if client_id not in training_status:
        raise HTTPException(status_code=404, detail="No training found for this client")
    
    return training_status[client_id]


@router.post("/clients/{client_id}/predict", response_model=PredictionResult)
async def predict_sentiment(client_id: str, request: PredictionRequest):
    """Predicts sentiment for a text"""
    
    # Check if model exists
    if client_id not in active_models:
        raise HTTPException(status_code=404, detail="Model not found. Please train a model first.")
    
    try:
        start_time = time.time()
        trainer = active_models[client_id]
        
        # Prepare text
        if request.language:
            detected_language = request.language
        else:
            detected_language = trainer.processor.detect_language(request.text)
        
        # Prediction
        input_ids = trainer.processor.text_to_indices(request.text, detected_language)
        input_tensor = torch.tensor([input_ids]).to(trainer.device)
        attention_mask = (input_tensor != 0).float()
        
        trainer.model.eval()
        with torch.no_grad():
            outputs = trainer.model(input_tensor, attention_mask)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        processing_time = time.time() - start_time
        
        # Prepare response
        result = PredictionResult(
            text=request.text,
            sentiment=predicted_class - 2,  # Convert [0,4] back to [-2,+2]
            confidence=confidence,
            language_detected=detected_language,
            processing_time=processing_time
        )
        
        # Add probabilities if requested
        if request.return_probabilities:
            prob_dict = {}
            for i, prob in enumerate(probabilities[0]):
                sentiment_level = i - 2  # Convert [0,4] back to [-2,+2]
                prob_dict[str(sentiment_level)] = prob.item()
            result.probabilities = prob_dict
        
        # Update client statistics
        if client_id not in client_stats:
            client_stats[client_id] = ClientStats(
                client_id=client_id,
                models_count=1,
                total_predictions=0,
                avg_accuracy=0.0,
                languages_used=[],
                last_activity=get_current_timestamp(),
                storage_used=0.0
            )
        
        client_stats[client_id].total_predictions += 1
        client_stats[client_id].last_activity = get_current_timestamp()
        if detected_language not in client_stats[client_id].languages_used:
            client_stats[client_id].languages_used.append(detected_language)
        
        return result
        
    except Exception as e:
        logger.error(f"Prediction error for {client_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@router.post("/clients/{client_id}/batch-predict", response_model=BatchPredictionResult)
async def batch_predict_sentiment(client_id: str, request: BatchPredictionRequest):
    """Predicts sentiment for multiple texts at once"""
    
    if client_id not in active_models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        start_time = time.time()
        trainer = active_models[client_id]
        results = []
        
        for i, text in enumerate(request.texts):
            # Specific language or automatic detection
            if request.languages and i < len(request.languages):
                language = request.languages[i]
            else:
                language = trainer.processor.detect_language(text)
            
            # Individual prediction
            pred_request = PredictionRequest(
                text=text,
                language=language,
                return_probabilities=request.return_probabilities
            )
            
            result = await predict_sentiment(client_id, pred_request)
            results.append(result)
        
        total_time = time.time() - start_time
        avg_confidence = sum(r.confidence for r in results) / len(results)
        
        return BatchPredictionResult(
            results=results,
            total_processing_time=total_time,
            average_confidence=avg_confidence
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error for {client_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/clients/{client_id}/model-info", response_model=ModelInfo)
async def get_model_info(client_id: str):
    """Retrieves information about a client's model"""
    
    if client_id not in active_models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    trainer = active_models[client_id]
    
    return ModelInfo(
        client_id=client_id,
        architecture=trainer.config.architecture,
        sentiment_levels=trainer.config.sentiment_levels,
        languages=trainer.config.languages or ["auto-detect"],
        vocab_size=trainer.processor.vocab_size if trainer.processor else 0,
        training_date=get_current_timestamp(),  # Should be improved with actual date
        accuracy=trainer.metrics.best_val_acc if hasattr(trainer, 'metrics') else 0.0,
        status="active"
    )


@router.get("/clients/{client_id}/stats", response_model=ClientStats)
async def get_client_stats(client_id: str):
    """Retrieves client statistics"""
    
    if client_id not in client_stats:
        raise HTTPException(status_code=404, detail="Statistics not found")
    
    return client_stats[client_id]


@router.delete("/clients/{client_id}/model")
async def delete_client_model(client_id: str):
    """Deletes a client's model"""
    
    if client_id not in active_models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        # Remove from memory
        del active_models[client_id]
        
        # Delete files
        model_dir = Path(settings.MODELS_PATH) / client_id
        if model_dir.exists():
            import shutil
            shutil.rmtree(model_dir)
        
        # Clean up status
        if client_id in training_status:
            del training_status[client_id]
        
        logger.info(f"Model {client_id} deleted")
        
        return {"message": f"Model {client_id} deleted successfully"}
        
    except Exception as e:
        logger.error(f"Model deletion error for {client_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket for real-time training monitoring"""
    await manager.connect(websocket, client_id)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(client_id)