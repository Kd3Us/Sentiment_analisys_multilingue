import asyncio
import time
import uuid
import os
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, WebSocket, WebSocketDisconnect, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from loguru import logger
import torch

from .models import *
from ..services.trainer import ModelTrainer, TrainingConfig
from ..core.data_processor import MultilingualProcessor
from ..config.settings import settings

try:
    from ..aws.s3_integration import S3ModelManager, S3ModelTrainer, get_s3_config
    S3_AVAILABLE = True
except ImportError:
    logger.warning("S3 integration not available - running in local mode")
    S3_AVAILABLE = False

router = APIRouter(prefix=settings.API_V1_STR)

active_models: Dict[str, ModelTrainer] = {}
training_status: Dict[str, TrainingStatus] = {}
client_stats: Dict[str, ClientStats] = {}

s3_manager = None
if S3_AVAILABLE:
    try:
        s3_config = get_s3_config()
        if s3_config['bucket_name'] and s3_config['access_key_id']:
            s3_manager = S3ModelManager(
                bucket_name=s3_config['bucket_name'],
                region=s3_config['region']
            )
            logger.info("S3 integration enabled")
    except Exception as e:
        logger.warning(f"S3 setup failed: {e}. Running in local mode.")

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

def get_current_timestamp() -> str:
    return datetime.now().isoformat()

def generate_request_id() -> str:
    return str(uuid.uuid4())

async def validate_client_data(data: List[TrainingDataItem]) -> DataValidationResult:
    warnings = []
    errors = []
    recommendations = []
    
    total_samples = len(data)
    language_dist = {}
    sentiment_dist = {}
    text_lengths = []
    
    for item in data:
        lang = item.language or "unknown"
        language_dist[lang] = language_dist.get(lang, 0) + 1
        
        sentiment_str = str(item.sentiment)
        sentiment_dist[sentiment_str] = sentiment_dist.get(sentiment_str, 0) + 1
        
        text_lengths.append(len(item.text))
    
    avg_text_length = sum(text_lengths) / len(text_lengths)
    
    if total_samples < 100:
        warnings.append(f"Limited data ({total_samples}). Recommended: >500 samples")
    
    if len(sentiment_dist) < 3:
        warnings.append("Few sentiment classes. Consider diversifying your data.")
    
    min_count = min(sentiment_dist.values())
    max_count = max(sentiment_dist.values())
    if max_count / min_count > 5:
        warnings.append("Imbalanced classes detected")
        recommendations.append("Balance your data or use resampling techniques")
    
    if avg_text_length < 10:
        warnings.append("Very short texts detected")
    elif avg_text_length > 500:
        warnings.append("Very long texts detected")
        recommendations.append("Consider truncating long texts")
    
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

@router.get("/health", response_model=HealthCheck)
async def health_check():
    system_info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "pytorch_version": torch.__version__,
        "s3_available": s3_manager is not None,
        "storage_mode": "S3" if s3_manager else "Local"
    }
    
    if s3_manager:
        try:
            s3_manager.s3_client.head_bucket(Bucket=s3_manager.bucket_name)
            system_info["s3_status"] = "healthy"
        except Exception as e:
            system_info["s3_status"] = f"error: {str(e)}"
    
    return HealthCheck(
        status="healthy",
        version=settings.VERSION,
        uptime=time.time(),
        models_loaded=len(active_models),
        system_info=system_info
    )

@router.post("/clients/{client_id}/validate-data", response_model=DataValidationResult)
async def validate_training_data(client_id: str, data: List[TrainingDataItem]):
    try:
        result = await validate_client_data(data)
        logger.info(f"Data validation for client {client_id}: {result.total_samples} samples")
        return result
    except Exception as e:
        logger.error(f"Data validation error for {client_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Validation error: {str(e)}")

@router.post("/clients/{client_id}/upload-data")
async def upload_training_data(
    client_id: str,
    file: UploadFile = File(...),
    language: Optional[str] = None
):
    try:
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in settings.ALLOWED_FILE_TYPES:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type. Allowed: {settings.ALLOWED_FILE_TYPES}"
            )
        
        upload_dir = Path(settings.DATA_PATH) / client_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = upload_dir / file.filename
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        if file_ext == '.csv':
            import pandas as pd
            df = pd.read_csv(file_path)
            
            required_cols = ['text', 'sentiment']
            if not all(col in df.columns for col in required_cols):
                raise HTTPException(
                    status_code=400,
                    detail=f"CSV must contain columns: {required_cols}"
                )
            
            training_data = []
            for _, row in df.iterrows():
                item = TrainingDataItem(
                    text=str(row['text']),
                    sentiment=int(row['sentiment']),
                    language=language or row.get('language'),
                    confidence=row.get('confidence')
                )
                training_data.append(item)
        
        elif file_ext == '.json':
            import json
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            training_data = [TrainingDataItem(**item) for item in data]
        
        elif file_ext in ['.xlsx', '.xls']:
            import pandas as pd
            df = pd.read_excel(file_path)
            
            training_data = []
            for _, row in df.iterrows():
                item = TrainingDataItem(
                    text=str(row['text']),
                    sentiment=int(row['sentiment']),
                    language=language or row.get('language'),
                    confidence=row.get('confidence')
                )
                training_data.append(item)
        
        validation_result = await validate_client_data(training_data)
        
        processed_file = upload_dir / f"processed_{file.filename}.json"
        with open(processed_file, 'w', encoding='utf-8') as f:
            import json
            json.dump([item.dict() for item in training_data], f, indent=2, ensure_ascii=False)
        
        return {
            "message": "File uploaded and processed successfully",
            "file_path": str(processed_file),
            "samples_count": len(training_data),
            "validation": validation_result.dict()
        }
        
    except Exception as e:
        logger.error(f"File upload error for {client_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")

@router.post("/clients/{client_id}/train", response_model=dict)
async def start_training(
    client_id: str, 
    request: TrainingRequest, 
    background_tasks: BackgroundTasks
):
    
    if client_id in training_status and training_status[client_id].status == "training":
        raise HTTPException(
            status_code=409, 
            detail="Training already in progress for this client"
        )
    
    try:
        validation_result = await validate_client_data(request.data)
        if not validation_result.is_valid:
            raise HTTPException(
                status_code=400,
                detail={"validation_errors": validation_result.errors}
            )
        
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
        
        training_status[client_id] = TrainingStatus(
            client_id=client_id,
            status="initializing",
            progress=0.0,
            message="Initializing training..."
        )
        
        background_tasks.add_task(
            run_training_background_with_s3,
            client_id,
            training_config,
            [item.dict() for item in request.data]
        )
        
        logger.info(f"Training started for client {client_id}")
        
        return {
            "message": "Training started successfully",
            "client_id": client_id,
            "status": "training",
            "storage_mode": "S3" if s3_manager else "Local",
            "estimated_time": request.config.epochs * 2
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Training start error for {client_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

async def run_training_background_with_s3(client_id: str, config: TrainingConfig, data: List[Dict]):
    try:
        training_status[client_id].status = "training"
        training_status[client_id].message = "Training in progress..."
        
        if s3_manager:
            trainer = S3ModelTrainer(config, s3_manager)
            result = trainer.train_and_upload(data)
            active_models[client_id] = trainer.trainer
        else:
            trainer = ModelTrainer(config)
            result = trainer.train(data)
            active_models[client_id] = trainer
        
        training_status[client_id].status = "completed"
        training_status[client_id].progress = 1.0
        training_status[client_id].message = "Training completed successfully"
        
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
    if client_id not in training_status:
        raise HTTPException(status_code=404, detail="No training found for this client")
    
    return training_status[client_id]

@router.get("/clients/{client_id}/models", response_model=List[Dict])
async def list_client_models(client_id: str):
    try:
        models = []
        
        local_dir = Path(settings.MODELS_PATH) / client_id
        if local_dir.exists():
            for model_file in local_dir.glob("*.pt"):
                stat = model_file.stat()
                models.append({
                    "model_id": model_file.stem,
                    "location": "local",
                    "path": str(model_file),
                    "size": stat.st_size,
                    "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "type": "pytorch"
                })
        
        if s3_manager:
            s3_models = s3_manager.list_client_models(client_id)
            for s3_model in s3_models:
                models.append({
                    "model_id": Path(s3_model['s3_key']).stem,
                    "location": "s3",
                    "s3_key": s3_model['s3_key'],
                    "size": s3_model['size'],
                    "created": s3_model['last_modified'],
                    "type": "pytorch",
                    "metadata": s3_model['metadata']
                })
        
        return models
        
    except Exception as e:
        logger.error(f"Error listing models for {client_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/clients/{client_id}/load-model")
async def load_model_from_storage(client_id: str, model_id: str, location: str = "auto"):
    try:
        if location == "s3" or (location == "auto" and s3_manager):
            if not s3_manager:
                raise HTTPException(status_code=400, detail="S3 not configured")
            
            s3_models = s3_manager.list_client_models(client_id)
            target_model = None
            for model in s3_models:
                if model_id in model['s3_key']:
                    target_model = model
                    break
            
            if not target_model:
                raise HTTPException(status_code=404, detail="Model not found in S3")
            
            config = TrainingConfig(client_id=client_id)
            trainer = S3ModelTrainer(config, s3_manager)
            model = trainer.load_from_s3(target_model['s3_key'])
            
            active_models[client_id] = trainer.trainer
            
        else:
            model_path = Path(settings.MODELS_PATH) / client_id / f"{model_id}.pt"
            if not model_path.exists():
                raise HTTPException(status_code=404, detail="Model not found locally")
            
            config = TrainingConfig(client_id=client_id)
            trainer = ModelTrainer(config)
            model = trainer.load_model(str(model_path))
            
            active_models[client_id] = trainer
        
        return {
            "message": "Model loaded successfully",
            "model_id": model_id,
            "location": location,
            "client_id": client_id
        }
        
    except Exception as e:
        logger.error(f"Error loading model {model_id} for {client_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/clients/{client_id}/download-model/{model_id}")
async def download_model(client_id: str, model_id: str, location: str = "auto"):
    try:
        if location == "s3" and s3_manager:
            s3_models = s3_manager.list_client_models(client_id)
            target_model = None
            for model in s3_models:
                if model_id in model['s3_key']:
                    target_model = model
                    break
            
            if not target_model:
                raise HTTPException(status_code=404, detail="Model not found in S3")
            
            download_url = s3_manager.create_presigned_url(target_model['s3_key'])
            if not download_url:
                raise HTTPException(status_code=500, detail="Failed to generate download URL")
            
            return {
                "download_url": download_url,
                "expires_in": 3600,
                "model_id": model_id,
                "location": "s3"
            }
        
        else:
            model_path = Path(settings.MODELS_PATH) / client_id / f"{model_id}.pt"
            if not model_path.exists():
                raise HTTPException(status_code=404, detail="Model not found locally")
            
            return FileResponse(
                path=model_path,
                filename=f"{client_id}_{model_id}.pt",
                media_type="application/octet-stream"
            )
        
    except Exception as e:
        logger.error(f"Error downloading model {model_id} for {client_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/clients/{client_id}/predict", response_model=PredictionResult)
async def predict_sentiment(client_id: str, request: PredictionRequest):
    
    if client_id not in active_models:
        raise HTTPException(status_code=404, detail="Model not found. Please train or load a model first.")
    
    try:
        start_time = time.time()
        trainer = active_models[client_id]
        
        if request.language:
            detected_language = request.language
        else:
            detected_language = trainer.processor.detect_language(request.text)
        
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
        
        result = PredictionResult(
            text=request.text,
            sentiment=predicted_class - 2,
            confidence=confidence,
            language_detected=detected_language,
            processing_time=processing_time
        )
        
        if request.return_probabilities:
            prob_dict = {}
            for i, prob in enumerate(probabilities[0]):
                sentiment_level = i - 2
                prob_dict[str(sentiment_level)] = prob.item()
            result.probabilities = prob_dict
        
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
    
    if client_id not in active_models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        start_time = time.time()
        trainer = active_models[client_id]
        results = []
        
        for i, text in enumerate(request.texts):
            if request.languages and i < len(request.languages):
                language = request.languages[i]
            else:
                language = trainer.processor.detect_language(text)
            
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
    
    if client_id not in active_models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    trainer = active_models[client_id]
    
    return ModelInfo(
        client_id=client_id,
        architecture=trainer.config.architecture,
        sentiment_levels=trainer.config.sentiment_levels,
        languages=trainer.config.languages or ["auto-detect"],
        vocab_size=trainer.processor.vocab_size if trainer.processor else 0,
        training_date=get_current_timestamp(),
        accuracy=trainer.metrics.best_val_acc if hasattr(trainer, 'metrics') else 0.0,
        status="active"
    )

@router.get("/clients/{client_id}/stats", response_model=ClientStats)
async def get_client_stats(client_id: str):
    
    if client_id not in client_stats:
        raise HTTPException(status_code=404, detail="Statistics not found")
    
    return client_stats[client_id]

@router.delete("/clients/{client_id}/models/{model_id}")
async def delete_model(client_id: str, model_id: str, location: str = "auto"):
    try:
        deleted_locations = []
        
        local_path = Path(settings.MODELS_PATH) / client_id / f"{model_id}.pt"
        if local_path.exists():
            local_path.unlink()
            deleted_locations.append("local")
        
        if s3_manager and (location in ["s3", "auto"]):
            s3_models = s3_manager.list_client_models(client_id)
            for model in s3_models:
                if model_id in model['s3_key']:
                    if s3_manager.delete_model(model['s3_key']):
                        deleted_locations.append("s3")
                    break
        
        if client_id in active_models:
            del active_models[client_id]
        
        if not deleted_locations:
            raise HTTPException(status_code=404, detail="Model not found")
        
        return {
            "message": f"Model {model_id} deleted successfully",
            "deleted_from": deleted_locations
        }
        
    except Exception as e:
        logger.error(f"Error deleting model {model_id} for {client_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/clients/{client_id}/model")
async def delete_client_model(client_id: str):
    
    if client_id not in active_models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        del active_models[client_id]
        
        model_dir = Path(settings.MODELS_PATH) / client_id
        if model_dir.exists():
            import shutil
            shutil.rmtree(model_dir)
        
        if client_id in training_status:
            del training_status[client_id]
        
        logger.info(f"Model {client_id} deleted")
        
        return {"message": f"Model {client_id} deleted successfully"}
        
    except Exception as e:
        logger.error(f"Model deletion error for {client_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(client_id)