"""
Routes API pour la plateforme d'analyse de sentiment.
Ce module définit tous les endpoints de l'API REST.
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

# Router principal
router = APIRouter(prefix=settings.API_V1_STR)

# Stockage en mémoire des modèles actifs et des statuts d'entraînement
active_models: Dict[str, ModelTrainer] = {}
training_status: Dict[str, TrainingStatus] = {}
client_stats: Dict[str, ClientStats] = {}

# Gestionnaire de connexions WebSocket
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Client {client_id} connecté via WebSocket")

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"Client {client_id} déconnecté")

    async def send_message(self, client_id: str, message: dict):
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_json(message)
            except Exception as e:
                logger.error(f"Erreur envoi message WebSocket {client_id}: {e}")
                self.disconnect(client_id)

manager = ConnectionManager()


# Utilitaires
def get_current_timestamp() -> str:
    """Retourne le timestamp actuel"""
    return datetime.now().isoformat()


def generate_request_id() -> str:
    """Génère un ID unique pour la requête"""
    return str(uuid.uuid4())


async def validate_client_data(data: List[TrainingDataItem]) -> DataValidationResult:
    """Valide les données d'entraînement du client"""
    warnings = []
    errors = []
    recommendations = []
    
    # Statistiques de base
    total_samples = len(data)
    language_dist = {}
    sentiment_dist = {}
    text_lengths = []
    
    for item in data:
        # Distribution des langues
        lang = item.language or "unknown"
        language_dist[lang] = language_dist.get(lang, 0) + 1
        
        # Distribution des sentiments (convertir les clés en strings)
        sentiment_str = str(item.sentiment)
        sentiment_dist[sentiment_str] = sentiment_dist.get(sentiment_str, 0) + 1
        
        # Longueurs des textes
        text_lengths.append(len(item.text))
    
    avg_text_length = sum(text_lengths) / len(text_lengths)
    
    # Validations
    if total_samples < 100:
        warnings.append(f"Peu de données ({total_samples}). Recommandé: >500 échantillons")
    
    if len(sentiment_dist) < 3:
        warnings.append("Peu de classes de sentiment. Diversifiez vos données.")
    
    # Vérifier l'équilibre des classes
    min_count = min(sentiment_dist.values())
    max_count = max(sentiment_dist.values())
    if max_count / min_count > 5:
        warnings.append("Classes déséquilibrées détectées")
        recommendations.append("Équilibrez vos données ou utilisez des techniques de rééchantillonnage")
    
    # Vérifier la longueur des textes
    if avg_text_length < 10:
        warnings.append("Textes très courts détectés")
    elif avg_text_length > 500:
        warnings.append("Textes très longs détectés")
        recommendations.append("Considérez tronquer les textes longs")
    
    # Languages multiples
    if len(language_dist) > 1:
        recommendations.append("Données multilingues détectées - assurez-vous d'avoir assez d'exemples par langue")
    
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
    """Vérification de l'état de santé de l'API"""
    return HealthCheck(
        status="healthy",
        version=settings.VERSION,
        uptime=time.time(),  # Simplification pour l'exemple
        models_loaded=len(active_models),
        system_info={
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "pytorch_version": torch.__version__
        }
    )


@router.post("/clients/{client_id}/validate-data", response_model=DataValidationResult)
async def validate_training_data(client_id: str, data: List[TrainingDataItem]):
    """Valide les données d'entraînement avant le lancement"""
    try:
        result = await validate_client_data(data)
        logger.info(f"Validation données client {client_id}: {result.total_samples} échantillons")
        return result
    except Exception as e:
        logger.error(f"Erreur validation données {client_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur de validation: {str(e)}")


@router.post("/clients/{client_id}/train", response_model=dict)
async def start_training(
    client_id: str, 
    request: TrainingRequest, 
    background_tasks: BackgroundTasks
):
    """Lance l'entraînement d'un modèle pour un client"""
    
    # Vérifier si un entraînement est déjà en cours
    if client_id in training_status and training_status[client_id].status == "training":
        raise HTTPException(
            status_code=409, 
            detail="Un entraînement est déjà en cours pour ce client"
        )
    
    try:
        # Valider les données
        validation_result = await validate_client_data(request.data)
        if not validation_result.is_valid:
            raise HTTPException(
                status_code=400,
                detail={"validation_errors": validation_result.errors}
            )
        
        # Créer la configuration d'entraînement
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
        
        # Initialiser le statut
        training_status[client_id] = TrainingStatus(
            client_id=client_id,
            status="initializing",
            progress=0.0,
            message="Initialisation de l'entraînement..."
        )
        
        # Lancer l'entraînement en arrière-plan
        background_tasks.add_task(
            run_training_background,
            client_id,
            training_config,
            [item.dict() for item in request.data]
        )
        
        logger.info(f"Entraînement lancé pour client {client_id}")
        
        return {
            "message": "Entraînement lancé avec succès",
            "client_id": client_id,
            "status": "training",
            "estimated_time": request.config.epochs * 2
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lancement entraînement {client_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")


async def run_training_background(client_id: str, config: TrainingConfig, data: List[Dict]):
    """Exécute l'entraînement en arrière-plan avec suivi en temps réel"""
    try:
        # Mettre à jour le statut
        training_status[client_id].status = "training"
        training_status[client_id].message = "Entraînement en cours..."
        
        # Créer le trainer
        trainer = ModelTrainer(config)
        
        # Lancer l'entraînement
        result = trainer.train(data)
        
        # Sauvegarder le modèle actif
        active_models[client_id] = trainer
        
        # Mettre à jour le statut final
        training_status[client_id].status = "completed"
        training_status[client_id].progress = 1.0
        training_status[client_id].message = "Entraînement terminé avec succès"
        
        # Notifier via WebSocket
        await manager.send_message(client_id, {
            "type": "training_completed",
            "data": result
        })
        
        logger.info(f"Entraînement terminé pour {client_id}")
        
    except Exception as e:
        logger.error(f"Erreur entraînement background {client_id}: {str(e)}")
        training_status[client_id].status = "error"
        training_status[client_id].message = f"Erreur: {str(e)}"
        
        await manager.send_message(client_id, {
            "type": "training_error",
            "data": {"error": str(e)}
        })


@router.get("/clients/{client_id}/training-status", response_model=TrainingStatus)
async def get_training_status(client_id: str):
    """Récupère le statut d'entraînement d'un client"""
    if client_id not in training_status:
        raise HTTPException(status_code=404, detail="Aucun entraînement trouvé pour ce client")
    
    return training_status[client_id]


@router.post("/clients/{client_id}/predict", response_model=PredictionResult)
async def predict_sentiment(client_id: str, request: PredictionRequest):
    """Prédit le sentiment d'un texte"""
    
    # Vérifier si le modèle existe
    if client_id not in active_models:
        raise HTTPException(status_code=404, detail="Modèle non trouvé. Entraînez d'abord un modèle.")
    
    try:
        start_time = time.time()
        trainer = active_models[client_id]
        
        # Préparer le texte
        if request.language:
            detected_language = request.language
        else:
            detected_language = trainer.processor.detect_language(request.text)
        
        # Prédiction
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
        
        # Préparer la réponse
        result = PredictionResult(
            text=request.text,
            sentiment=predicted_class - 2,  # Reconvertir [0,4] vers [-2,+2]
            confidence=confidence,
            language_detected=detected_language,
            processing_time=processing_time
        )
        
        # Ajouter les probabilités si demandées
        if request.return_probabilities:
            prob_dict = {}
            for i, prob in enumerate(probabilities[0]):
                sentiment_level = i - 2  # Reconvertir [0,4] vers [-2,+2]
                prob_dict[str(sentiment_level)] = prob.item()
            result.probabilities = prob_dict
        
        # Mettre à jour les statistiques client
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
        logger.error(f"Erreur prédiction {client_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction: {str(e)}")


@router.post("/clients/{client_id}/batch-predict", response_model=BatchPredictionResult)
async def batch_predict_sentiment(client_id: str, request: BatchPredictionRequest):
    """Prédit le sentiment de plusieurs textes en une fois"""
    
    if client_id not in active_models:
        raise HTTPException(status_code=404, detail="Modèle non trouvé")
    
    try:
        start_time = time.time()
        trainer = active_models[client_id]
        results = []
        
        for i, text in enumerate(request.texts):
            # Langue spécifiée ou détection automatique
            if request.languages and i < len(request.languages):
                language = request.languages[i]
            else:
                language = trainer.processor.detect_language(text)
            
            # Prédiction individuelle
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
        logger.error(f"Erreur batch prédiction {client_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/clients/{client_id}/model-info", response_model=ModelInfo)
async def get_model_info(client_id: str):
    """Récupère les informations sur le modèle d'un client"""
    
    if client_id not in active_models:
        raise HTTPException(status_code=404, detail="Modèle non trouvé")
    
    trainer = active_models[client_id]
    
    return ModelInfo(
        client_id=client_id,
        architecture=trainer.config.architecture,
        sentiment_levels=trainer.config.sentiment_levels,
        languages=trainer.config.languages or ["auto-detect"],
        vocab_size=trainer.processor.vocab_size if trainer.processor else 0,
        training_date=get_current_timestamp(),  # À améliorer avec la vraie date
        accuracy=trainer.metrics.best_val_acc if hasattr(trainer, 'metrics') else 0.0,
        status="active"
    )


@router.get("/clients/{client_id}/stats", response_model=ClientStats)
async def get_client_stats(client_id: str):
    """Récupère les statistiques d'un client"""
    
    if client_id not in client_stats:
        raise HTTPException(status_code=404, detail="Statistiques non trouvées")
    
    return client_stats[client_id]


@router.delete("/clients/{client_id}/model")
async def delete_client_model(client_id: str):
    """Supprime le modèle d'un client"""
    
    if client_id not in active_models:
        raise HTTPException(status_code=404, detail="Modèle non trouvé")
    
    try:
        # Supprimer de la mémoire
        del active_models[client_id]
        
        # Supprimer les fichiers
        model_dir = Path(settings.MODELS_PATH) / client_id
        if model_dir.exists():
            import shutil
            shutil.rmtree(model_dir)
        
        # Nettoyer les statuts
        if client_id in training_status:
            del training_status[client_id]
        
        logger.info(f"Modèle {client_id} supprimé")
        
        return {"message": f"Modèle {client_id} supprimé avec succès"}
        
    except Exception as e:
        logger.error(f"Erreur suppression modèle {client_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket pour le suivi en temps réel de l'entraînement"""
    await manager.connect(websocket, client_id)
    try:
        while True:
            # Maintenir la connexion vivante
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(client_id)