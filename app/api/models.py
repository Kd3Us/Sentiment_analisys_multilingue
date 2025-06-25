"""
Modèles Pydantic pour l'API.
Ces modèles définissent les schémas de données pour les requêtes et réponses.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from enum import Enum


class ArchitectureType(str, Enum):
    """Types d'architectures supportées"""
    LSTM = "lstm"
    CNN = "cnn"
    TRANSFORMER = "transformer"
    HYBRID = "hybrid"


class SentimentLevel(int, Enum):
    """Niveaux de sentiment prédéfinis"""
    VERY_NEGATIVE = -2
    NEGATIVE = -1
    NEUTRAL = 0
    POSITIVE = 1
    VERY_POSITIVE = 2


class TrainingDataItem(BaseModel):
    """Item de donnée d'entraînement"""
    text: str = Field(..., description="Texte à analyser", min_length=1, max_length=1000)
    sentiment: int = Field(..., description="Niveau de sentiment", ge=-10, le=10)
    language: Optional[str] = Field(None, description="Code langue (ex: 'fr', 'en')")
    domain: Optional[str] = Field(None, description="Domaine métier")
    confidence: Optional[float] = Field(None, description="Confiance de l'annotation", ge=0.0, le=1.0)
    
    @validator('text')
    def validate_text(cls, v):
        if not v or v.isspace():
            raise ValueError('Le texte ne peut pas être vide')
        return v.strip()


class ModelConfiguration(BaseModel):
    """Configuration d'un modèle client"""
    architecture: ArchitectureType = Field(ArchitectureType.LSTM, description="Type d'architecture")
    sentiment_levels: int = Field(5, description="Nombre de niveaux de sentiment", ge=2, le=10)
    languages: List[str] = Field(default=["fr", "en"], description="Langues supportées")
    domain: Optional[str] = Field(None, description="Domaine d'application")
    
    # Hyperparamètres
    batch_size: int = Field(32, description="Taille du batch", ge=1, le=256)
    learning_rate: float = Field(0.001, description="Taux d'apprentissage", gt=0.0, le=1.0)
    epochs: int = Field(50, description="Nombre d'epochs", ge=1, le=1000)
    validation_split: float = Field(0.2, description="Proportion validation", gt=0.0, lt=1.0)
    early_stopping_patience: int = Field(10, description="Patience early stopping", ge=1)
    
    # Architecture spécifique
    embed_dim: int = Field(300, description="Dimension embeddings", ge=50, le=1000)
    hidden_dim: int = Field(256, description="Dimension cachée", ge=50, le=2048)
    dropout: float = Field(0.3, description="Taux de dropout", ge=0.0, le=0.9)


class TrainingRequest(BaseModel):
    """Requête d'entraînement"""
    data: List[TrainingDataItem] = Field(..., description="Données d'entraînement", min_items=10)
    config: ModelConfiguration = Field(..., description="Configuration du modèle")
    
    @validator('data')
    def validate_data_distribution(cls, v):
        # Vérifier qu'il y a assez de données par classe
        sentiment_counts = {}
        for item in v:
            sentiment_counts[item.sentiment] = sentiment_counts.get(item.sentiment, 0) + 1
        
        if len(sentiment_counts) < 2:
            raise ValueError('Il faut au moins 2 classes de sentiment différentes')
        
        # Vérifier qu'il y a au moins 5 exemples par classe
        for sentiment, count in sentiment_counts.items():
            if count < 5:
                raise ValueError(f'Il faut au moins 5 exemples pour le sentiment {sentiment}')
        
        return v


class PredictionRequest(BaseModel):
    """Requête de prédiction"""
    text: str = Field(..., description="Texte à analyser", min_length=1, max_length=1000)
    language: Optional[str] = Field(None, description="Langue du texte (auto-détection si None)")
    return_probabilities: bool = Field(False, description="Retourner les probabilités")
    
    @validator('text')
    def validate_text(cls, v):
        if not v or v.isspace():
            raise ValueError('Le texte ne peut pas être vide')
        return v.strip()


class BatchPredictionRequest(BaseModel):
    """Requête de prédiction en lot"""
    texts: List[str] = Field(..., description="Textes à analyser", min_items=1, max_items=1000)
    languages: Optional[List[str]] = Field(None, description="Langues des textes")
    return_probabilities: bool = Field(False, description="Retourner les probabilités")
    
    @validator('texts')
    def validate_texts(cls, v):
        cleaned_texts = []
        for text in v:
            if not text or text.isspace():
                raise ValueError('Tous les textes doivent être non vides')
            cleaned_texts.append(text.strip())
        return cleaned_texts
    
    @validator('languages')
    def validate_languages_length(cls, v, values):
        if v is not None and 'texts' in values:
            if len(v) != len(values['texts']):
                raise ValueError('Le nombre de langues doit correspondre au nombre de textes')
        return v


class PredictionResult(BaseModel):
    """Résultat de prédiction"""
    text: str = Field(..., description="Texte analysé")
    sentiment: int = Field(..., description="Sentiment prédit")
    confidence: float = Field(..., description="Confiance de la prédiction")
    language_detected: str = Field(..., description="Langue détectée")
    probabilities: Optional[Dict[str, float]] = Field(None, description="Probabilités par classe")
    processing_time: float = Field(..., description="Temps de traitement en secondes")


class BatchPredictionResult(BaseModel):
    """Résultat de prédiction en lot"""
    results: List[PredictionResult] = Field(..., description="Résultats individuels")
    total_processing_time: float = Field(..., description="Temps total de traitement")
    average_confidence: float = Field(..., description="Confiance moyenne")


class TrainingStatus(BaseModel):
    """Statut d'entraînement"""
    client_id: str = Field(..., description="ID du client")
    status: str = Field(..., description="Statut actuel")
    progress: float = Field(..., description="Progression (0.0 à 1.0)")
    current_epoch: Optional[int] = Field(None, description="Epoch actuelle")
    total_epochs: Optional[int] = Field(None, description="Total d'epochs")
    train_loss: Optional[float] = Field(None, description="Perte d'entraînement")
    val_loss: Optional[float] = Field(None, description="Perte de validation")
    val_accuracy: Optional[float] = Field(None, description="Précision de validation")
    estimated_time_remaining: Optional[int] = Field(None, description="Temps restant estimé (secondes)")
    message: Optional[str] = Field(None, description="Message informatif")


class TrainingResult(BaseModel):
    """Résultat d'entraînement"""
    client_id: str = Field(..., description="ID du client")
    success: bool = Field(..., description="Succès de l'entraînement")
    training_time: float = Field(..., description="Temps d'entraînement total")
    total_epochs: int = Field(..., description="Nombre d'epochs exécutées")
    vocab_size: int = Field(..., description="Taille du vocabulaire")
    model_parameters: Dict[str, int] = Field(..., description="Nombre de paramètres")
    best_val_accuracy: float = Field(..., description="Meilleure précision de validation")
    best_val_loss: float = Field(..., description="Meilleure perte de validation")
    best_epoch: int = Field(..., description="Meilleure epoch")
    model_path: str = Field(..., description="Chemin du modèle sauvegardé")
    vocab_info: Dict[str, Any] = Field(..., description="Informations sur le vocabulaire")
    language_distribution: Dict[str, int] = Field(..., description="Distribution des langues")
    error_message: Optional[str] = Field(None, description="Message d'erreur si échec")


class ModelInfo(BaseModel):
    """Informations sur un modèle client"""
    client_id: str = Field(..., description="ID du client")
    model_name: Optional[str] = Field(None, description="Nom du modèle")
    architecture: str = Field(..., description="Architecture utilisée")
    sentiment_levels: int = Field(..., description="Nombre de niveaux de sentiment")
    languages: List[str] = Field(..., description="Langues supportées")
    vocab_size: int = Field(..., description="Taille du vocabulaire")
    training_date: str = Field(..., description="Date d'entraînement")
    accuracy: float = Field(..., description="Précision du modèle")
    status: str = Field(..., description="Statut du modèle (active, training, error)")
    version: str = Field("1.0", description="Version du modèle")


class DataValidationResult(BaseModel):
    """Résultat de validation des données"""
    is_valid: bool = Field(..., description="Données valides")
    total_samples: int = Field(..., description="Nombre total d'échantillons")
    language_distribution: Dict[str, int] = Field(..., description="Distribution des langues")
    sentiment_distribution: Dict[str, int] = Field(..., description="Distribution des sentiments")
    avg_text_length: float = Field(..., description="Longueur moyenne des textes")
    warnings: List[str] = Field(default=[], description="Avertissements")
    errors: List[str] = Field(default=[], description="Erreurs détectées")
    recommendations: List[str] = Field(default=[], description="Recommandations")


class ErrorResponse(BaseModel):
    """Réponse d'erreur standardisée"""
    error: str = Field(..., description="Type d'erreur")
    message: str = Field(..., description="Message d'erreur")
    details: Optional[Dict[str, Any]] = Field(None, description="Détails supplémentaires")
    timestamp: str = Field(..., description="Timestamp de l'erreur")
    request_id: Optional[str] = Field(None, description="ID de la requête")


class HealthCheck(BaseModel):
    """Statut de santé de l'API"""
    status: str = Field("healthy", description="Statut général")
    version: str = Field(..., description="Version de l'API")
    uptime: float = Field(..., description="Temps de fonctionnement en secondes")
    models_loaded: int = Field(..., description="Nombre de modèles chargés")
    system_info: Dict[str, Any] = Field(..., description="Informations système")


class ClientStats(BaseModel):
    """Statistiques d'un client"""
    client_id: str = Field(..., description="ID du client")
    models_count: int = Field(..., description="Nombre de modèles")
    total_predictions: int = Field(..., description="Prédictions totales")
    avg_accuracy: float = Field(..., description="Précision moyenne")
    languages_used: List[str] = Field(..., description="Langues utilisées")
    last_activity: str = Field(..., description="Dernière activité")
    storage_used: float = Field(..., description="Stockage utilisé (MB)")


# Modèles pour les WebSockets
class WSMessage(BaseModel):
    """Message WebSocket générique"""
    type: str = Field(..., description="Type de message")
    client_id: str = Field(..., description="ID du client")
    data: Dict[str, Any] = Field(..., description="Données du message")
    timestamp: str = Field(..., description="Timestamp")


class TrainingProgress(BaseModel):
    """Progression d'entraînement pour WebSocket"""
    epoch: int = Field(..., description="Epoch actuelle")
    total_epochs: int = Field(..., description="Total epochs")
    train_loss: float = Field(..., description="Perte d'entraînement")
    val_loss: float = Field(..., description="Perte de validation")
    val_accuracy: float = Field(..., description="Précision validation")
    learning_rate: float = Field(..., description="Taux d'apprentissage actuel")
    time_elapsed: float = Field(..., description="Temps écoulé")
    eta: Optional[float] = Field(None, description="Temps restant estimé")


# Modèles de configuration avancée
class AdvancedModelConfig(BaseModel):
    """Configuration avancée pour utilisateurs experts"""
    # Optimiseur personnalisé
    optimizer_type: str = Field("adamw", description="Type d'optimiseur")
    weight_decay: float = Field(1e-5, description="Decay des poids")
    beta1: float = Field(0.9, description="Beta1 pour Adam")
    beta2: float = Field(0.999, description="Beta2 pour Adam")
    
    # Scheduler
    scheduler_type: str = Field("plateau", description="Type de scheduler")
    scheduler_factor: float = Field(0.7, description="Facteur de réduction LR")
    scheduler_patience: int = Field(5, description="Patience du scheduler")
    
    # Augmentation de données
    data_augmentation: bool = Field(False, description="Activer l'augmentation")
    augmentation_prob: float = Field(0.1, description="Probabilité d'augmentation")
    
    # Régularisation
    gradient_clip_norm: float = Field(1.0, description="Clipping des gradients")
    label_smoothing: float = Field(0.0, description="Lissage des labels")
    
    # Architecture spécifique
    lstm_layers: Optional[int] = Field(None, description="Nombre de couches LSTM")
    attention_heads: Optional[int] = Field(None, description="Têtes d'attention")
    transformer_layers: Optional[int] = Field(None, description="Couches Transformer")


class ExportRequest(BaseModel):
    """Requête d'export de modèle"""
    format: str = Field("pytorch", description="Format d'export (pytorch, onnx, tflite)")
    include_processor: bool = Field(True, description="Inclure le processeur")
    optimize: bool = Field(False, description="Optimiser pour la production")
    quantize: bool = Field(False, description="Quantification du modèle")


class ExportResult(BaseModel):
    """Résultat d'export"""
    success: bool = Field(..., description="Succès de l'export")
    file_path: str = Field(..., description="Chemin du fichier exporté")
    file_size: int = Field(..., description="Taille du fichier en bytes")
    format: str = Field(..., description="Format d'export")
    metadata: Dict[str, Any] = Field(..., description="Métadonnées du modèle")


# Validation personnalisée pour certains champs
def validate_sentiment_range(sentiment_levels: int, sentiment_value: int) -> bool:
    """Valide qu'une valeur de sentiment est dans la bonne plage"""
    if sentiment_levels == 5:
        return sentiment_value in [-2, -1, 0, 1, 2]
    elif sentiment_levels == 3:
        return sentiment_value in [-1, 0, 1]
    else:
        # Pour un nombre personnalisé de niveaux
        half_range = sentiment_levels // 2
        if sentiment_levels % 2 == 1:  # Impair, avec classe neutre
            return -half_range <= sentiment_value <= half_range
        else:  # Pair, sans classe neutre explicite
            return -(sentiment_levels//2) <= sentiment_value < (sentiment_levels//2)


# Configuration des exemples pour la documentation API
class Config:
    schema_extra = {
        "example": {
            "text": "Ce produit est absolument fantastique !",
            "sentiment": 2,
            "language": "fr",
            "confidence": 0.95
        }
    }