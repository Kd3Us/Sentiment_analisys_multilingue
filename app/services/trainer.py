"""
Service d'entraînement des modèles pour chaque client.
Ce module gère l'entraînement personnalisé des modèles de sentiment.
"""

import os
import json
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import numpy as np
from loguru import logger

from ..models.architectures import get_model_architecture, count_parameters
from ..core.data_processor import MultilingualProcessor
from ..config.settings import settings, model_config


@dataclass
class TrainingConfig:
    """Configuration pour l'entraînement"""
    client_id: str
    architecture: str = "lstm"
    sentiment_levels: int = 5
    languages: List[str] = None
    
    # Hyperparamètres
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 50
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    
    # Modèle
    embed_dim: int = 300
    hidden_dim: int = 256
    dropout: float = 0.3
    
    # Sauvegardes
    save_best_only: bool = True
    save_frequency: int = 5  # Sauvegarder tous les X epochs


class SentimentDataset(Dataset):
    """Dataset PyTorch pour l'analyse de sentiment"""
    
    def __init__(self, data: List[Dict], max_length: int = 512):
        self.data = data
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        input_ids = torch.tensor(item['input_ids'][:self.max_length], dtype=torch.long)
        
        # Convertir sentiment de [-2, +2] vers [0, 4] pour PyTorch
        sentiment_normalized = item['sentiment'] + 2  # -2->0, -1->1, 0->2, 1->3, 2->4
        sentiment = torch.tensor(sentiment_normalized, dtype=torch.long)
        
        # Créer le masque d'attention (1 pour les tokens réels, 0 pour le padding)
        attention_mask = (input_ids != 0).float()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'sentiment': sentiment,
            'language': item.get('language', 'unknown')
        }


class TrainingMetrics:
    """Classe pour suivre les métriques d'entraînement"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.val_f1_scores = []
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.epochs_without_improvement = 0
    
    def update(self, epoch: int, train_loss: float, val_loss: float, 
               train_acc: float, val_acc: float, val_f1: float):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accuracies.append(train_acc)
        self.val_accuracies.append(val_acc)
        self.val_f1_scores.append(val_f1)
        
        # Vérifier si c'est le meilleur modèle
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_val_acc = val_acc
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
            return True  # Nouveau meilleur modèle
        else:
            self.epochs_without_improvement += 1
            return False
    
    def should_stop_early(self, patience: int) -> bool:
        return self.epochs_without_improvement >= patience
    
    def get_summary(self) -> Dict:
        return {
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_val_loss,
            'best_val_accuracy': self.best_val_acc,
            'final_train_loss': self.train_losses[-1] if self.train_losses else 0,
            'final_val_loss': self.val_losses[-1] if self.val_losses else 0,
            'final_train_accuracy': self.train_accuracies[-1] if self.train_accuracies else 0,
            'final_val_accuracy': self.val_accuracies[-1] if self.val_accuracies else 0,
            'final_val_f1': self.val_f1_scores[-1] if self.val_f1_scores else 0
        }


class ModelTrainer:
    """Classe principale pour l'entraînement des modèles clients"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.processor = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.metrics = TrainingMetrics()
        
        # Créer les dossiers de sauvegarde
        self.model_dir = Path(settings.MODELS_PATH) / config.client_id
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Trainer initialisé pour client {config.client_id} sur {self.device}")
    
    def prepare_data(self, raw_data: List[Dict]) -> Tuple[DataLoader, DataLoader]:
        """
        Prépare les données pour l'entraînement
        
        Args:
            raw_data: Données brutes du client
            
        Returns:
            Tuple (train_loader, val_loader)
        """
        logger.info(f"Préparation des données: {len(raw_data)} échantillons")
        
        # Initialiser le processeur
        self.processor = MultilingualProcessor()
        
        # Traiter les données
        train_data, val_data = self.processor.process_dataset(
            raw_data, 
            validation_split=self.config.validation_split
        )
        
        # Créer les datasets
        train_dataset = SentimentDataset(train_data['data'])
        val_dataset = SentimentDataset(val_data['data'])
        
        # Créer les dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        logger.info(f"Données préparées: {len(train_dataset)} train, {len(val_dataset)} val")
        return train_loader, val_loader
    
    def build_model(self, vocab_size: int) -> nn.Module:
        """
        Construit le modèle selon la configuration
        
        Args:
            vocab_size: Taille du vocabulaire
            
        Returns:
            Modèle PyTorch
        """
        logger.info(f"Construction du modèle {self.config.architecture}")
        
        model = get_model_architecture(
            architecture=self.config.architecture,
            vocab_size=vocab_size,
            embed_dim=self.config.embed_dim,
            num_classes=self.config.sentiment_levels,
            hidden_dim=self.config.hidden_dim,
            dropout=self.config.dropout
        )
        
        # Compter les paramètres
        param_count = count_parameters(model)
        logger.info(f"Modèle créé avec {param_count['trainable_parameters']:,} paramètres entraînables")
        
        # Déplacer sur GPU si disponible
        model = model.to(self.device)
        
        return model
    
    def setup_training(self, model: nn.Module):
        """Configure l'optimiseur, le scheduler et la fonction de perte"""
        
        # Fonction de perte
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimiseur
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=model_config.OPTIMIZER_CONFIG['weight_decay']
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.7,
            patience=5,
            verbose=True
        )
        
        logger.info("Configuration d'entraînement terminée")
    
    def train_epoch(self, model: nn.Module, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Entraîne le modèle pour une epoch
        
        Returns:
            Tuple (loss, accuracy)
        """
        model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['sentiment'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Statistiques
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_predictions
        
        return avg_loss, accuracy
    
    def validate_epoch(self, model: nn.Module, val_loader: DataLoader) -> Tuple[float, float, float]:
        """
        Valide le modèle
        
        Returns:
            Tuple (loss, accuracy, f1_score)
        """
        model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['sentiment'].to(self.device)
                
                outputs = model(input_ids, attention_mask)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        return avg_loss, accuracy, f1
    
    def save_model(self, model: nn.Module, epoch: int, is_best: bool = False):
        """Sauvegarde le modèle"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.__dict__,
            'vocab_size': self.processor.vocab_size,
            'processor_state': {
                'vocab_to_idx': self.processor.vocab_to_idx,
                'idx_to_vocab': self.processor.idx_to_vocab,
                'special_tokens': self.processor.special_tokens
            },
            'metrics': self.metrics.get_summary()
        }
        
        # Sauvegarde standard
        checkpoint_path = self.model_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Sauvegarde du meilleur modèle
        if is_best:
            best_path = self.model_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Nouveau meilleur modèle sauvegardé: {best_path}")
        
        logger.debug(f"Checkpoint sauvegardé: {checkpoint_path}")
    
    def train(self, raw_data: List[Dict]) -> Dict:
        """
        Lance l'entraînement complet
        
        Args:
            raw_data: Données d'entraînement du client
            
        Returns:
            Dictionnaire avec les résultats d'entraînement
        """
        start_time = time.time()
        logger.info(f"Début de l'entraînement pour {self.config.client_id}")
        
        try:
            # Préparer les données
            train_loader, val_loader = self.prepare_data(raw_data)
            
            # Construire le modèle
            self.model = self.build_model(self.processor.vocab_size)
            
            # Configurer l'entraînement
            self.setup_training(self.model)
            
            # Boucle d'entraînement
            for epoch in range(self.config.epochs):
                epoch_start = time.time()
                
                # Entraînement
                train_loss, train_acc = self.train_epoch(self.model, train_loader)
                
                # Validation
                val_loss, val_acc, val_f1 = self.validate_epoch(self.model, val_loader)
                
                # Mettre à jour les métriques
                is_best = self.metrics.update(epoch, train_loss, val_loss, train_acc, val_acc, val_f1)
                
                # Scheduler
                self.scheduler.step(val_loss)
                
                # Sauvegarde
                if is_best or (epoch + 1) % self.config.save_frequency == 0:
                    self.save_model(self.model, epoch, is_best)
                
                # Logging
                epoch_time = time.time() - epoch_start
                logger.info(
                    f"Epoch {epoch+1}/{self.config.epochs} - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f} - "
                    f"Time: {epoch_time:.2f}s"
                )
                
                # Early stopping
                if self.metrics.should_stop_early(self.config.early_stopping_patience):
                    logger.info(f"Early stopping à l'epoch {epoch+1}")
                    break
            
            # Sauvegarde finale
            self.save_model(self.model, epoch, False)
            
            # Résultats
            total_time = time.time() - start_time
            results = {
                'client_id': self.config.client_id,
                'training_time': total_time,
                'total_epochs': epoch + 1,
                'vocab_size': self.processor.vocab_size,
                'model_parameters': count_parameters(self.model),
                'metrics': self.metrics.get_summary(),
                'model_path': str(self.model_dir / "best_model.pt")
            }
            
            logger.info(f"Entraînement terminé en {total_time:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"Erreur pendant l'entraînement: {str(e)}")
            raise
    
    def load_model(self, model_path: str) -> nn.Module:
        """Charge un modèle sauvegardé"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Reconstruire le modèle
        vocab_size = checkpoint['vocab_size']
        model = self.build_model(vocab_size)
        
        # Charger les poids
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Restaurer le processeur
        self.processor = MultilingualProcessor()
        self.processor.vocab_to_idx = checkpoint['processor_state']['vocab_to_idx']
        self.processor.idx_to_vocab = checkpoint['processor_state']['idx_to_vocab']
        self.processor.special_tokens = checkpoint['processor_state']['special_tokens']
        self.processor.vocab_size = vocab_size
        
        logger.info(f"Modèle chargé depuis {model_path}")
        return model