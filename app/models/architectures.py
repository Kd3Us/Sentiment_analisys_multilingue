"""
Architectures de modèles neuronal pour l'analyse de sentiment.
Ce module contient différentes architectures (LSTM, CNN, Transformer)
que le client peut choisir selon ses besoins.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import math


class SentimentLSTM(nn.Module):
    """Architecture LSTM bidirectionnelle avec attention"""
    
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, 
                 num_classes: int, num_layers: int = 2, dropout: float = 0.3):
        super(SentimentLSTM, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        
        # Couche d'embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # LSTM bidirectionnel
        self.lstm = nn.LSTM(
            embed_dim, 
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Mécanisme d'attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Couches de classification
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Initialisation des poids
        self._init_weights()
    
    def _init_weights(self):
        """Initialise les poids du modèle"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """
        Forward pass
        
        Args:
            input_ids: Tensor de forme (batch_size, seq_len)
            attention_mask: Masque d'attention (optionnel)
            
        Returns:
            Logits de classification
        """
        batch_size, seq_len = input_ids.size()
        
        # Embedding
        embedded = self.embedding(input_ids)  # (batch_size, seq_len, embed_dim)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)  # (batch_size, seq_len, hidden_dim*2)
        
        # Self-attention
        if attention_mask is not None:
            # Convertir le masque pour l'attention
            attention_mask = attention_mask.bool()
            attended_out, _ = self.attention(lstm_out, lstm_out, lstm_out, key_padding_mask=~attention_mask)
        else:
            attended_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global average pooling
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (attended_out * mask).sum(dim=1) / mask.sum(dim=1)
        else:
            pooled = attended_out.mean(dim=1)
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits


class SentimentCNN(nn.Module):
    """Architecture CNN avec plusieurs tailles de filtres"""
    
    def __init__(self, vocab_size: int, embed_dim: int, num_classes: int,
                 filter_sizes: list = [3, 4, 5], num_filters: int = 100, dropout: float = 0.5):
        super(SentimentCNN, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        
        # Couche d'embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Couches convolutionnelles
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=filter_size)
            for filter_size in filter_sizes
        ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Couche de classification
        self.classifier = nn.Linear(len(filter_sizes) * num_filters, num_classes)
        
        # Initialisation des poids
        self._init_weights()
    
    def _init_weights(self):
        """Initialise les poids du modèle"""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """Forward pass"""
        # Embedding
        embedded = self.embedding(input_ids)  # (batch_size, seq_len, embed_dim)
        embedded = embedded.transpose(1, 2)  # (batch_size, embed_dim, seq_len)
        
        # Convolutions + ReLU + Max pooling
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(embedded))  # (batch_size, num_filters, conv_seq_len)
            pooled = F.max_pool1d(conv_out, conv_out.size(2))  # (batch_size, num_filters, 1)
            conv_outputs.append(pooled.squeeze(2))  # (batch_size, num_filters)
        
        # Concaténation des sorties
        concat_output = torch.cat(conv_outputs, dim=1)  # (batch_size, len(filter_sizes) * num_filters)
        
        # Dropout et classification
        output = self.dropout(concat_output)
        logits = self.classifier(output)
        
        return logits


class PositionalEncoding(nn.Module):
    """Encodage positionnel pour Transformer"""
    
    def __init__(self, embed_dim: int, max_seq_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                           (-math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class SentimentTransformer(nn.Module):
    """Architecture Transformer pour l'analyse de sentiment"""
    
    def __init__(self, vocab_size: int, embed_dim: int, num_classes: int,
                 num_heads: int = 8, num_layers: int = 6, 
                 dim_feedforward: int = 2048, dropout: float = 0.1):
        super(SentimentTransformer, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        
        # Embedding et encodage positionnel
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoding = PositionalEncoding(embed_dim)
        
        # Couches Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )
        
        # Initialisation
        self._init_weights()
    
    def _init_weights(self):
        """Initialise les poids du modèle"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.1)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """Forward pass"""
        # Embedding
        embedded = self.embedding(input_ids) * math.sqrt(self.embed_dim)
        embedded = self.pos_encoding(embedded.transpose(0, 1)).transpose(0, 1)
        
        # Créer le masque d'attention pour le padding
        if attention_mask is not None:
            # Inverser le masque pour PyTorch (True = ignore)
            src_key_padding_mask = ~attention_mask.bool()
        else:
            src_key_padding_mask = None
        
        # Transformer
        transformer_out = self.transformer(
            embedded, 
            src_key_padding_mask=src_key_padding_mask
        )
        
        # Global average pooling avec masque
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (transformer_out * mask).sum(dim=1) / mask.sum(dim=1)
        else:
            pooled = transformer_out.mean(dim=1)
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits


class HybridModel(nn.Module):
    """Architecture hybride combinant CNN et LSTM"""
    
    def __init__(self, vocab_size: int, embed_dim: int, num_classes: int,
                 hidden_dim: int = 256, dropout: float = 0.3):
        super(HybridModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Branche CNN
        self.conv1 = nn.Conv1d(embed_dim, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        
        # Branche LSTM
        self.lstm = nn.LSTM(embed_dim, hidden_dim//2, bidirectional=True, batch_first=True)
        
        # Fusion et classification
        self.fusion = nn.Linear(128 + hidden_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        # Embedding
        embedded = self.embedding(input_ids)
        
        # Branche CNN
        cnn_input = embedded.transpose(1, 2)  # (batch, embed_dim, seq_len)
        cnn_out = F.relu(self.conv1(cnn_input))
        cnn_out = F.relu(self.conv2(cnn_out))
        cnn_pooled = F.max_pool1d(cnn_out, cnn_out.size(2)).squeeze(2)
        
        # Branche LSTM
        lstm_out, _ = self.lstm(embedded)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            lstm_pooled = (lstm_out * mask).sum(dim=1) / mask.sum(dim=1)
        else:
            lstm_pooled = lstm_out.mean(dim=1)
        
        # Fusion
        combined = torch.cat([cnn_pooled, lstm_pooled], dim=1)
        fused = self.fusion(combined)
        
        # Classification
        logits = self.classifier(fused)
        return logits


def get_model_architecture(architecture: str, vocab_size: int, embed_dim: int, 
                          num_classes: int, **kwargs) -> nn.Module:
    """
    Factory function pour créer une architecture de modèle
    
    Args:
        architecture: Type d'architecture ('lstm', 'cnn', 'transformer', 'hybrid')
        vocab_size: Taille du vocabulaire
        embed_dim: Dimension des embeddings
        num_classes: Nombre de classes de sentiment
        **kwargs: Arguments supplémentaires spécifiques à l'architecture
        
    Returns:
        Instance du modèle
    """
    
    if architecture.lower() == 'lstm':
        return SentimentLSTM(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=kwargs.get('hidden_dim', 256),
            num_classes=num_classes,
            num_layers=kwargs.get('num_layers', 2),
            dropout=kwargs.get('dropout', 0.3)
        )
    
    elif architecture.lower() == 'cnn':
        return SentimentCNN(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_classes=num_classes,
            filter_sizes=kwargs.get('filter_sizes', [3, 4, 5]),
            num_filters=kwargs.get('num_filters', 100),
            dropout=kwargs.get('dropout', 0.5)
        )
    
    elif architecture.lower() == 'transformer':
        return SentimentTransformer(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_classes=num_classes,
            num_heads=kwargs.get('num_heads', 8),
            num_layers=kwargs.get('num_layers', 6),
            dim_feedforward=kwargs.get('dim_feedforward', 2048),
            dropout=kwargs.get('dropout', 0.1)
        )
    
    elif architecture.lower() == 'hybrid':
        return HybridModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_classes=num_classes,
            hidden_dim=kwargs.get('hidden_dim', 256),
            dropout=kwargs.get('dropout', 0.3)
        )
    
    else:
        raise ValueError(f"Architecture non supportée: {architecture}")


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Compte le nombre de paramètres dans un modèle
    
    Args:
        model: Modèle PyTorch
        
    Returns:
        Dictionnaire avec le nombre de paramètres
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params
    }