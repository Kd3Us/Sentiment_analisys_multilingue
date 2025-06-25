# Sentiment AI Platform 🚀

Plateforme d'analyse de sentiment multilingue avec modèles neuronal personnalisables from scratch.

## 📋 Aperçu du projet

Cette plateforme permet à chaque client de créer et entraîner son propre modèle d'analyse de sentiment selon ses besoins spécifiques :

- **🧠 Modèles from scratch** : LSTM, CNN, Transformer, Hybride
- **🌍 Support multilingue** : Détection automatique + preprocessing adapté
- **📊 5 niveaux de sentiment** : Très négatif, Négatif, Neutre, Positif, Très positif
- **🎯 Personnalisation totale** : Chaque client entraîne avec ses propres données
- **⚡ Temps réel** : Suivi de l'entraînement via WebSocket
- **🔄 API REST complète** : Interface simple et puissante

## 🏗️ Architecture du projet

```
sentiment-ai-platform/
├── app/
│   ├── __init__.py
│   ├── config/
│   │   └── settings.py          # Configuration centralisée
│   ├── core/
│   │   └── data_processor.py    # Preprocessing multilingue
│   ├── models/
│   │   └── architectures.py     # Modèles neuraux (LSTM, CNN, etc.)
│   ├── services/
│   │   └── trainer.py          # Service d'entraînement
│   └── api/
│       ├── models.py           # Modèles Pydantic
│       └── routes.py           # Routes API
├── models/                     # Modèles sauvegardés par client
├── data/                      # Données temporaires
├── logs/                      # Logs de l'application
├── requirements.txt           # Dépendances Python
├── main.py                   # Point d'entrée principal
├── .env.example             # Configuration d'environnement
└── README.md               # Ce fichier
```

## 🚀 Installation et démarrage

### 1. Prérequis

```bash
# Python 3.8+
python --version

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows
```

### 2. Installation des dépendances

```bash
pip install -r requirements.txt

# Télécharger les modèles spaCy (optionnel mais recommandé)
python -m spacy download en_core_web_sm
python -m spacy download fr_core_news_sm
python -m spacy download es_core_news_sm
```

### 3. Configuration

```bash
# Copier et adapter la configuration
cp .env.example .env

# Créer les dossiers nécessaires
mkdir -p models data logs
```

### 4. Lancement

```bash
# Développement
python main.py

# ou avec uvicorn directement
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

L'API sera accessible sur `http://localhost:8000`

## 📖 Utilisation

### Workflow typique

1. **Validation des données** : Vérifier la qualité des données d'entraînement
2. **Configuration du modèle** : Choisir l'architecture et les hyperparamètres
3. **Lancement de l'entraînement** : Entraîner le modèle personnalisé
4. **Suivi temps réel** : Monitorer la progression via WebSocket
5. **Prédiction** : Utiliser le modèle entraîné pour de nouvelles prédictions

### Exemples d'utilisation

#### 1. Valider des données d'entraînement

```python
import requests

# Données d'exemple
data = [
    {"text": "Ce produit est fantastique!", "sentiment": 2, "language": "fr"},
    {"text": "Service client décevant", "sentiment": -1, "language": "fr"},
    {"text": "Produit correct, sans plus", "sentiment": 0, "language": "fr"}
]

response = requests.post(
    "http://localhost:8000/api/v1/clients/client123/validate-data",
    json=data
)
print(response.json())
```

#### 2. Lancer un entraînement

```python
import requests

# Configuration complète
training_request = {
    "data": [
        {"text": "Excellent produit, je recommande!", "sentiment": 2, "language": "fr"},
        {"text": "Très déçu de mon achat", "sentiment": -2, "language": "fr"},
        # ... plus de données (minimum 100 recommandé)
    ],
    "config": {
        "architecture": "lstm",
        "sentiment_levels": 5,
        "languages": ["fr", "en"],
        "batch_size": 32,
        "learning_rate": 0.001,
        "epochs": 50,
        "embed_dim": 300,
        "hidden_dim": 256
    }
}

response = requests.post(
    "http://localhost:8000/api/v1/clients/client123/train",
    json=training_request
)
print(response.json())
```

#### 3. Suivre l'entraînement en temps réel

```python
import websocket
import json

def on_message(ws, message):
    data = json.loads(message)
    if data['type'] == 'training_progress':
        progress = data['data']
        print(f"Epoch {progress['current_epoch']}/{progress['total_epochs']} - "
              f"Accuracy: {progress['val_accuracy']:.3f}")

ws = websocket.WebSocketApp(
    "ws://localhost:8000/api/v1/ws/client123",
    on_message=on_message
)
ws.run_forever()
```

#### 4. Faire des prédictions

```python
import requests

# Prédiction simple
response = requests.post(
    "http://localhost:8000/api/v1/clients/client123/predict",
    json={
        "text": "Ce service est absolument parfait!",
        "return_probabilities": True
    }
)

result = response.json()
print(f"Sentiment: {result['sentiment']}")
print(f"Confiance: {result['confidence']:.3f}")
print(f"Langue: {result['language_detected']}")
```

## 🔧 Configuration avancée

### Architectures disponibles

1. **LSTM** : Réseaux récurrents bidirectionnels avec mécanisme d'attention
2. **CNN** : Convolutions 1D avec plusieurs tailles de filtres
3. **Transformer** : Architecture d'attention pure avec encodage positionnel
4. **Hybrid** : Combinaison CNN + LSTM pour capturer différents patterns

### Paramètres d'entraînement

```python
config = {
    "architecture": "lstm",           # lstm, cnn, transformer, hybrid
    "sentiment_levels": 5,            # 2-10 niveaux
    "languages": ["fr", "en", "es"],  # Langues supportées
    "batch_size": 32,                 # Taille des batches
    "learning_rate": 0.001,           # Taux d'apprentissage
    "epochs": 50,                     # Nombre d'epochs
    "validation_split": 0.2,          # Proportion validation
    "early_stopping_patience": 10,    # Patience early stopping
    "embed_dim": 300,                 # Dimension embeddings
    "hidden_dim": 256,                # Dimension cachée
    "dropout": 0.3                    # Taux de dropout
}
```

## 📊 Endpoints API

### Entraînement
- `POST /api/v1/clients/{client_id}/validate-data` - Valider les données
- `POST /api/v1/clients/{client_id}/train` - Lancer l'entraînement
- `GET /api/v1/clients/{client_id}/training-status` - Statut d'entraînement

### Prédiction
- `POST /api/v1/clients/{client_id}/predict` - Prédiction simple
- `POST /api/v1/clients/{client_id}/batch-predict` - Prédiction en lot

### Gestion
- `GET /api/v1/clients/{client_id}/model-info` - Infos sur le modèle
- `GET /api/v1/clients/{client_id}/stats` - Statistiques client
- `DELETE /api/v1/clients/{client_id}/model` - Supprimer le modèle

### Système
- `GET /api/v1/health` - Santé de l'API
- `GET /info` - Informations détaillées

### WebSocket
- `WS /api/v1/ws/{client_id}` - Suivi temps réel

## 🔍 Monitoring et logs

Les logs sont automatiquement générés dans le dossier `logs/` :
- Niveau INFO affiché dans la console
- Niveau DEBUG sauvegardé dans les fichiers
- Rotation automatique des logs

## 🧪 Tests et validation

### Format des données d'entraînement

```json
{
  "text": "Votre texte à analyser",
  "sentiment": 2,
  "language": "fr",
  "confidence": 0.95
}
```

### Niveaux de sentiment recommandés

- **5 niveaux** : -2 (très négatif), -1 (négatif), 0 (neutre), 1 (positif), 2 (très positif)
- **3 niveaux** : -1 (négatif), 0 (neutre), 1 (positif)

## 🚀 Déploiement en production

### Docker (recommandé)

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Variables d'environnement importantes

```bash
# Production
DEBUG=False
SECRET_KEY=your-production-secret-key

# Base de données
DATABASE_URL=postgresql://user:password@db:5432/sentiment_db

# Limites
MAX_UPLOAD_SIZE_MB=500
MAX_TRAINING_TIME_HOURS=48
```

## 🤝 Contribution

Ce projet est conçu pour être facilement extensible :

1. **Nouvelles architectures** : Ajouter dans `app/models/architectures.py`
2. **Nouvelles langues** : Étendre `MultilingualProcessor`
3. **Nouvelles métriques** : Modifier `TrainingMetrics`
4. **Nouveaux endpoints** : Ajouter dans `app/api/routes.py`

## 📞 Support

Pour toute question ou problème :
1. Vérifiez les logs dans `logs/app.log`
2. Consultez la documentation API sur `/docs`
3. Testez avec des données d'exemple simples

## 🎯 Prochaines étapes

Votre projet est maintenant prêt ! Vous pouvez :

1. **Tester** avec des données simples
2. **Personnaliser** les architectures selon vos besoins
3. **Intégrer** avec le site web de votre entreprise
4. **Optimiser** les performances selon l'usage

Bonne chance avec votre stage ! 🚀