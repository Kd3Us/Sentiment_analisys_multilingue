"""
Processeur de données multilingue.
Ce module gère le preprocessing des données textuelles pour l'entraînement
et la prédiction, indépendamment de la langue.
"""

import re
import string
from typing import List, Dict, Tuple, Optional, Set
from collections import Counter
import pandas as pd
import numpy as np
from langdetect import detect, DetectorFactory
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy

# Assurer la reproductibilité de langdetect
DetectorFactory.seed = 0


class MultilingualProcessor:
    """Processeur de données multilingue pour l'analyse de sentiment"""
    
    def __init__(self, max_vocab_size: int = 50000, min_word_freq: int = 2):
        self.max_vocab_size = max_vocab_size
        self.min_word_freq = min_word_freq
        self.vocab_to_idx = {}
        self.idx_to_vocab = {}
        self.vocab_size = 0
        self.language_stats = {}
        
        # Tokens spéciaux
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<START>': 2,
            '<END>': 3
        }
        
        # Système universel - pas de dépendances spécifiques
        print("Processeur multilingue universel initialisé")
        print("Support: Toutes les langues via tokenisation Unicode")
    
    def _download_nltk_data(self):
        """Télécharge les données NLTK nécessaires"""
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
        except Exception as e:
            print(f"Erreur lors du téléchargement NLTK: {e}")
    
    def detect_language(self, text: str) -> str:
        """Détecte la langue d'un texte - Version universelle"""
        try:
            # Essayer langdetect d'abord (pour les langues connues)
            detected = detect(text)
            return detected
        except:
            # Fallback : détection heuristique universelle
            return self._universal_language_detection(text)
    
    def _universal_language_detection(self, text: str) -> str:
        """Détection de langue basée sur les patterns Unicode"""
        # Analyser les plages Unicode pour identifier les scripts
        scripts = {}
        for char in text:
            if char.isalpha():
                # Détecter le script Unicode
                if '\u0000' <= char <= '\u007F':  # Latin basique
                    scripts['latin'] = scripts.get('latin', 0) + 1
                elif '\u0080' <= char <= '\u00FF':  # Latin étendu
                    scripts['latin_ext'] = scripts.get('latin_ext', 0) + 1
                elif '\u0100' <= char <= '\u017F':  # Latin étendu A
                    scripts['latin_ext_a'] = scripts.get('latin_ext_a', 0) + 1
                elif '\u1200' <= char <= '\u137F':  # Éthiopien (Amharique)
                    scripts['ethiopic'] = scripts.get('ethiopic', 0) + 1
                elif '\u0600' <= char <= '\u06FF':  # Arabe
                    scripts['arabic'] = scripts.get('arabic', 0) + 1
                elif '\u4E00' <= char <= '\u9FFF':  # Chinois
                    scripts['chinese'] = scripts.get('chinese', 0) + 1
                else:
                    scripts['other'] = scripts.get('other', 0) + 1
        
        # Retourner le script dominant ou 'unknown'
        if scripts:
            dominant_script = max(scripts, key=scripts.get)
            return f"script_{dominant_script}"
        return 'unknown'
    
    def clean_text(self, text: str, language: str = None) -> str:
        """
        Nettoie le texte - Version universelle pour toutes les langues
        
        Args:
            text: Texte à nettoyer
            language: Code langue (optionnel)
            
        Returns:
            Texte nettoyé
        """
        if not isinstance(text, str):
            return ""
        
        import unicodedata
        
        # Normalisation Unicode (important pour les langues avec accents/diacritiques)
        text = unicodedata.normalize('NFD', text)
        
        # Conversion en minuscules (fonctionne pour la plupart des scripts)
        text = text.lower()
        
        # Suppression des URLs (universel)
        text = re.sub(r'http\S+|www\.\S+', '', text)
        
        # Suppression des mentions et hashtags (universel)
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Suppression des emails (universel)
        text = re.sub(r'\S+@\S+', '', text)
        
        # Suppression des chiffres (optionnel, universel)
        text = re.sub(r'\d+', '', text)
        
        # Nettoyage de la ponctuation avec support Unicode
        # Garder les apostrophes et tirets (importants dans beaucoup de langues)
        text = re.sub(r'[^\w\s\'-]', ' ', text, flags=re.UNICODE)
        
        # Suppression des espaces multiples
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def tokenize_text(self, text: str, language: str = None) -> List[str]:
        """
        Tokenise le texte - Version universelle pour toutes les langues
        
        Args:
            text: Texte à tokeniser
            language: Code langue (peut être ignoré pour l'universalité)
            
        Returns:
            Liste de tokens
        """
        try:
            # Tokenisation universelle basée sur Unicode
            tokens = self._universal_tokenize(text)
            
            # Filtrage intelligent
            filtered_tokens = []
            for token in tokens:
                # Garder les tokens avec au moins 1 caractère alphabétique
                if any(c.isalpha() for c in token) and len(token) > 0:
                    filtered_tokens.append(token)
            
            return filtered_tokens
            
        except Exception as e:
            print(f"Erreur tokenisation universelle: {e}")
            # Fallback simple mais robuste
            return self._simple_tokenize(text)
    
    def _universal_tokenize(self, text: str) -> List[str]:
        """Tokenisation universelle basée sur les propriétés Unicode"""
        import unicodedata
        
        # Normaliser le texte (décomposer les accents, etc.)
        normalized = unicodedata.normalize('NFD', text.lower())
        
        # Tokenisation basée sur les espaces et la ponctuation Unicode
        tokens = []
        current_token = ""
        
        for char in normalized:
            if char.isalnum() or char in "'-":  # Lettres, chiffres, apostrophes, tirets
                current_token += char
            else:
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
        
        # Ajouter le dernier token
        if current_token:
            tokens.append(current_token)
        
        return tokens
    
    def _simple_tokenize(self, text: str) -> List[str]:
        """Tokenisation de secours très simple"""
        import re
        # Expression régulière Unicode pour capturer tous les caractères alphabétiques
        tokens = re.findall(r'\b[\w\u00C0-\u017F\u0100-\u024F\u1E00-\u1EFF]+\b', text.lower())
        return [token for token in tokens if len(token) > 1]
    
    def _load_spacy_model(self, language: str) -> bool:
        """Charge le modèle spaCy pour une langue"""
        if language in self.spacy_models:
            return True
            
        # Mapping des codes langue vers les modèles spaCy
        spacy_models = {
            'en': 'en_core_web_sm',
            'fr': 'fr_core_news_sm',
            'es': 'es_core_news_sm',
            'de': 'de_core_news_sm',
            'it': 'it_core_news_sm'
        }
        
        if language not in spacy_models:
            return False
            
        try:
            self.spacy_models[language] = spacy.load(spacy_models[language])
            return True
        except OSError:
            print(f"Modèle spaCy {spacy_models[language]} non trouvé")
            return False
    
    def build_vocabulary(self, texts: List[str], languages: List[str] = None) -> Dict:
        """
        Construit le vocabulaire à partir des textes
        
        Args:
            texts: Liste des textes
            languages: Liste des langues correspondantes
            
        Returns:
            Dictionnaire avec statistiques du vocabulaire
        """
        print("Construction du vocabulaire...")
        
        if languages is None:
            languages = [self.detect_language(text) for text in texts]
        
        # Compteur de mots global
        word_counter = Counter()
        
        # Statistiques par langue
        self.language_stats = {}
        
        for text, lang in zip(texts, languages):
            if lang not in self.language_stats:
                self.language_stats[lang] = {'count': 0, 'words': Counter()}
            
            # Nettoyer et tokeniser
            cleaned_text = self.clean_text(text, lang)
            tokens = self.tokenize_text(cleaned_text, lang)
            
            # Mettre à jour les compteurs
            word_counter.update(tokens)
            self.language_stats[lang]['count'] += 1
            self.language_stats[lang]['words'].update(tokens)
        
        # Créer le vocabulaire avec les tokens spéciaux
        self.vocab_to_idx = self.special_tokens.copy()
        self.idx_to_vocab = {v: k for k, v in self.special_tokens.items()}
        
        # Ajouter les mots les plus fréquents
        most_common_words = word_counter.most_common(self.max_vocab_size - len(self.special_tokens))
        
        idx = len(self.special_tokens)
        for word, freq in most_common_words:
            if freq >= self.min_word_freq:
                self.vocab_to_idx[word] = idx
                self.idx_to_vocab[idx] = word
                idx += 1
        
        self.vocab_size = len(self.vocab_to_idx)
        
        return {
            'vocab_size': self.vocab_size,
            'total_words': sum(word_counter.values()),
            'unique_words': len(word_counter),
            'languages': list(self.language_stats.keys()),
            'language_distribution': {lang: stats['count'] for lang, stats in self.language_stats.items()}
        }
    
    def text_to_indices(self, text: str, language: str = None, max_length: int = 512) -> List[int]:
        """
        Convertit un texte en indices du vocabulaire
        
        Args:
            text: Texte à convertir
            language: Langue du texte
            max_length: Longueur maximale de la séquence
            
        Returns:
            Liste d'indices
        """
        if language is None:
            language = self.detect_language(text)
        
        cleaned_text = self.clean_text(text, language)
        tokens = self.tokenize_text(cleaned_text, language)
        
        # Convertir en indices
        indices = [self.special_tokens['<START>']]
        for token in tokens[:max_length-2]:  # -2 pour START et END
            indices.append(self.vocab_to_idx.get(token, self.special_tokens['<UNK>']))
        indices.append(self.special_tokens['<END>'])
        
        # Padding si nécessaire
        while len(indices) < max_length:
            indices.append(self.special_tokens['<PAD>'])
        
        return indices[:max_length]
    
    def process_dataset(self, data: List[Dict], validation_split: float = 0.2) -> Tuple[Dict, Dict]:
        """
        Traite un dataset complet pour l'entraînement
        
        Args:
            data: Liste de dictionnaires {text, sentiment, language}
            validation_split: Proportion pour la validation
            
        Returns:
            Tuple (train_data, val_data)
        """
        print(f"Traitement de {len(data)} échantillons...")
        
        # Extraction des textes et construction du vocabulaire
        texts = [item['text'] for item in data]
        languages = [item.get('language') for item in data]
        
        # Détecter les langues manquantes
        for i, lang in enumerate(languages):
            if lang is None:
                languages[i] = self.detect_language(texts[i])
        
        # Construire le vocabulaire
        vocab_stats = self.build_vocabulary(texts, languages)
        
        # Convertir les textes en indices
        processed_data = []
        for item in data:
            indices = self.text_to_indices(item['text'], item.get('language'))
            processed_data.append({
                'input_ids': indices,
                'sentiment': item['sentiment'],
                'language': item.get('language', self.detect_language(item['text']))
            })
        
        # Split train/validation
        train_data, val_data = train_test_split(
            processed_data, 
            test_size=validation_split, 
            random_state=42,
            stratify=[item['sentiment'] for item in processed_data]
        )
        
        return {
            'data': train_data,
            'vocab_stats': vocab_stats,
            'processor': self
        }, {
            'data': val_data,
            'vocab_stats': vocab_stats,
            'processor': self
        }
    
    def get_vocab_info(self) -> Dict:
        """Retourne des informations sur le vocabulaire"""
        return {
            'vocab_size': self.vocab_size,
            'special_tokens': self.special_tokens,
            'language_stats': self.language_stats,
            'sample_words': list(self.vocab_to_idx.keys())[:20]
        }