"""
Multilingual data processor.
This module handles text data preprocessing for training and prediction,
independent of language.
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

# Ensure langdetect reproducibility
DetectorFactory.seed = 0


class MultilingualProcessor:
    """Multilingual data processor for sentiment analysis"""
    
    def __init__(self, max_vocab_size: int = 50000, min_word_freq: int = 2):
        self.max_vocab_size = max_vocab_size
        self.min_word_freq = min_word_freq
        self.vocab_to_idx = {}
        self.idx_to_vocab = {}
        self.vocab_size = 0
        self.language_stats = {}
        
        # Special tokens
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<START>': 2,
            '<END>': 3
        }
        
        # Universal system - no specific dependencies
        print("Universal multilingual processor initialized")
        print("Support: All languages via Unicode tokenization")
    
    def _download_nltk_data(self):
        """Downloads required NLTK data"""
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
        except Exception as e:
            print(f"Error downloading NLTK data: {e}")
    
    def detect_language(self, text: str) -> str:
        """Detects text language - Universal version"""
        try:
            # Try langdetect first (for known languages)
            detected = detect(text)
            return detected
        except:
            # Fallback: universal heuristic detection
            return self._universal_language_detection(text)
    
    def _universal_language_detection(self, text: str) -> str:
        """Language detection based on Unicode patterns"""
        # Analyze Unicode ranges to identify scripts
        scripts = {}
        for char in text:
            if char.isalpha():
                # Detect Unicode script
                if '\u0000' <= char <= '\u007F':  # Basic Latin
                    scripts['latin'] = scripts.get('latin', 0) + 1
                elif '\u0080' <= char <= '\u00FF':  # Extended Latin
                    scripts['latin_ext'] = scripts.get('latin_ext', 0) + 1
                elif '\u0100' <= char <= '\u017F':  # Extended Latin A
                    scripts['latin_ext_a'] = scripts.get('latin_ext_a', 0) + 1
                elif '\u1200' <= char <= '\u137F':  # Ethiopic (Amharic)
                    scripts['ethiopic'] = scripts.get('ethiopic', 0) + 1
                elif '\u0600' <= char <= '\u06FF':  # Arabic
                    scripts['arabic'] = scripts.get('arabic', 0) + 1
                elif '\u4E00' <= char <= '\u9FFF':  # Chinese
                    scripts['chinese'] = scripts.get('chinese', 0) + 1
                else:
                    scripts['other'] = scripts.get('other', 0) + 1
        
        # Return dominant script or 'unknown'
        if scripts:
            dominant_script = max(scripts, key=scripts.get)
            return f"script_{dominant_script}"
        return 'unknown'
    
    def clean_text(self, text: str, language: str = None) -> str:
        """
        Cleans text - Universal version for all languages
        
        Args:
            text: Text to clean
            language: Language code (optional)
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        import unicodedata
        
        # Unicode normalization (important for languages with accents/diacritics)
        text = unicodedata.normalize('NFD', text)
        
        # Convert to lowercase (works for most scripts)
        text = text.lower()
        
        # Remove URLs (universal)
        text = re.sub(r'http\S+|www\.\S+', '', text)
        
        # Remove mentions and hashtags (universal)
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove emails (universal)
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove numbers (optional, universal)
        text = re.sub(r'\d+', '', text)
        
        # Clean punctuation with Unicode support
        # Keep apostrophes and hyphens (important in many languages)
        text = re.sub(r'[^\w\s\'-]', ' ', text, flags=re.UNICODE)
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def tokenize_text(self, text: str, language: str = None) -> List[str]:
        """
        Tokenizes text - Universal version for all languages
        
        Args:
            text: Text to tokenize
            language: Language code (can be ignored for universality)
            
        Returns:
            List of tokens
        """
        try:
            # Universal tokenization based on Unicode
            tokens = self._universal_tokenize(text)
            
            # Smart filtering
            filtered_tokens = []
            for token in tokens:
                # Keep tokens with at least 1 alphabetic character
                if any(c.isalpha() for c in token) and len(token) > 0:
                    filtered_tokens.append(token)
            
            return filtered_tokens
            
        except Exception as e:
            print(f"Universal tokenization error: {e}")
            # Robust but simple fallback
            return self._simple_tokenize(text)
    
    def _universal_tokenize(self, text: str) -> List[str]:
        """Universal tokenization based on Unicode properties"""
        import unicodedata
        
        # Normalize text (decompose accents, etc.)
        normalized = unicodedata.normalize('NFD', text.lower())
        
        # Tokenization based on spaces and Unicode punctuation
        tokens = []
        current_token = ""
        
        for char in normalized:
            if char.isalnum() or char in "'-":  # Letters, numbers, apostrophes, hyphens
                current_token += char
            else:
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
        
        # Add last token
        if current_token:
            tokens.append(current_token)
        
        return tokens
    
    def _simple_tokenize(self, text: str) -> List[str]:
        """Very simple fallback tokenization"""
        import re
        # Unicode regex to capture all alphabetic characters
        tokens = re.findall(r'\b[\w\u00C0-\u017F\u0100-\u024F\u1E00-\u1EFF]+\b', text.lower())
        return [token for token in tokens if len(token) > 1]
    
    def build_vocabulary(self, texts: List[str], languages: List[str] = None) -> Dict:
        """
        Builds vocabulary from texts
        
        Args:
            texts: List of texts
            languages: List of corresponding languages
            
        Returns:
            Dictionary with vocabulary statistics
        """
        print("Building vocabulary...")
        
        if languages is None:
            languages = [self.detect_language(text) for text in texts]
        
        # Global word counter
        word_counter = Counter()
        
        # Statistics per language
        self.language_stats = {}
        
        for text, lang in zip(texts, languages):
            if lang not in self.language_stats:
                self.language_stats[lang] = {'count': 0, 'words': Counter()}
            
            # Clean and tokenize
            cleaned_text = self.clean_text(text, lang)
            tokens = self.tokenize_text(cleaned_text, lang)
            
            # Update counters
            word_counter.update(tokens)
            self.language_stats[lang]['count'] += 1
            self.language_stats[lang]['words'].update(tokens)
        
        # Create vocabulary with special tokens
        self.vocab_to_idx = self.special_tokens.copy()
        self.idx_to_vocab = {v: k for k, v in self.special_tokens.items()}
        
        # Add most frequent words
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
        Converts text to vocabulary indices
        
        Args:
            text: Text to convert
            language: Text language
            max_length: Maximum sequence length
            
        Returns:
            List of indices
        """
        if language is None:
            language = self.detect_language(text)
        
        cleaned_text = self.clean_text(text, language)
        tokens = self.tokenize_text(cleaned_text, language)
        
        # Convert to indices
        indices = [self.special_tokens['<START>']]
        for token in tokens[:max_length-2]:  # -2 for START and END
            indices.append(self.vocab_to_idx.get(token, self.special_tokens['<UNK>']))
        indices.append(self.special_tokens['<END>'])
        
        # Padding if necessary
        while len(indices) < max_length:
            indices.append(self.special_tokens['<PAD>'])
        
        return indices[:max_length]
    
    def process_dataset(self, data: List[Dict], validation_split: float = 0.2) -> Tuple[Dict, Dict]:
        """
        Processes complete dataset for training
        
        Args:
            data: List of dictionaries {text, sentiment, language}
            validation_split: Proportion for validation
            
        Returns:
            Tuple (train_data, val_data)
        """
        print(f"Processing {len(data)} samples...")
        
        # Extract texts and build vocabulary
        texts = [item['text'] for item in data]
        languages = [item.get('language') for item in data]
        
        # Detect missing languages
        for i, lang in enumerate(languages):
            if lang is None:
                languages[i] = self.detect_language(texts[i])
        
        # Build vocabulary
        vocab_stats = self.build_vocabulary(texts, languages)
        
        # Convert texts to indices
        processed_data = []
        for item in data:
            indices = self.text_to_indices(item['text'], item.get('language'))
            processed_data.append({
                'input_ids': indices,
                'sentiment': item['sentiment'],
                'language': item.get('language', self.detect_language(item['text']))
            })
        
        # Train/validation split
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
        """Returns vocabulary information"""
        return {
            'vocab_size': self.vocab_size,
            'special_tokens': self.special_tokens,
            'language_stats': self.language_stats,
            'sample_words': list(self.vocab_to_idx.keys())[:20]
        }