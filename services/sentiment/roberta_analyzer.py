"""
RoBERTa-based sentiment analysis for QIE Risk Oracle.

This module implements sentiment analysis using the cardiffnlp/twitter-roberta-base-sentiment model
with batch processing and output normalization to 0.0-1.0 range.
"""

import logging
import asyncio
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
import numpy as np
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    pipeline,
    Pipeline
)

from config import get_config
from .telegram_client import TelegramMessage


@dataclass
class SentimentResult:
    """Represents the result of sentiment analysis."""
    text: str
    sentiment_score: float  # Normalized to 0.0-1.0 (0.0 = very negative, 1.0 = very positive)
    confidence: float
    raw_scores: Dict[str, float]  # Raw model outputs
    processing_time_ms: float


class RoBERTaSentimentAnalyzer:
    """RoBERTa-based sentiment analyzer for financial text."""
    
    def __init__(self, model_name: Optional[str] = None, batch_size: int = 32):
        """Initialize the sentiment analyzer.
        
        Args:
            model_name: HuggingFace model name, defaults to config value
            batch_size: Batch size for processing multiple texts
        """
        config = get_config()
        self.model_name = model_name or config.model.sentiment_model_name
        self.batch_size = batch_size
        self.logger = logging.getLogger(__name__)
        
        # Model components
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForSequenceClassification] = None
        self.pipeline: Optional[Pipeline] = None
        
        # Device configuration
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Using device: {self.device}")
        
        # Model loaded flag
        self._model_loaded = False
    
    async def load_model(self) -> None:
        """Load the RoBERTa model and tokenizer."""
        try:
            self.logger.info(f"Loading sentiment model: {self.model_name}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            # Move model to appropriate device
            self.model.to(self.device)
            
            # Create pipeline for easier inference
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                return_all_scores=True
            )
            
            self._model_loaded = True
            self.logger.info("Sentiment model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load sentiment model: {e}")
            raise
    
    def _ensure_model_loaded(self) -> None:
        """Ensure the model is loaded before processing."""
        if not self._model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
    
    def _normalize_sentiment_score(self, raw_scores: List[Dict[str, Union[str, float]]]) -> Tuple[float, float, Dict[str, float]]:
        """Normalize sentiment scores to 0.0-1.0 range.
        
        The cardiffnlp/twitter-roberta-base-sentiment model outputs:
        - LABEL_0: Negative
        - LABEL_1: Neutral  
        - LABEL_2: Positive
        
        We normalize to 0.0-1.0 where:
        - 0.0 = Very negative (high risk)
        - 0.5 = Neutral
        - 1.0 = Very positive (low risk)
        
        Args:
            raw_scores: Raw model output scores
            
        Returns:
            Tuple of (normalized_score, confidence, raw_scores_dict)
        """
        # Convert to dictionary for easier access
        scores_dict = {item['label']: item['score'] for item in raw_scores}
        
        # Get individual scores
        negative_score = scores_dict.get('LABEL_0', 0.0)
        neutral_score = scores_dict.get('LABEL_1', 0.0)
        positive_score = scores_dict.get('LABEL_2', 0.0)
        
        # Calculate normalized sentiment score
        # Formula: (positive - negative + 1) / 2
        # This maps [-1, 1] to [0, 1]
        normalized_score = (positive_score - negative_score + 1.0) / 2.0
        
        # Confidence is the maximum probability
        confidence = max(negative_score, neutral_score, positive_score)
        
        # Create readable scores dictionary
        readable_scores = {
            'negative': negative_score,
            'neutral': neutral_score,
            'positive': positive_score
        }
        
        return normalized_score, confidence, readable_scores
    
    async def analyze_text(self, text: str) -> SentimentResult:
        """Analyze sentiment of a single text.
        
        Args:
            text: Text to analyze
            
        Returns:
            SentimentResult with normalized scores
        """
        self._ensure_model_loaded()
        
        if not text or not text.strip():
            return SentimentResult(
                text=text,
                sentiment_score=0.5,  # Neutral for empty text
                confidence=0.0,
                raw_scores={'negative': 0.0, 'neutral': 1.0, 'positive': 0.0},
                processing_time_ms=0.0
            )
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Run inference
            raw_results = self.pipeline(text.strip())
            
            # Normalize scores
            normalized_score, confidence, raw_scores = self._normalize_sentiment_score(raw_results)
            
            end_time = asyncio.get_event_loop().time()
            processing_time_ms = (end_time - start_time) * 1000
            
            return SentimentResult(
                text=text,
                sentiment_score=normalized_score,
                confidence=confidence,
                raw_scores=raw_scores,
                processing_time_ms=processing_time_ms
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing text sentiment: {e}")
            # Return neutral sentiment on error
            return SentimentResult(
                text=text,
                sentiment_score=0.5,
                confidence=0.0,
                raw_scores={'negative': 0.0, 'neutral': 1.0, 'positive': 0.0},
                processing_time_ms=0.0
            )
    
    async def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        """Analyze sentiment of multiple texts in batches.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of SentimentResult objects
        """
        self._ensure_model_loaded()
        
        if not texts:
            return []
        
        results = []
        start_time = asyncio.get_event_loop().time()
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            # Filter out empty texts but keep track of indices
            valid_texts = []
            valid_indices = []
            
            for j, text in enumerate(batch):
                if text and text.strip():
                    valid_texts.append(text.strip())
                    valid_indices.append(i + j)
            
            # Process valid texts
            if valid_texts:
                try:
                    raw_results = self.pipeline(valid_texts)
                    
                    # Process results
                    for k, raw_result in enumerate(raw_results):
                        original_index = valid_indices[k]
                        original_text = texts[original_index]
                        
                        normalized_score, confidence, raw_scores = self._normalize_sentiment_score(raw_result)
                        
                        result = SentimentResult(
                            text=original_text,
                            sentiment_score=normalized_score,
                            confidence=confidence,
                            raw_scores=raw_scores,
                            processing_time_ms=0.0  # Will be set after batch processing
                        )
                        
                        # Insert result at correct position
                        while len(results) <= original_index:
                            results.append(None)
                        results[original_index] = result
                        
                except Exception as e:
                    self.logger.error(f"Error processing batch: {e}")
                    # Add neutral results for failed batch
                    for k in valid_indices:
                        while len(results) <= k:
                            results.append(None)
                        results[k] = SentimentResult(
                            text=texts[k],
                            sentiment_score=0.5,
                            confidence=0.0,
                            raw_scores={'negative': 0.0, 'neutral': 1.0, 'positive': 0.0},
                            processing_time_ms=0.0
                        )
        
        # Fill in any missing results (empty texts)
        for i, text in enumerate(texts):
            if i >= len(results) or results[i] is None:
                results.insert(i, SentimentResult(
                    text=text,
                    sentiment_score=0.5,  # Neutral for empty text
                    confidence=0.0,
                    raw_scores={'negative': 0.0, 'neutral': 1.0, 'positive': 0.0},
                    processing_time_ms=0.0
                ))
        
        # Set processing time for all results
        end_time = asyncio.get_event_loop().time()
        total_processing_time_ms = (end_time - start_time) * 1000
        avg_processing_time_ms = total_processing_time_ms / len(texts) if texts else 0.0
        
        for result in results:
            if result:
                result.processing_time_ms = avg_processing_time_ms
        
        return results
    
    async def analyze_telegram_messages(self, messages: List[TelegramMessage]) -> List[TelegramMessage]:
        """Analyze sentiment of Telegram messages and update them with scores.
        
        Args:
            messages: List of TelegramMessage objects
            
        Returns:
            List of TelegramMessage objects with sentiment_score populated
        """
        if not messages:
            return []
        
        # Extract texts
        texts = [msg.text for msg in messages]
        
        # Analyze sentiment
        sentiment_results = await self.analyze_batch(texts)
        
        # Update messages with sentiment scores
        updated_messages = []
        for i, (message, sentiment_result) in enumerate(zip(messages, sentiment_results)):
            # Create a copy of the message with sentiment score
            updated_message = TelegramMessage(
                message_id=message.message_id,
                text=message.text,
                timestamp=message.timestamp,
                user_id=message.user_id,
                channel=message.channel,
                sentiment_score=sentiment_result.sentiment_score if sentiment_result else 0.5,
                is_anomaly=message.is_anomaly
            )
            updated_messages.append(updated_message)
        
        self.logger.info(f"Analyzed sentiment for {len(messages)} messages")
        return updated_messages
    
    def get_model_info(self) -> Dict[str, Union[str, bool]]:
        """Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_name': self.model_name,
            'device': self.device,
            'model_loaded': self._model_loaded,
            'batch_size': self.batch_size,
            'cuda_available': torch.cuda.is_available()
        }


# Factory function for easy instantiation
async def create_sentiment_analyzer(model_name: Optional[str] = None, batch_size: int = 32) -> RoBERTaSentimentAnalyzer:
    """Create and initialize a RoBERTa sentiment analyzer.
    
    Args:
        model_name: HuggingFace model name, defaults to config value
        batch_size: Batch size for processing
        
    Returns:
        Initialized RoBERTaSentimentAnalyzer
    """
    analyzer = RoBERTaSentimentAnalyzer(model_name=model_name, batch_size=batch_size)
    await analyzer.load_model()
    return analyzer