"""
QIE Risk Oracle Sentiment Analysis Package

This package provides comprehensive sentiment analysis capabilities for the QIE Risk Oracle,
including Telegram message processing, RoBERTa-based sentiment analysis, EWMA smoothing,
and anomaly detection.
"""

from .telegram_client import (
    TelegramMessageProcessor,
    TelegramMessage,
    TelegramConfig,
    create_telegram_processor
)

from .roberta_analyzer import (
    RoBERTaSentimentAnalyzer,
    SentimentResult,
    create_sentiment_analyzer
)

from .ewma_analyzer import (
    EWMAAnalyzer,
    EWMAState,
    AnomalyEvent,
    create_ewma_analyzer
)

from .sentiment_engine import (
    SentimentEngine,
    SentimentEngineResult,
    create_sentiment_engine
)

__all__ = [
    # Telegram components
    'TelegramMessageProcessor',
    'TelegramMessage',
    'TelegramConfig',
    'create_telegram_processor',
    
    # Sentiment analysis components
    'RoBERTaSentimentAnalyzer',
    'SentimentResult',
    'create_sentiment_analyzer',
    
    # EWMA and anomaly detection
    'EWMAAnalyzer',
    'EWMAState',
    'AnomalyEvent',
    'create_ewma_analyzer',
    
    # Main engine
    'SentimentEngine',
    'SentimentEngineResult',
    'create_sentiment_engine',
]
