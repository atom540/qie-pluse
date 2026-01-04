"""
Main sentiment analysis engine for QIE Risk Oracle.

This module combines Telegram message processing, RoBERTa sentiment analysis,
and EWMA smoothing with anomaly detection into a unified sentiment engine.
"""

import asyncio
import logging
from typing import List, Dict, Optional, Callable, Any
from dataclasses import dataclass
from datetime import datetime

from config import get_config
from .telegram_client import TelegramMessageProcessor, TelegramMessage, create_telegram_processor
from .roberta_analyzer import RoBERTaSentimentAnalyzer, SentimentResult, create_sentiment_analyzer
from .ewma_analyzer import EWMAAnalyzer, AnomalyEvent, create_ewma_analyzer


@dataclass
class SentimentEngineResult:
    """Complete result from sentiment engine processing."""
    message: TelegramMessage
    sentiment_result: SentimentResult
    smoothed_score: float
    anomaly: Optional[AnomalyEvent]
    processing_timestamp: datetime


class SentimentEngine:
    """Main sentiment analysis engine combining all components."""
    
    def __init__(self):
        """Initialize the sentiment engine."""
        self.logger = logging.getLogger(__name__)
        self.config = get_config()
        
        # Components
        self.telegram_processor: Optional[TelegramMessageProcessor] = None
        self.sentiment_analyzer: Optional[RoBERTaSentimentAnalyzer] = None
        self.ewma_analyzer: Optional[EWMAAnalyzer] = None
        
        # State
        self.is_running = False
        self.message_handlers: List[Callable[[SentimentEngineResult], None]] = []
        
        # Statistics
        self.stats = {
            'messages_processed': 0,
            'anomalies_detected': 0,
            'last_processing_time': None,
            'average_sentiment': 0.0,
            'current_ewma': 0.0
        }
    
    async def initialize(self) -> None:
        """Initialize all components of the sentiment engine."""
        try:
            self.logger.info("Initializing sentiment engine components...")
            
            # Initialize Telegram processor
            self.telegram_processor = create_telegram_processor()
            await self.telegram_processor.connect()
            
            # Initialize sentiment analyzer
            self.sentiment_analyzer = await create_sentiment_analyzer()
            
            # Initialize EWMA analyzer
            self.ewma_analyzer = create_ewma_analyzer()
            
            # Register message handler
            self.telegram_processor.add_message_handler(self._process_telegram_message)
            
            self.logger.info("Sentiment engine initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize sentiment engine: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown the sentiment engine and cleanup resources."""
        try:
            self.is_running = False
            
            if self.telegram_processor:
                await self.telegram_processor.disconnect()
            
            self.logger.info("Sentiment engine shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during sentiment engine shutdown: {e}")
    
    def add_result_handler(self, handler: Callable[[SentimentEngineResult], None]) -> None:
        """Add a handler for processed sentiment results.
        
        Args:
            handler: Function to call with each SentimentEngineResult
        """
        self.message_handlers.append(handler)
    
    async def _process_telegram_message(self, message: TelegramMessage) -> None:
        """Process a single Telegram message through the full pipeline.
        
        Args:
            message: TelegramMessage to process
        """
        try:
            # Analyze sentiment
            sentiment_result = await self.sentiment_analyzer.analyze_text(message.text)
            
            # Apply EWMA smoothing and anomaly detection
            smoothed_score, anomaly = self.ewma_analyzer.process_sentiment_score(
                sentiment_result.sentiment_score
            )
            
            # Update message with sentiment score
            message.sentiment_score = sentiment_result.sentiment_score
            if anomaly:
                message.is_anomaly = True
            
            # Create complete result
            result = SentimentEngineResult(
                message=message,
                sentiment_result=sentiment_result,
                smoothed_score=smoothed_score,
                anomaly=anomaly,
                processing_timestamp=datetime.now()
            )
            
            # Update statistics
            self._update_stats(result)
            
            # Call all result handlers
            for handler in self.message_handlers:
                try:
                    handler(result)
                except Exception as e:
                    self.logger.error(f"Error in result handler: {e}")
            
            # Log processing
            self.logger.debug(f"Processed message: sentiment={sentiment_result.sentiment_score:.3f}, "
                            f"smoothed={smoothed_score:.3f}, anomaly={anomaly is not None}")
            
        except Exception as e:
            self.logger.error(f"Error processing Telegram message: {e}")
    
    def _update_stats(self, result: SentimentEngineResult) -> None:
        """Update engine statistics.
        
        Args:
            result: Processing result to update stats with
        """
        self.stats['messages_processed'] += 1
        self.stats['last_processing_time'] = result.processing_timestamp
        
        if result.anomaly:
            self.stats['anomalies_detected'] += 1
        
        # Update running averages
        count = self.stats['messages_processed']
        old_avg = self.stats['average_sentiment']
        new_score = result.sentiment_result.sentiment_score
        self.stats['average_sentiment'] = old_avg + (new_score - old_avg) / count
        
        # Update current EWMA
        self.stats['current_ewma'] = result.smoothed_score
    
    async def start_monitoring(self, channels: Optional[List[str]] = None) -> None:
        """Start monitoring Telegram channels for sentiment analysis.
        
        Args:
            channels: List of channel usernames to monitor, defaults to config
        """
        if not self.telegram_processor:
            raise RuntimeError("Sentiment engine not initialized. Call initialize() first.")
        
        # Use configured channels if none provided
        if channels is None:
            from .telegram_client import TelegramConfig
            config = TelegramConfig()
            channels = config.monitored_channels
        
        self.is_running = True
        self.logger.info(f"Starting sentiment monitoring for channels: {channels}")
        
        try:
            await self.telegram_processor.listen_to_channels(channels)
        except Exception as e:
            self.logger.error(f"Error in sentiment monitoring: {e}")
            self.is_running = False
            raise
    
    async def analyze_historical_messages(self, channel: str, limit: int = 100) -> List[SentimentEngineResult]:
        """Analyze historical messages from a channel.
        
        Args:
            channel: Channel username to analyze
            limit: Maximum number of messages to analyze
            
        Returns:
            List of SentimentEngineResult objects
        """
        if not self.telegram_processor or not self.sentiment_analyzer or not self.ewma_analyzer:
            raise RuntimeError("Sentiment engine not initialized. Call initialize() first.")
        
        # Fetch historical messages
        messages = await self.telegram_processor.get_recent_messages(channel, limit)
        
        if not messages:
            return []
        
        # Analyze sentiment for all messages
        sentiment_results = await self.sentiment_analyzer.analyze_batch([msg.text for msg in messages])
        
        # Process through EWMA and create results
        results = []
        for message, sentiment_result in zip(messages, sentiment_results):
            smoothed_score, anomaly = self.ewma_analyzer.process_sentiment_score(
                sentiment_result.sentiment_score
            )
            
            # Update message
            message.sentiment_score = sentiment_result.sentiment_score
            if anomaly:
                message.is_anomaly = True
            
            result = SentimentEngineResult(
                message=message,
                sentiment_result=sentiment_result,
                smoothed_score=smoothed_score,
                anomaly=anomaly,
                processing_timestamp=datetime.now()
            )
            
            results.append(result)
            self._update_stats(result)
        
        self.logger.info(f"Analyzed {len(results)} historical messages from {channel}")
        return results
    
    def get_current_sentiment_state(self) -> Dict[str, Any]:
        """Get current sentiment analysis state and statistics.
        
        Returns:
            Dictionary with current state information
        """
        ewma_state = self.ewma_analyzer.get_current_state() if self.ewma_analyzer else {}
        
        return {
            'engine_stats': self.stats.copy(),
            'ewma_state': ewma_state,
            'is_running': self.is_running,
            'recent_anomalies': len(self.ewma_analyzer.get_recent_anomalies(1)) if self.ewma_analyzer else 0,
            'model_info': self.sentiment_analyzer.get_model_info() if self.sentiment_analyzer else {}
        }
    
    def get_recent_anomalies(self, hours: int = 1) -> List[AnomalyEvent]:
        """Get recent anomalies detected by the engine.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List of recent AnomalyEvent objects
        """
        if not self.ewma_analyzer:
            return []
        
        return self.ewma_analyzer.get_recent_anomalies(hours)


# Factory function for easy instantiation
async def create_sentiment_engine() -> SentimentEngine:
    """Create and initialize a complete sentiment engine.
    
    Returns:
        Initialized SentimentEngine instance
    """
    engine = SentimentEngine()
    await engine.initialize()
    return engine