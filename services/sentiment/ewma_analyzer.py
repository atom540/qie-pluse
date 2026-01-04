"""
EWMA smoothing and anomaly detection for sentiment analysis.

This module implements Exponentially Weighted Moving Average (EWMA) smoothing
for sentiment scores and anomaly detection for sentiment spikes.
"""

import logging
import asyncio
from typing import List, Dict, Optional, Tuple, Deque
from dataclasses import dataclass, field
from collections import deque
import numpy as np
from datetime import datetime, timedelta

from config import get_config
from .roberta_analyzer import SentimentResult


@dataclass
class EWMAState:
    """Represents the current EWMA state."""
    current_ewma: float
    variance: float
    count: int
    last_updated: datetime
    
    
@dataclass
class AnomalyEvent:
    """Represents a detected anomaly in sentiment."""
    timestamp: datetime
    raw_score: float
    ewma_score: float
    deviation: float
    confidence: float
    severity: str  # 'low', 'medium', 'high'
    description: str


class EWMAAnalyzer:
    """EWMA smoothing and anomaly detection for sentiment scores."""
    
    def __init__(self, alpha: Optional[float] = None, 
                 anomaly_threshold: Optional[float] = None,
                 window_size: int = 100):
        """Initialize the EWMA analyzer.
        
        Args:
            alpha: EWMA smoothing parameter (0 < alpha <= 1)
            anomaly_threshold: Standard deviations for anomaly detection
            window_size: Size of rolling window for variance calculation
        """
        config = get_config()
        self.alpha = alpha or config.model.ewma_alpha
        self.anomaly_threshold = anomaly_threshold or config.model.anomaly_threshold
        self.window_size = window_size
        self.logger = logging.getLogger(__name__)
        
        # EWMA state
        self.ewma_state: Optional[EWMAState] = None
        
        # Rolling window for variance calculation
        self.score_history: Deque[float] = deque(maxlen=window_size)
        self.ewma_history: Deque[float] = deque(maxlen=window_size)
        
        # Anomaly tracking
        self.recent_anomalies: List[AnomalyEvent] = []
        self.anomaly_history_hours = 24  # Keep 24 hours of anomaly history
        
        self.logger.info(f"EWMA Analyzer initialized with alpha={self.alpha}, "
                        f"threshold={self.anomaly_threshold}")
    
    def calculate_ewma(self, new_score: float) -> float:
        """Calculate EWMA for a new sentiment score.
        
        Args:
            new_score: New sentiment score (0.0-1.0)
            
        Returns:
            Updated EWMA value
        """
        if self.ewma_state is None:
            # Initialize EWMA with first score
            self.ewma_state = EWMAState(
                current_ewma=new_score,
                variance=0.0,
                count=1,
                last_updated=datetime.now()
            )
            ewma_value = new_score
        else:
            # Update EWMA: EWMA_t = α * X_t + (1-α) * EWMA_{t-1}
            ewma_value = self.alpha * new_score + (1 - self.alpha) * self.ewma_state.current_ewma
            
            # Update state
            self.ewma_state.current_ewma = ewma_value
            self.ewma_state.count += 1
            self.ewma_state.last_updated = datetime.now()
        
        # Update history
        self.score_history.append(new_score)
        self.ewma_history.append(ewma_value)
        
        # Update variance estimate
        self._update_variance()
        
        return ewma_value
    
    def _update_variance(self) -> None:
        """Update variance estimate using rolling window."""
        if len(self.score_history) < 2:
            return
        
        # Calculate variance of recent scores
        scores_array = np.array(list(self.score_history))
        variance = np.var(scores_array, ddof=1)
        
        if self.ewma_state:
            self.ewma_state.variance = variance
    
    def detect_anomaly(self, raw_score: float, ewma_score: float) -> Optional[AnomalyEvent]:
        """Detect if a sentiment score represents an anomaly.
        
        Args:
            raw_score: Raw sentiment score
            ewma_score: EWMA smoothed score
            
        Returns:
            AnomalyEvent if anomaly detected, None otherwise
        """
        if self.ewma_state is None or self.ewma_state.variance == 0:
            return None
        
        # Calculate deviation in standard deviations
        std_dev = np.sqrt(self.ewma_state.variance)
        if std_dev == 0:
            return None
        
        deviation = abs(raw_score - ewma_score) / std_dev
        
        # Check if deviation exceeds threshold
        if deviation >= self.anomaly_threshold:
            # Determine severity
            if deviation >= 3.0:
                severity = 'high'
            elif deviation >= 2.5:
                severity = 'medium'
            else:
                severity = 'low'
            
            # Calculate confidence (normalized deviation)
            confidence = min(deviation / 4.0, 1.0)  # Cap at 1.0
            
            # Create description
            direction = "spike" if raw_score > ewma_score else "drop"
            description = f"Sentiment {direction} detected: {deviation:.2f}σ deviation"
            
            anomaly = AnomalyEvent(
                timestamp=datetime.now(),
                raw_score=raw_score,
                ewma_score=ewma_score,
                deviation=deviation,
                confidence=confidence,
                severity=severity,
                description=description
            )
            
            # Add to recent anomalies
            self.recent_anomalies.append(anomaly)
            self._cleanup_old_anomalies()
            
            self.logger.warning(f"Anomaly detected: {description}")
            return anomaly
        
        return None
    
    def _cleanup_old_anomalies(self) -> None:
        """Remove anomalies older than the specified history window."""
        cutoff_time = datetime.now() - timedelta(hours=self.anomaly_history_hours)
        self.recent_anomalies = [
            anomaly for anomaly in self.recent_anomalies 
            if anomaly.timestamp > cutoff_time
        ]
    
    def process_sentiment_score(self, raw_score: float) -> Tuple[float, Optional[AnomalyEvent]]:
        """Process a new sentiment score with EWMA smoothing and anomaly detection.
        
        Args:
            raw_score: Raw sentiment score (0.0-1.0)
            
        Returns:
            Tuple of (smoothed_score, anomaly_event_or_none)
        """
        # Calculate EWMA
        ewma_score = self.calculate_ewma(raw_score)
        
        # Detect anomaly
        anomaly = self.detect_anomaly(raw_score, ewma_score)
        
        return ewma_score, anomaly
    
    async def process_sentiment_results(self, results: List[SentimentResult]) -> List[Tuple[SentimentResult, float, Optional[AnomalyEvent]]]:
        """Process multiple sentiment results with EWMA smoothing.
        
        Args:
            results: List of SentimentResult objects
            
        Returns:
            List of tuples (original_result, smoothed_score, anomaly_or_none)
        """
        processed_results = []
        
        for result in results:
            smoothed_score, anomaly = self.process_sentiment_score(result.sentiment_score)
            processed_results.append((result, smoothed_score, anomaly))
        
        return processed_results
    
    def get_current_state(self) -> Dict:
        """Get current EWMA state and statistics.
        
        Returns:
            Dictionary with current state information
        """
        if self.ewma_state is None:
            return {
                'initialized': False,
                'current_ewma': None,
                'variance': None,
                'count': 0,
                'recent_anomalies_count': 0
            }
        
        return {
            'initialized': True,
            'current_ewma': self.ewma_state.current_ewma,
            'variance': self.ewma_state.variance,
            'count': self.ewma_state.count,
            'last_updated': self.ewma_state.last_updated.isoformat(),
            'recent_anomalies_count': len(self.recent_anomalies),
            'alpha': self.alpha,
            'anomaly_threshold': self.anomaly_threshold,
            'window_size': self.window_size
        }
    
    def get_recent_anomalies(self, hours: int = 1) -> List[AnomalyEvent]:
        """Get anomalies from the last N hours.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List of recent AnomalyEvent objects
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            anomaly for anomaly in self.recent_anomalies 
            if anomaly.timestamp > cutoff_time
        ]
    
    def calculate_confidence_score(self, raw_scores: List[float]) -> float:
        """Calculate confidence score for anomalous events.
        
        Args:
            raw_scores: List of recent raw sentiment scores
            
        Returns:
            Confidence score (0.0-1.0)
        """
        if len(raw_scores) < 2:
            return 0.0
        
        # Calculate volatility
        volatility = np.std(raw_scores)
        
        # Calculate trend strength
        if len(raw_scores) >= 3:
            # Simple trend detection using linear regression slope
            x = np.arange(len(raw_scores))
            slope = np.polyfit(x, raw_scores, 1)[0]
            trend_strength = abs(slope)
        else:
            trend_strength = 0.0
        
        # Combine volatility and trend for confidence
        # High volatility + strong trend = high confidence in anomaly
        confidence = min((volatility * 2.0 + trend_strength * 3.0) / 2.0, 1.0)
        
        return confidence
    
    def reset_state(self) -> None:
        """Reset EWMA state and history."""
        self.ewma_state = None
        self.score_history.clear()
        self.ewma_history.clear()
        self.recent_anomalies.clear()
        self.logger.info("EWMA state reset")


# Factory function for easy instantiation
def create_ewma_analyzer(alpha: Optional[float] = None, 
                        anomaly_threshold: Optional[float] = None,
                        window_size: int = 100) -> EWMAAnalyzer:
    """Create an EWMA analyzer with configuration.
    
    Args:
        alpha: EWMA smoothing parameter
        anomaly_threshold: Anomaly detection threshold
        window_size: Rolling window size
        
    Returns:
        Configured EWMAAnalyzer instance
    """
    return EWMAAnalyzer(
        alpha=alpha,
        anomaly_threshold=anomaly_threshold,
        window_size=window_size
    )