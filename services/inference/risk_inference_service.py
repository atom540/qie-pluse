"""
Core Risk Inference Service
Provides real-time risk score predictions using trained ML models
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
from dataclasses import dataclass

from config import get_config
from logging_config import get_data_logger, get_performance_logger
from services.timeseries.ml_trainer import XGBoostRiskModel, MLModelTrainer, ModelPrediction
from services.timeseries.oracle_processor import OracleDataProcessor, FeatureSet
from services.timeseries.correlation_engine import CorrelationEngine
from services.collectors.crypto_collector import CryptoDataCollector
from services.collectors.traditional_asset_collector import TraditionalAssetCollector
from services.sentiment.roberta_analyzer import RoBERTaSentimentAnalyzer
from services.sentiment.telegram_client import TelegramMessageProcessor

@dataclass
class RiskAssessment:
    """Complete risk assessment result"""
    asset_symbol: str
    timestamp: datetime
    
    # Risk scores (0.0 to 1.0)
    ml_risk_score: float
    sentiment_risk_score: float
    combined_risk_score: float
    
    # Confidence and metadata
    confidence: float
    risk_level: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    
    # Supporting data
    market_regime: str
    correlation_signals: List[str]
    feature_importance: Dict[str, float]
    
    # Performance metrics
    inference_latency_ms: float
    data_freshness_seconds: int

class RiskInferenceService:
    """Core service for real-time risk inference"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.config = get_config()
        self.data_logger = get_data_logger()
        self.performance_logger = get_performance_logger()
        
        # Initialize components
        self.ml_trainer = MLModelTrainer()
        self.oracle_processor = OracleDataProcessor()
        self.correlation_engine = CorrelationEngine()
        
        # Data collectors
        self.crypto_collector = CryptoDataCollector()
        self.traditional_collector = TraditionalAssetCollector()
        
        # Sentiment analysis
        self.sentiment_analyzer = RoBERTaSentimentAnalyzer()
        
        # Initialize Telegram client with config
        try:
            telegram_config = getattr(self.config, 'telegram', {})
            api_id = telegram_config.get('api_id')
            api_hash = telegram_config.get('api_hash') 
            phone_number = telegram_config.get('phone_number')
            
            if api_id and api_hash and phone_number:
                self.telegram_client = TelegramMessageProcessor(api_id, api_hash, phone_number)
            else:
                self.telegram_client = None
                self.data_logger.log_data_collection(
                    "telegram_init", "config", 0, False, 
                    "Telegram credentials not configured - sentiment analysis will use fallback"
                )
        except Exception as e:
            self.telegram_client = None
            self.data_logger.log_data_collection(
                "telegram_init", "error", 0, False, str(e)
            )
        
        # Model state
        self.current_model: Optional[XGBoostRiskModel] = None
        self.model_version: str = "unknown"
        self.model_loaded_at: Optional[datetime] = None
        
        # Cache for recent data
        self.data_cache: Dict[str, Any] = {}
        self.cache_ttl_seconds = 60  # 1 minute cache
        
        # Load model if path provided
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> bool:
        """Load trained model for inference"""
        try:
            start_time = time.time()
            
            # Load the model
            self.current_model = self.ml_trainer.load_model(model_path)
            self.model_version = self.current_model.model_version
            self.model_loaded_at = datetime.now()
            
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time("model_loading", duration_ms, True)
            
            self.data_logger.log_data_collection(
                "model_loading", model_path, 1, True,
                f"Model {self.model_version} loaded successfully"
            )
            
            return True
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time("model_loading", duration_ms, False)
            
            self.data_logger.log_data_collection(
                "model_loading", model_path, 0, False, str(e)
            )
            return False
    
    def load_latest_model(self) -> bool:
        """Load the most recent trained model"""
        try:
            models_dir = Path("models/trained")
            if not models_dir.exists():
                raise FileNotFoundError("No trained models directory found")
            
            # Find the most recent model file
            model_files = list(models_dir.glob("risk_model_*.pkl"))
            if not model_files:
                raise FileNotFoundError("No trained model files found")
            
            # Sort by modification time, get the newest
            latest_model = max(model_files, key=lambda f: f.stat().st_mtime)
            
            return self.load_model(str(latest_model))
            
        except Exception as e:
            self.data_logger.log_data_collection(
                "model_loading", "latest", 0, False, str(e)
            )
            return False
    
    async def get_risk_assessment(self, asset: str) -> Optional[RiskAssessment]:
        """Get comprehensive risk assessment for an asset"""
        start_time = time.time()
        
        try:
            if not self.current_model:
                raise ValueError("No model loaded. Call load_model() first.")
            
            # Step 1: Collect real-time data
            market_data = await self._collect_real_time_data(asset)
            if not market_data:
                raise ValueError(f"No market data available for {asset}")
            
            # Step 2: Prepare features
            feature_set = await self._prepare_features(asset, market_data)
            if not feature_set:
                raise ValueError(f"Could not prepare features for {asset}")
            
            # Step 3: Get ML prediction
            ml_prediction = await self._get_ml_prediction(feature_set)
            
            # Step 4: Get sentiment score
            sentiment_score = await self._get_sentiment_risk_score(asset)
            
            # Step 5: Combine scores
            combined_score = self._combine_risk_scores(
                ml_prediction.risk_score, sentiment_score
            )
            
            # Step 6: Determine risk level
            risk_level = self._determine_risk_level(combined_score)
            
            # Step 7: Calculate data freshness
            data_freshness = self._calculate_data_freshness(market_data)
            
            # Create assessment
            assessment = RiskAssessment(
                asset_symbol=asset,
                timestamp=datetime.now(),
                ml_risk_score=ml_prediction.risk_score,
                sentiment_risk_score=sentiment_score,
                combined_risk_score=combined_score,
                confidence=ml_prediction.confidence,
                risk_level=risk_level,
                market_regime=ml_prediction.market_regime or "normal",
                correlation_signals=ml_prediction.correlation_signals or [],
                feature_importance=ml_prediction.feature_importance or {},
                inference_latency_ms=(time.time() - start_time) * 1000,
                data_freshness_seconds=data_freshness
            )
            
            # Log successful inference
            self.performance_logger.log_inference_time(
                f"risk_assessment_{asset.lower()}", assessment.inference_latency_ms, True
            )
            
            self.data_logger.log_data_collection(
                "risk_assessment", asset, 1, True,
                f"Risk: {combined_score:.3f}, Level: {risk_level}"
            )
            
            return assessment
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time(
                f"risk_assessment_{asset.lower()}", duration_ms, False
            )
            
            self.data_logger.log_data_collection(
                "risk_assessment", asset, 0, False, str(e)
            )
            return None
    
    async def get_multi_asset_assessment(self, 
                                       assets: List[str] = None) -> Dict[str, RiskAssessment]:
        """Get risk assessments for multiple assets"""
        if assets is None:
            assets = ["BTC", "ETH", "XRP", "SOL", "BNB", "GOLD"]
        
        assessments = {}
        
        # Process assets concurrently
        tasks = [self.get_risk_assessment(asset) for asset in assets]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for asset, result in zip(assets, results):
            if isinstance(result, RiskAssessment):
                assessments[asset] = result
            else:
                self.data_logger.log_data_collection(
                    "multi_asset_assessment", asset, 0, False,
                    f"Failed: {str(result) if isinstance(result, Exception) else 'Unknown error'}"
                )
        
        return assessments
    
    async def _collect_real_time_data(self, asset: str) -> Optional[Dict[str, Any]]:
        """Collect real-time market data for an asset"""
        try:
            # Check cache first
            cache_key = f"market_data_{asset}"
            if cache_key in self.data_cache:
                cached_data, cached_time = self.data_cache[cache_key]
                if (datetime.now() - cached_time).seconds < self.cache_ttl_seconds:
                    return cached_data
            
            # Collect fresh data
            if asset == "GOLD":
                # Traditional asset
                data_list = await self.traditional_collector.collect_current_prices(
                    "alphavantage"
                )
                # Find the specific asset data
                data = None
                for item in data_list:
                    if hasattr(item, 'symbol') and item.symbol == asset:
                        data = item
                        break
            else:
                # Crypto asset
                data_list = await self.crypto_collector.collect_current_prices(
                    "coingecko"
                )
                # Find the specific asset data
                data = None
                for item in data_list:
                    if hasattr(item, 'symbol') and item.symbol == asset:
                        data = item
                        break
            
            if data:
                # Cache the data
                self.data_cache[cache_key] = (data, datetime.now())
                return {"current_data": data, "asset": asset}
            
            return None
            
        except Exception as e:
            self.data_logger.log_data_collection(
                "real_time_data", asset, 0, False, str(e)
            )
            return None
    
    async def _prepare_features(self, asset: str, market_data: Dict[str, Any]) -> Optional[FeatureSet]:
        """Prepare features for ML inference"""
        try:
            # Get recent historical data for feature calculation
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)  # Last week for features
            
            # Collect recent historical data
            if asset == "GOLD":
                historical_data = await self.traditional_collector.collect_historical_prices(
                    "alphavantage", asset, start_date, end_date
                )
            else:
                historical_data = await self.crypto_collector.collect_historical_prices(
                    "coingecko", asset, start_date, end_date
                )
            
            if not historical_data:
                return None
            
            # Add current data point
            current_data = market_data["current_data"]
            historical_data.append(current_data)
            
            # Calculate technical indicators
            indicators = self.oracle_processor.indicator_calculator.calculate_all_indicators(
                historical_data, asset
            )
            
            # Get correlation data
            correlation_data = await self._get_correlation_data()
            
            # Create feature set
            feature_set = self.oracle_processor.feature_engineer.create_feature_set(
                asset, historical_data, indicators, correlation_data
            )
            
            # Set current timestamp
            feature_set.timestamp = datetime.now()
            
            return feature_set
            
        except Exception as e:
            self.data_logger.log_data_collection(
                "feature_preparation", asset, 0, False, str(e)
            )
            return None
    
    async def _get_correlation_data(self) -> Dict[str, Any]:
        """Get correlation data for feature engineering"""
        try:
            # Check cache
            cache_key = "correlation_data"
            if cache_key in self.data_cache:
                cached_data, cached_time = self.data_cache[cache_key]
                if (datetime.now() - cached_time).seconds < self.cache_ttl_seconds * 5:  # 5 minute cache
                    return cached_data
            
            # Collect data for major assets
            assets = ["BTC", "ETH", "XRP", "SOL", "BNB", "GOLD"]
            correlation_data = {}
            
            # Get crypto data
            crypto_data_list = await self.crypto_collector.collect_current_prices("coingecko")
            for item in crypto_data_list:
                if hasattr(item, 'symbol') and item.symbol in assets:
                    correlation_data[item.symbol] = [item]
            
            # Get traditional data (Gold)
            try:
                traditional_data_list = await self.traditional_collector.collect_current_prices("alphavantage")
                for item in traditional_data_list:
                    if hasattr(item, 'symbol') and item.symbol == "GOLD":
                        correlation_data["GOLD"] = [item]
            except Exception as e:
                # Gold data might not be available, continue without it
                self.data_logger.log_data_collection(
                    "correlation_data", "GOLD", 0, False, str(e)
                )
            
            # Cache the data
            self.data_cache[cache_key] = (correlation_data, datetime.now())
            
            return correlation_data
            
        except Exception as e:
            self.data_logger.log_data_collection(
                "correlation_data", "all_assets", 0, False, str(e)
            )
            return {}
    
    async def _get_ml_prediction(self, feature_set: FeatureSet) -> ModelPrediction:
        """Get ML prediction using a properly fitted preprocessor"""
        try:
            from services.timeseries.ml_trainer import FeaturePreprocessor
            
            # Create a new preprocessor for this prediction
            preprocessor = FeaturePreprocessor()
            
            # Prepare features
            feature_df = preprocessor.prepare_features_from_feature_sets([feature_set])
            
            if feature_df.empty:
                raise ValueError("No valid features for prediction")
            
            # Fit and transform features (for single prediction, we fit on the single sample)
            X = preprocessor.fit_transform(feature_df)
            
            if X.size == 0:
                raise ValueError("No valid feature matrix for prediction")
            
            # Make prediction
            predictions, probabilities = self.current_model.predict(X, return_probabilities=True)
            
            # Extract results
            risk_score = float(probabilities[0])
            price_drop_probability = risk_score
            
            # Estimate volatility from features
            volatility_prediction = feature_set.current_volatility or 0.0
            
            # Calculate confidence based on feature quality
            confidence = self._calculate_prediction_confidence(feature_set, risk_score)
            
            # Determine market regime
            market_regime = self._determine_market_regime(feature_set)
            
            # Create prediction
            prediction = ModelPrediction(
                asset_symbol=feature_set.asset_symbol,
                prediction_timestamp=datetime.now(),
                risk_score=risk_score,
                price_drop_probability=price_drop_probability,
                volatility_prediction=volatility_prediction,
                confidence=confidence,
                model_version=self.current_model.model_version,
                feature_importance=self.current_model.feature_importance,
                input_features={
                    'current_price': feature_set.current_price,
                    'current_volatility': feature_set.current_volatility,
                    'price_change_24h': feature_set.price_change_24h,
                    'rsi_normalized': feature_set.rsi_normalized,
                    'btc_correlation': feature_set.btc_correlation
                },
                correlation_signals=[],
                market_regime=market_regime
            )
            
            return prediction
            
        except Exception as e:
            self.data_logger.log_data_collection(
                "ml_prediction", feature_set.asset_symbol, 0, False, str(e)
            )
            raise
    
    def _calculate_prediction_confidence(self, feature_set: FeatureSet, risk_score: float) -> float:
        """Calculate confidence score for prediction"""
        try:
            confidence_factors = []
            
            # Feature completeness
            total_features = 20  # Approximate number of key features
            non_null_features = sum(1 for attr in [
                feature_set.price_lag_1, feature_set.price_lag_3, feature_set.price_change_24h,
                feature_set.rsi_normalized, feature_set.btc_correlation, feature_set.gold_correlation,
                feature_set.volatility_percentile, feature_set.volume_anomaly_score
            ] if attr is not None)
            
            feature_completeness = non_null_features / total_features
            confidence_factors.append(feature_completeness)
            
            # Risk score certainty (closer to 0 or 1 is more certain)
            risk_certainty = 2 * abs(risk_score - 0.5)  # 0.5 is maximum uncertainty
            confidence_factors.append(risk_certainty)
            
            # Data recency (more recent data is more reliable)
            time_diff = (datetime.now() - feature_set.timestamp).total_seconds()
            recency_score = max(0, 1 - time_diff / 3600)  # Decay over 1 hour
            confidence_factors.append(recency_score)
            
            # Overall confidence
            confidence = sum(confidence_factors) / len(confidence_factors)
            return float(max(0.0, min(1.0, confidence)))
        
        except Exception:
            return 0.5  # Default moderate confidence
    
    def _determine_market_regime(self, feature_set: FeatureSet) -> str:
        """Determine current market regime"""
        try:
            volatility = feature_set.current_volatility or 0.0
            volatility_percentile = feature_set.volatility_percentile or 0.5
            volume_anomaly = feature_set.volume_anomaly_score or 0.0
            
            # High volatility regime
            if volatility_percentile > 0.8 or volume_anomaly > 2.0:
                return "high_volatility"
            
            # Crisis regime (extreme conditions)
            if volatility_percentile > 0.95 and volume_anomaly > 3.0:
                return "crisis"
            
            # Normal regime
            return "normal"
        
        except Exception:
            return "normal"
    
    async def _get_sentiment_risk_score(self, asset: str) -> float:
        """Get sentiment-based risk score"""
        try:
            # Check cache
            cache_key = f"sentiment_{asset}"
            if cache_key in self.data_cache:
                cached_score, cached_time = self.data_cache[cache_key]
                if (datetime.now() - cached_time).seconds < self.cache_ttl_seconds * 2:  # 2 minute cache
                    return cached_score
            
            # If no Telegram client, return neutral sentiment
            if not self.telegram_client:
                return 0.5
            
            # Get recent messages from Telegram
            try:
                messages = await self.telegram_client.get_recent_messages(
                    limit=50, hours_back=1
                )
            except Exception as e:
                self.data_logger.log_data_collection(
                    "telegram_messages", asset, 0, False, str(e)
                )
                return 0.5  # Fallback to neutral
            
            if not messages:
                return 0.5  # Neutral sentiment if no data
            
            # Analyze sentiment
            sentiment_scores = []
            for message in messages:
                if asset.lower() in message.text.lower() or "crypto" in message.text.lower():
                    sentiment = await self.sentiment_analyzer.analyze_sentiment(message.text)
                    if sentiment:
                        # Convert sentiment to risk score (negative sentiment = higher risk)
                        risk_score = 1.0 - sentiment.compound_score  # Invert sentiment
                        sentiment_scores.append(risk_score)
            
            if sentiment_scores:
                avg_sentiment_risk = sum(sentiment_scores) / len(sentiment_scores)
            else:
                avg_sentiment_risk = 0.5  # Neutral if no relevant messages
            
            # Cache the score
            self.data_cache[cache_key] = (avg_sentiment_risk, datetime.now())
            
            return avg_sentiment_risk
            
        except Exception as e:
            self.data_logger.log_data_collection(
                "sentiment_analysis", asset, 0, False, str(e)
            )
            return 0.5  # Default neutral sentiment
    
    def _combine_risk_scores(self, ml_score: float, sentiment_score: float) -> float:
        """Combine ML and sentiment risk scores"""
        try:
            # Weighted combination: 70% ML, 30% sentiment
            ml_weight = 0.7
            sentiment_weight = 0.3
            
            combined = (ml_score * ml_weight) + (sentiment_score * sentiment_weight)
            
            # Ensure score is between 0 and 1
            return max(0.0, min(1.0, combined))
            
        except Exception:
            return ml_score  # Fallback to ML score only
    
    def _determine_risk_level(self, risk_score: float) -> str:
        """Determine risk level from combined score"""
        if risk_score >= 0.8:
            return "CRITICAL"
        elif risk_score >= 0.6:
            return "HIGH"
        elif risk_score >= 0.4:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _calculate_data_freshness(self, market_data: Dict[str, Any]) -> int:
        """Calculate how fresh the market data is in seconds"""
        try:
            current_data = market_data.get("current_data")
            if current_data and hasattr(current_data, 'timestamp'):
                data_time = current_data.timestamp
                freshness = (datetime.now() - data_time).total_seconds()
                return int(freshness)
            else:
                return 0  # Assume fresh if no timestamp
        except Exception:
            return 0
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get current service status"""
        return {
            "service": "RiskInferenceService",
            "status": "active" if self.current_model else "no_model",
            "model_version": self.model_version,
            "model_loaded_at": self.model_loaded_at.isoformat() if self.model_loaded_at else None,
            "cache_size": len(self.data_cache),
            "components": {
                "ml_trainer": "available",
                "oracle_processor": "available", 
                "correlation_engine": "available",
                "sentiment_analyzer": "available",
                "telegram_client": "available" if self.telegram_client else "not_configured"
            }
        }
    
    def clear_cache(self):
        """Clear the data cache"""
        self.data_cache.clear()
        self.data_logger.log_data_collection(
            "cache_management", "clear", 1, True, "Cache cleared"
        )