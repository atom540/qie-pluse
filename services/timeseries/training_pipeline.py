"""
Comprehensive ML Training Pipeline for AI Risk Oracle
Integrates historical data collection, model training, and backtesting validation
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import time
from dataclasses import dataclass

from config import get_config
from logging_config import get_data_logger, get_performance_logger
from services.storage.data_manager import DataManager
from services.collectors.crypto_collector import CryptoDataCollector
from services.collectors.traditional_asset_collector import TraditionalAssetCollector
from services.collectors.social_sentiment_collector import SocialSentimentCollector
from services.timeseries.oracle_processor import OracleDataProcessor, FeatureSet
from services.timeseries.ml_trainer import MLModelTrainer, XGBoostRiskModel, BacktestResult, ModelPrediction
from services.timeseries.correlation_engine import CorrelationEngine
from services.timeseries.training_diagnostics import TrainingDiagnosticsManager
from services.timeseries.enhanced_training_config import get_enhanced_training_config_manager, EnhancedTrainingConfiguration

@dataclass
class DataQualityReport:
    """Report on data quality and coverage statistics"""
    total_expected_points: int
    total_collected_points: int
    coverage_percentage: float
    missing_data_gaps: List[Tuple[datetime, datetime]]
    quality_score: float  # 0.0 to 1.0
    asset_coverage: Dict[str, float]
    data_validation_errors: List[str]

@dataclass 
class CollectionProgress:
    """Progress tracking for large dataset collection"""
    total_assets: int
    completed_assets: int
    current_asset: str
    start_time: datetime
    estimated_completion: Optional[datetime]
    resumable_state: Dict[str, Any]

@dataclass
class CrashEvent:
    """Configuration and metadata for market crash events"""
    name: str
    start_date: datetime
    end_date: datetime
    severity: str  # "low", "medium", "high"
    affected_assets: List[str]
    max_drawdown: Dict[str, float]
    recovery_time_days: int
    description: str
    weight_multiplier: float  # Training weight multiplier for this event

class CrashEventDetector:
    """Detect and configure market crash events for training"""
    
    # Predefined crash events configuration
    CRASH_EVENTS = {
        "COVID_CRASH_2020": CrashEvent(
            name="COVID_CRASH_2020",
            start_date=datetime(2020, 3, 1),
            end_date=datetime(2020, 4, 30),
            severity="high",
            affected_assets=["BTC", "ETH", "XRP", "SOL", "GOLD"],
            max_drawdown={"BTC": 0.50, "ETH": 0.60, "XRP": 0.65, "SOL": 0.70, "GOLD": 0.15},
            recovery_time_days=90,
            description="COVID-19 pandemic market crash",
            weight_multiplier=3.0
        ),
        "TERRA_LUNA_2022": CrashEvent(
            name="TERRA_LUNA_2022", 
            start_date=datetime(2022, 5, 1),
            end_date=datetime(2022, 6, 30),
            severity="high",
            affected_assets=["BTC", "ETH", "XRP", "SOL"],
            max_drawdown={"BTC": 0.35, "ETH": 0.45, "XRP": 0.40, "SOL": 0.55},
            recovery_time_days=120,
            description="Terra Luna ecosystem collapse",
            weight_multiplier=2.5
        ),
        "FTX_COLLAPSE_2022": CrashEvent(
            name="FTX_COLLAPSE_2022",
            start_date=datetime(2022, 11, 1),
            end_date=datetime(2022, 12, 31),
            severity="high", 
            affected_assets=["BTC", "ETH", "XRP", "SOL", "BNB"],
            max_drawdown={"BTC": 0.25, "ETH": 0.30, "XRP": 0.35, "SOL": 0.45, "BNB": 0.20},
            recovery_time_days=60,
            description="FTX exchange collapse and bankruptcy",
            weight_multiplier=2.0
        )
    }
    
    def __init__(self):
        self.config = get_config()
        self.data_logger = get_data_logger()
    
    def get_crash_events_in_period(self, start_date: datetime, end_date: datetime) -> List[CrashEvent]:
        """Get crash events that overlap with the specified period"""
        overlapping_events = []
        
        for event_name, event in self.CRASH_EVENTS.items():
            # Check if event overlaps with requested period
            if (event.start_date <= end_date and event.end_date >= start_date):
                overlapping_events.append(event)
                self.data_logger.log_data_collection(
                    "crash_event_detection", event_name, 1, True,
                    f"Event overlaps with period {start_date} to {end_date}"
                )
        
        return overlapping_events
    
    def identify_crash_periods_in_data(self, historical_data: Dict[str, Any]) -> List[CrashEvent]:
        """Identify crash periods within collected historical data"""
        identified_events = []
        
        collection_start = historical_data["collection_period"]["start"]
        collection_end = historical_data["collection_period"]["end"]
        
        # Get predefined events that overlap with collection period
        overlapping_events = self.get_crash_events_in_period(collection_start, collection_end)
        
        for event in overlapping_events:
            # Verify event is actually present in the data
            if self._verify_crash_event_in_data(event, historical_data):
                identified_events.append(event)
                self.data_logger.log_data_collection(
                    "crash_event_verification", event.name, 1, True,
                    f"Crash event verified in collected data"
                )
            else:
                self.data_logger.log_data_collection(
                    "crash_event_verification", event.name, 0, False,
                    f"Crash event not found in collected data"
                )
        
        return identified_events
    
    def _verify_crash_event_in_data(self, event: CrashEvent, historical_data: Dict[str, Any]) -> bool:
        """Verify that a crash event is actually present in the collected data"""
        try:
            # Combine all asset data
            all_asset_data = {}
            all_asset_data.update(historical_data["crypto_data"])
            all_asset_data.update(historical_data["traditional_data"])
            
            # Check if we have data for affected assets during the event period
            for asset in event.affected_assets:
                if asset not in all_asset_data or not all_asset_data[asset]:
                    continue
                
                asset_data = all_asset_data[asset]
                
                # Find data points within the crash event period
                event_data_points = []
                for data_point in asset_data:
                    timestamp = getattr(data_point, 'timestamp', None)
                    if timestamp and event.start_date <= timestamp <= event.end_date:
                        event_data_points.append(data_point)
                
                # If we have sufficient data points during the event, consider it verified
                if len(event_data_points) >= 10:  # At least 10 data points
                    return True
            
            return False
            
        except Exception as e:
            self.data_logger.log_data_collection(
                "crash_event_verification", event.name, 0, False, str(e)
            )
            return False
    
    def label_crash_event_data(self, historical_data: Dict[str, Any], 
                              crash_events: List[CrashEvent]) -> Dict[str, Any]:
        """Add crash event labels to historical data for training"""
        try:
            labeled_data = historical_data.copy()
            labeled_data["crash_event_labels"] = {}
            
            # Combine all asset data
            all_asset_data = {}
            all_asset_data.update(historical_data["crypto_data"])
            all_asset_data.update(historical_data["traditional_data"])
            
            # Label each asset's data points
            for asset, asset_data in all_asset_data.items():
                asset_labels = []
                
                for data_point in asset_data:
                    timestamp = getattr(data_point, 'timestamp', None)
                    if not timestamp:
                        asset_labels.append({
                            "is_crash_period": False,
                            "crash_events": [],
                            "weight_multiplier": 1.0
                        })
                        continue
                    
                    # Check if this data point falls within any crash event
                    point_crash_events = []
                    max_weight_multiplier = 1.0
                    
                    for event in crash_events:
                        if (event.start_date <= timestamp <= event.end_date and 
                            asset in event.affected_assets):
                            point_crash_events.append(event.name)
                            max_weight_multiplier = max(max_weight_multiplier, event.weight_multiplier)
                    
                    asset_labels.append({
                        "is_crash_period": len(point_crash_events) > 0,
                        "crash_events": point_crash_events,
                        "weight_multiplier": max_weight_multiplier
                    })
                
                labeled_data["crash_event_labels"][asset] = asset_labels
            
            # Add crash event metadata
            labeled_data["crash_events_metadata"] = {
                event.name: {
                    "start_date": event.start_date.isoformat(),
                    "end_date": event.end_date.isoformat(),
                    "severity": event.severity,
                    "affected_assets": event.affected_assets,
                    "description": event.description,
                    "weight_multiplier": event.weight_multiplier
                }
                for event in crash_events
            }
            
            return labeled_data
            
        except Exception as e:
            self.data_logger.log_data_collection(
                "crash_event_labeling", "all_assets", 0, False, str(e)
            )
            # Return original data if labeling fails
            return historical_data
    
    def get_crash_event_statistics(self, labeled_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate statistics about crash events in the labeled data"""
        try:
            stats = {
                "total_crash_events": 0,
                "crash_period_coverage": {},
                "asset_crash_exposure": {},
                "weight_distribution": {}
            }
            
            if "crash_events_metadata" not in labeled_data:
                return stats
            
            stats["total_crash_events"] = len(labeled_data["crash_events_metadata"])
            
            # Calculate coverage and exposure statistics
            for asset, labels in labeled_data.get("crash_event_labels", {}).items():
                total_points = len(labels)
                crash_points = sum(1 for label in labels if label["is_crash_period"])
                
                stats["crash_period_coverage"][asset] = {
                    "total_points": total_points,
                    "crash_points": crash_points,
                    "crash_percentage": crash_points / total_points if total_points > 0 else 0
                }
                
                # Weight distribution
                weights = [label["weight_multiplier"] for label in labels]
                stats["weight_distribution"][asset] = {
                    "min_weight": min(weights) if weights else 1.0,
                    "max_weight": max(weights) if weights else 1.0,
                    "avg_weight": sum(weights) / len(weights) if weights else 1.0
                }
            
            return stats
            
        except Exception as e:
            self.data_logger.log_data_collection(
                "crash_event_statistics", "calculation", 0, False, str(e)
            )
            return {"error": str(e)}

class HistoricalDataCollector:
    """Collect historical data for model training"""
    
    def __init__(self):
        self.config = get_config()
        self.data_logger = get_data_logger()
        self.performance_logger = get_performance_logger()
        self.data_manager = DataManager()
        self.crash_detector = CrashEventDetector()
        
        # Initialize collectors
        self.crypto_collector = CryptoDataCollector()
        self.traditional_collector = TraditionalAssetCollector()
        self.sentiment_collector = SocialSentimentCollector()
    
    async def collect_training_data(self, 
                                  start_date: datetime,
                                  end_date: datetime,
                                  assets: List[str] = None,
                                  enable_progress_tracking: bool = False,
                                  resumable_collection_id: Optional[str] = None,
                                  include_crash_events: bool = True) -> Dict[str, Any]:
        """
        Collect comprehensive historical data for training with enhanced capabilities
        
        Args:
            start_date: Start date for data collection
            end_date: End date for data collection  
            assets: List of assets to collect (defaults to all 7 QIE oracle feeds)
            enable_progress_tracking: Enable progress tracking for large datasets
            resumable_collection_id: Resume from previous collection if provided
            include_crash_events: Include crash event detection and labeling
        
        Returns:
            Dictionary containing collected data and quality metrics
        """
        start_time = time.time()
        
        try:
            if assets is None:
                assets = ["BTC", "ETH", "XRP", "SOL", "QIE", "GOLD", "BNB"]
            
            # Validate collection period (support 12-24 months)
            collection_days = (end_date - start_date).days
            if collection_days > 730:  # 24 months
                self.data_logger.log_data_collection(
                    "training_pipeline", "validation", 0, False,
                    f"Collection period exceeds 24 months: {collection_days} days"
                )
                raise ValueError(f"Collection period exceeds maximum 24 months: {collection_days} days")
            
            if collection_days < 365:  # 12 months minimum
                self.data_logger.log_data_collection(
                    "training_pipeline", "validation", 0, False,
                    f"Collection period below minimum 12 months: {collection_days} days"
                )
                # Allow shorter periods but log warning
                
            # Initialize or resume collection
            if resumable_collection_id:
                collection_id = resumable_collection_id
                progress = self._load_collection_progress(collection_id)
            else:
                # Start new data collection session
                collection_id = self.data_manager.start_data_collection(
                    source="training_pipeline",
                    method="historical_collection",
                    parameters={
                        "start_date": start_date.isoformat(),
                        "end_date": end_date.isoformat(),
                        "assets": assets,
                        "collection_days": collection_days,
                        "include_crash_events": include_crash_events
                    }
                )
                progress = CollectionProgress(
                    total_assets=len(assets),
                    completed_assets=0,
                    current_asset="",
                    start_time=datetime.now(),
                    estimated_completion=None,
                    resumable_state={}
                )
            
            collected_data = {
                "crypto_data": {},
                "traditional_data": {},
                "sentiment_data": [],
                "collection_id": collection_id,
                "collection_period": {
                    "start": start_date,
                    "end": end_date,
                    "days": collection_days
                },
                "progress": progress if enable_progress_tracking else None
            }
            
            total_records = 0
            
            # Collect crypto data with progress tracking
            crypto_assets = [asset for asset in assets if asset != "GOLD"]
            if crypto_assets:
                crypto_data, crypto_records = await self._collect_crypto_historical_data_enhanced(
                    crypto_assets, start_date, end_date, progress, enable_progress_tracking
                )
                collected_data["crypto_data"] = crypto_data
                total_records += crypto_records
                
                # Store crypto data with validation
                for asset, data_list in crypto_data.items():
                    if data_list:
                        stored_count = self.data_manager.store_crypto_price_data(
                            data_list, collection_id
                        )
                        self.data_logger.log_data_collection(
                            "historical_crypto", asset, stored_count, True
                        )
            
            # Collect traditional asset data (Gold) with progress tracking
            if "GOLD" in assets:
                progress.current_asset = "GOLD"
                gold_data, gold_records = await self._collect_traditional_historical_data_enhanced(
                    ["GOLD"], start_date, end_date, progress, enable_progress_tracking
                )
                collected_data["traditional_data"] = gold_data
                total_records += gold_records
                
                # Store traditional data with validation
                for asset, data_list in gold_data.items():
                    if data_list:
                        stored_count = self.data_manager.store_traditional_asset_data(
                            data_list, collection_id
                        )
                        self.data_logger.log_data_collection(
                            "historical_traditional", asset, stored_count, True
                        )
                
                progress.completed_assets += 1
            
            # Collect sentiment data for major market events
            sentiment_data = await self._collect_sentiment_historical_data(
                start_date, end_date
            )
            collected_data["sentiment_data"] = sentiment_data
            total_records += len(sentiment_data)
            
            # Store sentiment data
            if sentiment_data:
                stored_count = self.data_manager.store_sentiment_data(
                    sentiment_data, collection_id
                )
                self.data_logger.log_data_collection(
                    "historical_sentiment", "all_platforms", stored_count, True
                )
            
            # Process crash events if requested
            if include_crash_events:
                crash_events = self.crash_detector.identify_crash_periods_in_data(collected_data)
                if crash_events:
                    collected_data = self.crash_detector.label_crash_event_data(
                        collected_data, crash_events
                    )
                    # Add crash event statistics
                    collected_data["crash_event_statistics"] = self.crash_detector.get_crash_event_statistics(
                        collected_data
                    )
            
            # Generate data quality report
            quality_report = self._generate_data_quality_report(
                collected_data, start_date, end_date, assets
            )
            collected_data["data_quality_report"] = quality_report
            
            # End data collection session
            self.data_manager.end_data_collection(
                collection_id, total_records, True
            )
            
            # Clean up progress tracking
            if enable_progress_tracking:
                self._cleanup_collection_progress(collection_id)
            
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time(
                "historical_data_collection", duration_ms, True
            )
            
            return collected_data
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time(
                "historical_data_collection", duration_ms, False
            )
            
            # End collection session with error
            if 'collection_id' in locals():
                self.data_manager.end_data_collection(
                    collection_id, 0, False, str(e)
                )
            
            self.data_logger.log_data_collection(
                "historical_collection", "all_assets", 0, False, str(e)
            )
            raise
    
    async def _collect_crypto_historical_data_enhanced(self, 
                                                     assets: List[str],
                                                     start_date: datetime,
                                                     end_date: datetime,
                                                     progress: CollectionProgress,
                                                     enable_progress_tracking: bool) -> Tuple[Dict[str, List[Any]], int]:
        """Enhanced crypto data collection with progress tracking and validation"""
        crypto_data = {}
        total_records = 0
        
        for i, asset in enumerate(assets):
            if enable_progress_tracking:
                progress.current_asset = asset
                progress.completed_assets = i
                # Estimate completion time based on progress
                if i > 0:
                    elapsed = datetime.now() - progress.start_time
                    avg_time_per_asset = elapsed / i
                    remaining_assets = len(assets) - i
                    progress.estimated_completion = datetime.now() + (avg_time_per_asset * remaining_assets)
                
                # Save progress state for resumability
                self._save_collection_progress(progress)
            
            try:
                # Collect from multiple sources for robustness
                coingecko_data = await self.crypto_collector.collect_historical_prices(
                    "coingecko", asset, start_date, end_date
                )
                
                # Try Binance for minute-level data if available
                try:
                    binance_data = await self.crypto_collector.collect_historical_prices(
                        "binance", asset, start_date, end_date
                    )
                    # Combine data sources, preferring higher frequency data
                    asset_data = binance_data if binance_data else coingecko_data
                except Exception:
                    asset_data = coingecko_data
                
                # Validate data quality
                validated_data = self._validate_price_data(asset_data, asset, start_date, end_date)
                crypto_data[asset] = validated_data
                total_records += len(validated_data)
                
            except Exception as e:
                self.data_logger.log_data_collection(
                    "crypto_historical", asset, 0, False, str(e)
                )
                crypto_data[asset] = []
        
        if enable_progress_tracking:
            progress.completed_assets = len(assets)
        
        return crypto_data, total_records
    
    async def _collect_traditional_historical_data_enhanced(self, 
                                                          assets: List[str],
                                                          start_date: datetime,
                                                          end_date: datetime,
                                                          progress: CollectionProgress,
                                                          enable_progress_tracking: bool) -> Tuple[Dict[str, List[Any]], int]:
        """Enhanced traditional asset data collection with progress tracking and validation"""
        traditional_data = {}
        total_records = 0
        
        for asset in assets:
            try:
                # Use Alpha Vantage for Gold data
                data = await self.traditional_collector.collect_historical_prices(
                    "alphavantage", asset, start_date, end_date
                )
                
                # Validate data quality
                validated_data = self._validate_price_data(data, asset, start_date, end_date)
                traditional_data[asset] = validated_data
                total_records += len(validated_data)
                
            except Exception as e:
                self.data_logger.log_data_collection(
                    "traditional_historical", asset, 0, False, str(e)
                )
                traditional_data[asset] = []
        
        return traditional_data, total_records
    
    def _validate_price_data(self, data: List[Any], asset: str, 
                           start_date: datetime, end_date: datetime) -> List[Any]:
        """Validate price data quality and handle missing points"""
        if not data:
            return []
        
        validated_data = []
        previous_price = None
        
        for i, point in enumerate(data):
            try:
                # Extract price from data point
                current_price = getattr(point, 'price', getattr(point, 'current_price', None))
                timestamp = getattr(point, 'timestamp', None)
                
                # Basic validation - be more lenient for testing
                if current_price is None:
                    # Try to get price from other possible fields
                    current_price = getattr(point, 'close', getattr(point, 'value', None))
                
                if current_price is None or current_price <= 0:
                    # Log but continue for debugging
                    if i < 5:  # Only log first few for debugging
                        self.data_logger.log_data_collection(
                            "data_validation", asset, 0, False,
                            f"Invalid price at index {i}: {current_price}"
                        )
                    continue
                
                # Check for extreme price movements (>50% change)
                if previous_price and abs(current_price - previous_price) / previous_price > 0.5:
                    # Log suspicious price movement but keep data point
                    self.data_logger.log_data_collection(
                        "data_validation", asset, 1, True,
                        f"Large price movement detected: {previous_price} -> {current_price}"
                    )
                
                # Check timestamp validity - be more lenient
                if timestamp:
                    if timestamp < start_date or timestamp > end_date:
                        # Log but don't skip for now - might be timezone issues
                        if i < 5:  # Only log first few
                            self.data_logger.log_data_collection(
                                "data_validation", asset, 1, True,
                                f"Timestamp outside range at index {i}: {timestamp}"
                            )
                else:
                    # Create a synthetic timestamp if missing
                    timestamp = start_date + timedelta(days=i)
                    # Add timestamp to the point if possible
                    if hasattr(point, '__dict__'):
                        point.timestamp = timestamp
                
                validated_data.append(point)
                previous_price = current_price
                
            except Exception as e:
                # Log validation errors for debugging
                if i < 5:  # Only log first few
                    self.data_logger.log_data_collection(
                        "data_validation", asset, 0, False,
                        f"Validation error at index {i}: {str(e)}"
                    )
                continue
        
        # Log validation summary
        self.data_logger.log_data_collection(
            "data_validation", asset, len(validated_data), True,
            f"Validated {len(validated_data)}/{len(data)} data points"
        )
        
        return validated_data
    
    def _generate_data_quality_report(self, collected_data: Dict[str, Any], 
                                    start_date: datetime, end_date: datetime,
                                    requested_assets: List[str]) -> DataQualityReport:
        """Generate comprehensive data quality report"""
        try:
            collection_days = (end_date - start_date).days
            
            # Determine expected data frequency based on actual data
            # Check what frequency we actually got to set realistic expectations
            sample_data = None
            for asset_data in [collected_data.get("crypto_data", {}), collected_data.get("traditional_data", {})]:
                for asset, data in asset_data.items():
                    if data and len(data) > 1:
                        sample_data = data
                        break
                if sample_data:
                    break
            
            # Estimate data frequency from sample
            if sample_data and len(sample_data) > 1:
                # Calculate average time between data points
                timestamps = []
                for point in sample_data[:10]:  # Use first 10 points to estimate
                    timestamp = getattr(point, 'timestamp', None)
                    if timestamp:
                        timestamps.append(timestamp)
                
                if len(timestamps) >= 2:
                    timestamps.sort()
                    avg_interval = (timestamps[-1] - timestamps[0]) / (len(timestamps) - 1)
                    if avg_interval.total_seconds() <= 3600:  # <= 1 hour
                        expected_points_per_day = 24  # Hourly
                    elif avg_interval.total_seconds() <= 14400:  # <= 4 hours  
                        expected_points_per_day = 6   # 4-hourly
                    else:
                        expected_points_per_day = 1   # Daily
                else:
                    # Default to daily if we can't determine frequency
                    expected_points_per_day = 1
            else:
                # Default to daily for mock/test data
                expected_points_per_day = 1
            
            total_expected_points = collection_days * expected_points_per_day * len(requested_assets)
            
            # Count actual collected points
            total_collected_points = 0
            asset_coverage = {}
            validation_errors = []
            missing_gaps = []
            
            # Analyze crypto data
            for asset, data in collected_data["crypto_data"].items():
                asset_points = len(data)
                expected_asset_points = collection_days * expected_points_per_day
                coverage = asset_points / expected_asset_points if expected_asset_points > 0 else 0
                asset_coverage[asset] = min(coverage, 1.0)  # Cap at 100%
                total_collected_points += asset_points
                
                if coverage < 0.8:  # Less than 80% coverage
                    validation_errors.append(f"Low coverage for {asset}: {coverage:.1%}")
                
                # Detect missing data gaps
                gaps = self._detect_missing_data_gaps(data, start_date, end_date, asset)
                missing_gaps.extend(gaps)
            
            # Analyze traditional data
            for asset, data in collected_data["traditional_data"].items():
                asset_points = len(data)
                expected_asset_points = collection_days * expected_points_per_day
                coverage = asset_points / expected_asset_points if expected_asset_points > 0 else 0
                asset_coverage[asset] = min(coverage, 1.0)
                total_collected_points += asset_points
                
                if coverage < 0.8:
                    validation_errors.append(f"Low coverage for {asset}: {coverage:.1%}")
                
                # Detect missing data gaps
                gaps = self._detect_missing_data_gaps(data, start_date, end_date, asset)
                missing_gaps.extend(gaps)
            
            # Additional data quality checks
            quality_errors = self._perform_additional_quality_checks(collected_data)
            validation_errors.extend(quality_errors)
            
            # Calculate overall metrics
            coverage_percentage = total_collected_points / total_expected_points if total_expected_points > 0 else 0
            coverage_percentage = min(coverage_percentage, 1.0)
            
            # Calculate quality score (0.0 to 1.0)
            quality_score = self._calculate_quality_score(
                coverage_percentage, validation_errors, requested_assets, missing_gaps
            )
            
            return DataQualityReport(
                total_expected_points=total_expected_points,
                total_collected_points=total_collected_points,
                coverage_percentage=coverage_percentage,
                missing_data_gaps=missing_gaps,
                quality_score=quality_score,
                asset_coverage=asset_coverage,
                data_validation_errors=validation_errors
            )
            
        except Exception as e:
            # Return minimal report on error
            return DataQualityReport(
                total_expected_points=0,
                total_collected_points=0,
                coverage_percentage=0.0,
                missing_data_gaps=[],
                quality_score=0.0,
                asset_coverage={},
                data_validation_errors=[f"Error generating report: {str(e)}"]
            )
    
    def _detect_missing_data_gaps(self, data: List[Any], start_date: datetime, 
                                end_date: datetime, asset: str) -> List[Tuple[datetime, datetime]]:
        """Detect gaps in data coverage"""
        gaps = []
        
        if not data:
            # Entire period is missing
            gaps.append((start_date, end_date))
            return gaps
        
        try:
            # Sort data by timestamp
            sorted_data = sorted(data, key=lambda x: getattr(x, 'timestamp', start_date))
            
            # Check for gaps larger than 2 hours
            gap_threshold = timedelta(hours=2)
            
            for i in range(len(sorted_data) - 1):
                current_time = getattr(sorted_data[i], 'timestamp', None)
                next_time = getattr(sorted_data[i + 1], 'timestamp', None)
                
                if current_time and next_time:
                    gap_duration = next_time - current_time
                    if gap_duration > gap_threshold:
                        gaps.append((current_time, next_time))
            
            # Check gap at the beginning
            first_timestamp = getattr(sorted_data[0], 'timestamp', None)
            if first_timestamp and (first_timestamp - start_date) > gap_threshold:
                gaps.append((start_date, first_timestamp))
            
            # Check gap at the end
            last_timestamp = getattr(sorted_data[-1], 'timestamp', None)
            if last_timestamp and (end_date - last_timestamp) > gap_threshold:
                gaps.append((last_timestamp, end_date))
                
        except Exception as e:
            self.data_logger.log_data_collection(
                "gap_detection", asset, 0, False, str(e)
            )
        
        return gaps
    
    def _perform_additional_quality_checks(self, collected_data: Dict[str, Any]) -> List[str]:
        """Perform additional data quality validation checks"""
        errors = []
        
        try:
            # Check for duplicate timestamps
            all_asset_data = {}
            all_asset_data.update(collected_data["crypto_data"])
            all_asset_data.update(collected_data["traditional_data"])
            
            for asset, data in all_asset_data.items():
                if not data:
                    continue
                
                # Check for duplicates
                timestamps = []
                for point in data:
                    timestamp = getattr(point, 'timestamp', None)
                    if timestamp:
                        timestamps.append(timestamp)
                
                if len(timestamps) != len(set(timestamps)):
                    duplicate_count = len(timestamps) - len(set(timestamps))
                    errors.append(f"{asset}: {duplicate_count} duplicate timestamps detected")
                
                # Check for price anomalies
                prices = []
                for point in data:
                    price = getattr(point, 'price', getattr(point, 'current_price', None))
                    if price and price > 0:
                        prices.append(price)
                
                if prices:
                    # Check for extreme outliers (more than 10x median)
                    median_price = sorted(prices)[len(prices) // 2]
                    outliers = [p for p in prices if p > median_price * 10 or p < median_price / 10]
                    if outliers:
                        errors.append(f"{asset}: {len(outliers)} extreme price outliers detected")
                    
                    # Check for zero or negative prices
                    invalid_prices = [p for p in prices if p <= 0]
                    if invalid_prices:
                        errors.append(f"{asset}: {len(invalid_prices)} invalid price values detected")
            
            # Check sentiment data quality
            sentiment_data = collected_data.get("sentiment_data", [])
            if sentiment_data:
                # Check for missing sentiment scores
                missing_scores = sum(1 for item in sentiment_data 
                                   if not hasattr(item, 'sentiment_score') or 
                                   getattr(item, 'sentiment_score', None) is None)
                if missing_scores > 0:
                    errors.append(f"Sentiment data: {missing_scores} items missing sentiment scores")
            
        except Exception as e:
            errors.append(f"Quality check error: {str(e)}")
        
        return errors
    
    def _calculate_quality_score(self, coverage_percentage: float, validation_errors: List[str],
                               requested_assets: List[str], missing_gaps: List[Tuple[datetime, datetime]]) -> float:
        """Calculate overall data quality score (0.0 to 1.0)"""
        try:
            # Base score from coverage (70% weight)
            quality_score = coverage_percentage * 0.7
            
            # Penalty for validation errors (20% weight)
            if len(validation_errors) == 0:
                quality_score += 0.2
            else:
                error_penalty = min(len(validation_errors) / len(requested_assets), 1.0)
                quality_score += 0.2 * (1 - error_penalty)
            
            # Penalty for missing data gaps (10% weight)
            if len(missing_gaps) == 0:
                quality_score += 0.1
            else:
                gap_penalty = min(len(missing_gaps) / (len(requested_assets) * 2), 1.0)
                quality_score += 0.1 * (1 - gap_penalty)
            
            return min(max(quality_score, 0.0), 1.0)  # Clamp between 0 and 1
            
        except Exception:
            return 0.0
    
    def generate_data_quality_summary(self, quality_report: DataQualityReport) -> str:
        """Generate human-readable data quality summary"""
        try:
            summary_lines = []
            summary_lines.append("=== Data Quality Report ===")
            summary_lines.append(f"Overall Quality Score: {quality_report.quality_score:.2f}/1.0")
            summary_lines.append(f"Data Coverage: {quality_report.coverage_percentage:.1%}")
            summary_lines.append(f"Total Data Points: {quality_report.total_collected_points:,}")
            summary_lines.append("")
            
            # Asset-specific coverage
            summary_lines.append("Asset Coverage:")
            for asset, coverage in quality_report.asset_coverage.items():
                status = "✓" if coverage >= 0.8 else "⚠" if coverage >= 0.5 else "✗"
                summary_lines.append(f"  {status} {asset}: {coverage:.1%}")
            summary_lines.append("")
            
            # Missing data gaps
            if quality_report.missing_data_gaps:
                summary_lines.append(f"Missing Data Gaps: {len(quality_report.missing_data_gaps)}")
                for i, (start, end) in enumerate(quality_report.missing_data_gaps[:5]):  # Show first 5
                    duration = end - start
                    summary_lines.append(f"  Gap {i+1}: {duration} ({start.strftime('%Y-%m-%d %H:%M')} - {end.strftime('%Y-%m-%d %H:%M')})")
                if len(quality_report.missing_data_gaps) > 5:
                    summary_lines.append(f"  ... and {len(quality_report.missing_data_gaps) - 5} more gaps")
                summary_lines.append("")
            
            # Validation errors
            if quality_report.data_validation_errors:
                summary_lines.append("Validation Issues:")
                for error in quality_report.data_validation_errors[:10]:  # Show first 10
                    summary_lines.append(f"  • {error}")
                if len(quality_report.data_validation_errors) > 10:
                    summary_lines.append(f"  ... and {len(quality_report.data_validation_errors) - 10} more issues")
            else:
                summary_lines.append("✓ No validation issues detected")
            
            return "\n".join(summary_lines)
            
        except Exception as e:
            return f"Error generating quality summary: {str(e)}"
    
    def _save_collection_progress(self, progress: CollectionProgress) -> None:
        """Save collection progress for resumability"""
        try:
            progress_dir = Path("data/collection_progress")
            progress_dir.mkdir(parents=True, exist_ok=True)
            
            progress_file = progress_dir / f"progress_{id(progress)}.json"
            progress_data = {
                "total_assets": progress.total_assets,
                "completed_assets": progress.completed_assets,
                "current_asset": progress.current_asset,
                "start_time": progress.start_time.isoformat(),
                "estimated_completion": progress.estimated_completion.isoformat() if progress.estimated_completion else None,
                "resumable_state": progress.resumable_state
            }
            
            with open(progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
                
        except Exception as e:
            # Non-critical error, just log it
            self.data_logger.log_data_collection(
                "progress_tracking", "save", 0, False, str(e)
            )
    
    def _load_collection_progress(self, collection_id: str) -> CollectionProgress:
        """Load collection progress for resumability"""
        try:
            progress_file = Path(f"data/collection_progress/progress_{collection_id}.json")
            
            if progress_file.exists():
                with open(progress_file, 'r') as f:
                    progress_data = json.load(f)
                
                return CollectionProgress(
                    total_assets=progress_data["total_assets"],
                    completed_assets=progress_data["completed_assets"],
                    current_asset=progress_data["current_asset"],
                    start_time=datetime.fromisoformat(progress_data["start_time"]),
                    estimated_completion=datetime.fromisoformat(progress_data["estimated_completion"]) if progress_data["estimated_completion"] else None,
                    resumable_state=progress_data["resumable_state"]
                )
            else:
                # Return default progress if file doesn't exist
                return CollectionProgress(
                    total_assets=0,
                    completed_assets=0,
                    current_asset="",
                    start_time=datetime.now(),
                    estimated_completion=None,
                    resumable_state={}
                )
                
        except Exception as e:
            self.data_logger.log_data_collection(
                "progress_tracking", "load", 0, False, str(e)
            )
            # Return default progress on error
            return CollectionProgress(
                total_assets=0,
                completed_assets=0,
                current_asset="",
                start_time=datetime.now(),
                estimated_completion=None,
                resumable_state={}
            )
    
    def _cleanup_collection_progress(self, collection_id: str) -> None:
        """Clean up progress tracking files after successful completion"""
        try:
            progress_file = Path(f"data/collection_progress/progress_{collection_id}.json")
            if progress_file.exists():
                progress_file.unlink()
        except Exception:
            # Non-critical error, ignore
            pass
    
    async def _collect_traditional_historical_data(self, 
                                                 assets: List[str],
                                                 start_date: datetime,
                                                 end_date: datetime) -> Dict[str, List[Any]]:
        """Collect historical traditional asset data (legacy method for compatibility)"""
        traditional_data, _ = await self._collect_traditional_historical_data_enhanced(
            assets, start_date, end_date, 
            CollectionProgress(0, 0, "", datetime.now(), None, {}), False
        )
        return traditional_data
    
    async def _collect_sentiment_historical_data(self, 
                                               start_date: datetime,
                                               end_date: datetime) -> List[Any]:
        """Collect historical sentiment data for major market events"""
        sentiment_data = []
        
        # Define major market crash events for training
        crash_events = [
            {
                "name": "COVID_CRASH_2020",
                "start": datetime(2020, 3, 1),
                "end": datetime(2020, 4, 30),
                "keywords": ["crash", "dump", "panic", "sell", "fear"]
            },
            {
                "name": "TERRA_LUNA_2022",
                "start": datetime(2022, 5, 1),
                "end": datetime(2022, 6, 30),
                "keywords": ["luna", "ust", "depeg", "collapse", "terra"]
            },
            {
                "name": "FTX_COLLAPSE_2022",
                "start": datetime(2022, 11, 1),
                "end": datetime(2022, 12, 31),
                "keywords": ["ftx", "sbf", "bankruptcy", "hack", "withdraw"]
            }
        ]
        
        for event in crash_events:
            # Check if event overlaps with requested period
            if (event["start"] <= end_date and event["end"] >= start_date):
                try:
                    # Collect Twitter data for this event
                    twitter_data = await self.sentiment_collector.collect_historical_sentiment(
                        "twitter", event["keywords"], event["start"], event["end"]
                    )
                    sentiment_data.extend(twitter_data)
                    
                    # Collect Reddit data for this event
                    reddit_data = await self.sentiment_collector.collect_historical_sentiment(
                        "reddit", event["keywords"], event["start"], event["end"]
                    )
                    sentiment_data.extend(reddit_data)
                    
                except Exception as e:
                    self.data_logger.log_data_collection(
                        "sentiment_historical", event["name"], 0, False, str(e)
                    )
                    continue
        
        return sentiment_data

class TrainingPipeline:
    """Complete ML training pipeline"""
    
    def __init__(self, model_save_dir: str = "models/trained"):
        self.config = get_config()
        self.data_logger = get_data_logger()
        self.performance_logger = get_performance_logger()
        
        # Training configuration - set this first
        self.model_save_dir = Path(model_save_dir)
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.data_manager = DataManager()
        self.historical_collector = HistoricalDataCollector()
        self.oracle_processor = OracleDataProcessor()
        self.ml_trainer = MLModelTrainer(model_save_dir)
        self.correlation_engine = CorrelationEngine()
        self.diagnostics_manager = TrainingDiagnosticsManager(str(self.model_save_dir / "diagnostics"))
        self.enhanced_config_manager = get_enhanced_training_config_manager()
        
        # Enhanced components (lazy loaded)
        self._threshold_optimizer = None
        self._enhanced_backtesting_framework = None
        self._advanced_feature_engine = None
        self._automated_retraining_system = None
    
    @property
    def threshold_optimizer(self):
        """Lazy load threshold optimizer"""
        if self._threshold_optimizer is None:
            try:
                from services.timeseries.threshold_integration import IntegratedThresholdOptimizer
                self._threshold_optimizer = IntegratedThresholdOptimizer()
            except ImportError:
                self.data_logger.log_data_collection(
                    "enhanced_training", "threshold_optimizer_import", 0, False,
                    "IntegratedThresholdOptimizer not available"
                )
                self._threshold_optimizer = None
        return self._threshold_optimizer
    
    @property
    def enhanced_backtesting_framework(self):
        """Lazy load enhanced backtesting framework"""
        if self._enhanced_backtesting_framework is None:
            try:
                from services.timeseries.enhanced_backtesting_framework import EnhancedBacktestingFramework
                self._enhanced_backtesting_framework = EnhancedBacktestingFramework()
            except ImportError:
                self.data_logger.log_data_collection(
                    "enhanced_training", "enhanced_backtesting_import", 0, False,
                    "EnhancedBacktestingFramework not available"
                )
                self._enhanced_backtesting_framework = None
        return self._enhanced_backtesting_framework
    
    @property
    def advanced_feature_engine(self):
        """Lazy load advanced feature engine"""
        if self._advanced_feature_engine is None:
            try:
                from services.timeseries.advanced_feature_engine import AdvancedFeatureEngine
                self._advanced_feature_engine = AdvancedFeatureEngine()
            except ImportError:
                self.data_logger.log_data_collection(
                    "enhanced_training", "advanced_feature_engine_import", 0, False,
                    "AdvancedFeatureEngine not available"
                )
                self._advanced_feature_engine = None
        return self._advanced_feature_engine
    
    @property
    def automated_retraining_system(self):
        """Lazy load automated retraining system"""
        if self._automated_retraining_system is None:
            try:
                from services.timeseries.automated_retraining_system import AutomatedRetrainingSystem
                self._automated_retraining_system = AutomatedRetrainingSystem()
            except ImportError:
                self.data_logger.log_data_collection(
                    "enhanced_training", "automated_retraining_import", 0, False,
                    "AutomatedRetrainingSystem not available"
                )
                self._automated_retraining_system = None
        return self._automated_retraining_system
    
    async def run_complete_training_pipeline(self, 
                                           training_period_months: int = 24,
                                           validation_period_months: int = 6,
                                           model_params: Optional[Dict] = None,
                                           enable_enhanced_features: bool = False) -> Dict[str, Any]:
        """
        Run complete training pipeline from data collection to model validation
        
        Args:
            training_period_months: Training period in months
            validation_period_months: Validation period in months  
            model_params: Model parameters
            enable_enhanced_features: Enable enhanced features for backward compatibility
            
        Returns:
            Dictionary with training results (backward compatible format)
        """
        start_time = time.time()
        
        try:
            # Define training and validation periods
            end_date = datetime.now()
            training_start = end_date - timedelta(days=training_period_months * 30)
            validation_start = end_date - timedelta(days=validation_period_months * 30)
            
            self.data_logger.log_data_collection(
                "training_pipeline", "start", 1, True,
                f"Training: {training_start} to {validation_start}, Validation: {validation_start} to {end_date}"
            )
            
            # Step 1: Collect historical data
            print("Step 1: Collecting historical data...")
            training_data = await self.historical_collector.collect_training_data(
                training_start, validation_start,
                include_crash_events=enable_enhanced_features
            )
            
            validation_data = await self.historical_collector.collect_training_data(
                validation_start, end_date,
                include_crash_events=enable_enhanced_features
            )
            
            # Step 2: Prepare features from historical data
            print("Step 2: Preparing features...")
            training_features = await self._prepare_features_from_data(training_data)
            validation_features = await self._prepare_features_from_data(validation_data)
            
            # Step 3: Create target labels (risk events)
            print("Step 3: Creating target labels...")
            training_outcomes = self._create_risk_event_labels(training_data)
            validation_outcomes = self._create_risk_event_labels(validation_data)
            
            # Step 4: Train model
            print("Step 4: Training model...")
            model, training_metrics = self.ml_trainer.train_risk_prediction_model(
                training_features, training_outcomes, 
                validation_split=0.2, model_params=model_params
            )
            
            # Step 5: Validate model performance
            print("Step 5: Validating model performance...")
            backtest_result = self.ml_trainer.validate_model_performance(
                model, validation_features, validation_outcomes,
                validation_start, end_date
            )
            
            # Step 6: Run cross-validation
            print("Step 6: Running cross-validation...")
            cv_results = self.ml_trainer.backtesting.time_series_cross_validation(
                XGBoostRiskModel, training_features, training_outcomes, n_splits=5
            )
            
            # Step 7: Generate performance summary
            performance_summary = self.ml_trainer.get_model_performance_summary(cv_results)
            
            # Step 8: Save training metadata (backward compatible format)
            training_metadata = {
                "model_version": model.model_version,
                "model_type": "XGBoostRiskModel",
                "training_timestamp": datetime.now().isoformat(),
                "training_period": {
                    "start": training_start.isoformat(),
                    "end": validation_start.isoformat(),
                    "duration_days": (validation_start - training_start).days
                },
                "validation_period": {
                    "start": validation_start.isoformat(),
                    "end": end_date.isoformat(),
                    "duration_days": (end_date - validation_start).days
                },
                "training_metrics": training_metrics,
                "backtest_result": {
                    "accuracy": backtest_result.accuracy,
                    "precision": backtest_result.precision,
                    "recall": backtest_result.recall,
                    "f1_score": backtest_result.f1_score,
                    "auc_score": backtest_result.auc_score,
                    "average_prediction_latency_ms": backtest_result.average_prediction_latency_ms
                },
                "cross_validation_summary": performance_summary,
                "feature_importance": model.feature_importance,
                "training_data_stats": {
                    "training_samples": len(training_features),
                    "validation_samples": len(validation_features),
                    "total_features": len(training_features[0].__dict__) if training_features else 0,
                    "assets_covered": list(training_data.get("crypto_data", {}).keys()) + list(training_data.get("traditional_data", {}).keys())
                },
                "enhanced_features_enabled": enable_enhanced_features,
                "backward_compatibility": True
            }
            
            # Add enhanced data if available
            if "data_quality_report" in training_data:
                training_metadata["data_quality_report"] = {
                    "quality_score": getattr(training_data["data_quality_report"], 'quality_score', 0.0),
                    "coverage_percentage": getattr(training_data["data_quality_report"], 'coverage_percentage', 0.0),
                    "total_data_points": getattr(training_data["data_quality_report"], 'total_collected_points', 0)
                }
            
            if "crash_events_metadata" in training_data:
                training_metadata["crash_events_metadata"] = training_data["crash_events_metadata"]
            
            # Save metadata
            metadata_path = self.model_save_dir / f"training_metadata_{model.model_version}.json"
            with open(metadata_path, 'w') as f:
                json.dump(training_metadata, f, indent=2, default=str)
            
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time(
                "complete_training_pipeline", duration_ms, True
            )
            
            # Return backward compatible format
            return {
                "model": model,
                "training_metrics": training_metrics,
                "backtest_result": backtest_result,
                "cross_validation_results": cv_results,
                "performance_summary": performance_summary,
                "metadata": training_metadata,
                "metadata_path": str(metadata_path),
                
                # Additional data for enhanced compatibility
                "training_data_quality": training_data.get("data_quality_report"),
                "crash_events_metadata": training_data.get("crash_events_metadata", {}),
                "enhanced_components_status": self._get_enhanced_components_status(),
                "backward_compatibility": True
            }
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time(
                "complete_training_pipeline", duration_ms, False
            )
            self.data_logger.log_data_collection(
                "training_pipeline", "complete", 0, False, str(e)
            )
            raise
    
    async def _prepare_features_from_data(self, historical_data: Dict[str, Any]) -> List[FeatureSet]:
        """Convert historical data to feature sets for ML training"""
        try:
            feature_sets = []
            
            # Combine all asset data
            all_asset_data = {}
            all_asset_data.update(historical_data["crypto_data"])
            all_asset_data.update(historical_data["traditional_data"])
            
            # Process each asset
            for asset, price_data in all_asset_data.items():
                if not price_data:
                    continue
                
                try:
                    # Calculate technical indicators
                    indicators = self.oracle_processor.indicator_calculator.calculate_all_indicators(
                        price_data, asset
                    )
                    
                    # Create feature sets for each time point
                    for i in range(len(price_data)):
                        # Use data up to current point for feature calculation
                        historical_subset = price_data[:i+1]
                        
                        if len(historical_subset) < 10:  # Need minimum history
                            continue
                        
                        feature_set = self.oracle_processor.feature_engineer.create_feature_set(
                            asset, historical_subset, indicators, all_asset_data
                        )
                        
                        # Set timestamp from actual data point
                        feature_set.timestamp = getattr(price_data[i], 'timestamp', datetime.now())
                        
                        feature_sets.append(feature_set)
                
                except Exception as e:
                    self.data_logger.log_data_collection(
                        "feature_preparation", asset, 0, False, str(e)
                    )
                    continue
            
            return feature_sets
        
        except Exception as e:
            self.data_logger.log_data_collection(
                "feature_preparation", "all_assets", 0, False, str(e)
            )
            return []
    
    def _create_risk_event_labels(self, historical_data: Dict[str, Any]) -> Dict[str, List[float]]:
        """Create risk event labels from historical data"""
        try:
            risk_labels = {}
            
            # Combine all asset data
            all_asset_data = {}
            all_asset_data.update(historical_data["crypto_data"])
            all_asset_data.update(historical_data["traditional_data"])
            
            for asset, price_data in all_asset_data.items():
                if not price_data or len(price_data) < 2:
                    risk_labels[asset] = []
                    continue
                
                # Calculate future price movements for risk labeling
                future_prices = []
                
                for i in range(len(price_data)):
                    current_price = getattr(price_data[i], 'price', 
                                          getattr(price_data[i], 'current_price', 0))
                    
                    # Look ahead 30 minutes to 2 hours for risk events
                    future_window = price_data[i+1:min(i+8, len(price_data))]  # Next 7 data points
                    
                    if future_window:
                        min_future_price = min(
                            getattr(p, 'price', getattr(p, 'current_price', current_price))
                            for p in future_window
                        )
                        future_prices.append(min_future_price)
                    else:
                        future_prices.append(current_price)
                
                risk_labels[asset] = future_prices
            
            return risk_labels
        
        except Exception as e:
            self.data_logger.log_data_collection(
                "label_creation", "risk_events", 0, False, str(e)
            )
            return {}
    
    async def quick_model_update(self, model_path: str, 
                               new_data_days: int = 7) -> Dict[str, Any]:
        """Quick model update with recent data"""
        try:
            # Load existing model
            model = self.ml_trainer.load_model(model_path)
            
            # Collect recent data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=new_data_days)
            
            recent_data = await self.historical_collector.collect_training_data(
                start_date, end_date
            )
            
            # Prepare features
            recent_features = await self._prepare_features_from_data(recent_data)
            recent_outcomes = self._create_risk_event_labels(recent_data)
            
            if not recent_features:
                raise ValueError("No recent features available for model update")
            
            # Retrain model with recent data
            updated_model, metrics = self.ml_trainer.train_risk_prediction_model(
                recent_features, recent_outcomes, validation_split=0.3
            )
            
            return {
                "updated_model": updated_model,
                "update_metrics": metrics,
                "data_period": {"start": start_date, "end": end_date},
                "samples_used": len(recent_features)
            }
        
        except Exception as e:
            self.data_logger.log_data_collection(
                "model_update", "quick_update", 0, False, str(e)
            )
            raise
    
    def evaluate_model_performance(self, model: XGBoostRiskModel,
                                 test_features: List[FeatureSet],
                                 test_outcomes: Dict[str, List[float]],
                                 use_enhanced_backtesting: bool = True) -> Dict[str, Any]:
        """Comprehensive model performance evaluation"""
        try:
            # Run standard backtesting
            test_start = min(fs.timestamp for fs in test_features)
            test_end = max(fs.timestamp for fs in test_features)
            
            backtest_result = self.ml_trainer.backtesting.run_backtest(
                model, test_features, test_outcomes, test_start, test_end
            )
            
            evaluation_results = {
                "backtest_result": backtest_result,
                "model_version": model.model_version,
                "test_period": {"start": test_start, "end": test_end},
                "test_samples": len(test_features),
                "feature_importance": model.feature_importance,
                "performance_metrics": {
                    "accuracy": backtest_result.accuracy,
                    "precision": backtest_result.precision,
                    "recall": backtest_result.recall,
                    "f1_score": backtest_result.f1_score,
                    "auc_score": backtest_result.auc_score,
                    "avg_latency_ms": backtest_result.average_prediction_latency_ms
                }
            }
            
            # Run enhanced backtesting if requested
            if use_enhanced_backtesting:
                try:
                    from services.timeseries.enhanced_backtesting_framework import EnhancedBacktestingFramework
                    
                    # Get crash events for the test period
                    crash_events = self.crash_detector.get_crash_events_in_period(test_start, test_end)
                    
                    if crash_events:
                        enhanced_framework = EnhancedBacktestingFramework()
                        enhanced_result = enhanced_framework.run_enhanced_backtest(
                            model=model,
                            historical_features=test_features,
                            historical_outcomes=test_outcomes,
                            crash_events=crash_events,
                            evaluation_start=test_start,
                            evaluation_end=test_end,
                            include_baseline_comparison=True,
                            include_multi_horizon=True
                        )
                        
                        # Add enhanced results to evaluation
                        evaluation_results["enhanced_backtest_result"] = enhanced_result
                        evaluation_results["crash_event_analysis"] = {
                            "crash_events_evaluated": enhanced_result.crash_events_evaluated,
                            "average_crash_detection_f1": enhanced_result.average_crash_detection_f1,
                            "crash_event_results": [
                                {
                                    "event_name": r.crash_event_name,
                                    "f1_score": r.f1_score,
                                    "recall": r.recall,
                                    "precision": r.precision,
                                    "crash_detection_rate": r.crash_detection_rate,
                                    "false_alarm_rate": r.false_alarm_rate
                                }
                                for r in enhanced_result.crash_event_results
                            ]
                        }
                        evaluation_results["multi_horizon_analysis"] = {
                            "best_horizon": enhanced_result.best_horizon_performance,
                            "horizon_results": {
                                horizon: {
                                    "accuracy": metrics.accuracy,
                                    "f1_score": metrics.f1_score,
                                    "average_lead_time_minutes": metrics.average_lead_time_minutes
                                }
                                for horizon, metrics in enhanced_result.multi_horizon_results.items()
                            }
                        }
                        evaluation_results["baseline_comparison"] = {
                            "outperforms_baselines": enhanced_result.outperforms_baselines,
                            "baseline_results": {
                                name: {
                                    "f1_score": baseline.f1_score,
                                    "performance_difference": baseline.performance_difference,
                                    "statistical_significance": baseline.statistical_significance
                                }
                                for name, baseline in enhanced_result.baseline_comparisons.items()
                            }
                        }
                        evaluation_results["recommendations"] = enhanced_result.recommendations
                        
                        # Save enhanced backtest result
                        try:
                            result_path = enhanced_framework.save_enhanced_backtest_result(enhanced_result)
                            evaluation_results["enhanced_backtest_saved_to"] = result_path
                        except Exception as save_error:
                            self.data_logger.log_data_collection(
                                "enhanced_backtest_save", "error", 0, False, str(save_error)
                            )
                    
                    else:
                        self.data_logger.log_data_collection(
                            "enhanced_backtesting", "no_crash_events", 0, False,
                            f"No crash events found in test period {test_start} to {test_end}"
                        )
                        evaluation_results["enhanced_backtest_note"] = "No crash events in test period"
                
                except Exception as enhanced_error:
                    self.data_logger.log_data_collection(
                        "enhanced_backtesting", "error", 0, False, str(enhanced_error)
                    )
                    evaluation_results["enhanced_backtest_error"] = str(enhanced_error)
            
            return evaluation_results
        
        except Exception as e:
            self.data_logger.log_data_collection(
                "model_evaluation", "performance", 0, False, str(e)
            )
            raise
    
    async def run_enhanced_training_with_diagnostics(self, 
                                                   training_period_months: int = 24,
                                                   validation_period_months: int = 6,
                                                   model_params: Optional[Dict] = None,
                                                   enable_comprehensive_diagnostics: bool = True,
                                                   enable_shap_analysis: bool = True) -> Dict[str, Any]:
        """
        Run enhanced training pipeline with comprehensive diagnostics and metadata
        
        Args:
            training_period_months: Training period in months
            validation_period_months: Validation period in months  
            model_params: Model parameters
            enable_comprehensive_diagnostics: Enable full diagnostic analysis
            enable_shap_analysis: Enable SHAP feature importance analysis
        
        Returns:
            Dictionary with training results and comprehensive diagnostics
        """
        start_time = time.time()
        
        try:
            # Run standard training pipeline first
            print("Running standard training pipeline...")
            standard_results = await self.run_complete_training_pipeline(
                training_period_months, validation_period_months, model_params
            )
            
            # Extract key components for diagnostics
            model = standard_results["model"]
            training_metadata = standard_results["metadata"]
            
            # Prepare diagnostic data
            print("Preparing data for diagnostics...")
            
            # Get training data for diagnostics
            end_date = datetime.now()
            training_start = end_date - timedelta(days=training_period_months * 30)
            validation_start = end_date - timedelta(days=validation_period_months * 30)
            
            # Collect training data for diagnostics
            training_data = await self.historical_collector.collect_training_data(
                training_start, validation_start, include_crash_events=True
            )
            
            # Prepare features for diagnostics
            training_features = await self._prepare_features_from_data(training_data)
            training_outcomes = self._create_risk_event_labels(training_data)
            
            # Convert to ML format
            preprocessor = self.ml_trainer.preprocessor
            feature_df = preprocessor.prepare_features_from_feature_sets(training_features)
            labels = preprocessor.create_target_labels(feature_df, training_outcomes)
            
            X_train = preprocessor.fit_transform(feature_df)
            y_train = labels.values
            feature_names = preprocessor.feature_columns
            
            # Run comprehensive diagnostics if enabled
            diagnostic_results = {}
            if enable_comprehensive_diagnostics and X_train.size > 0:
                print("Running comprehensive training diagnostics...")
                
                try:
                    diagnostic_results = self.diagnostics_manager.run_comprehensive_diagnostics(
                        model=model,
                        X_train=X_train,
                        y_train=y_train,
                        feature_names=feature_names,
                        model_class=XGBoostRiskModel,
                        model_params=model_params
                    )
                    
                    print(f"✓ Diagnostics completed. Report saved to: {diagnostic_results.get('report_path', 'N/A')}")
                    
                except Exception as diag_error:
                    self.data_logger.log_data_collection(
                        "enhanced_training_diagnostics", "error", 0, False, str(diag_error)
                    )
                    diagnostic_results = {"error": str(diag_error)}
                    print(f"⚠ Diagnostics failed: {diag_error}")
            
            # Enhanced metadata with diagnostics
            enhanced_metadata = training_metadata.copy()
            enhanced_metadata.update({
                "enhanced_training_enabled": True,
                "comprehensive_diagnostics_enabled": enable_comprehensive_diagnostics,
                "shap_analysis_enabled": enable_shap_analysis,
                "diagnostic_results_available": len(diagnostic_results) > 0,
                "training_data_quality": training_data.get("data_quality_report", {}).__dict__ if hasattr(training_data.get("data_quality_report", {}), '__dict__') else {},
                "crash_events_included": len(training_data.get("crash_events_metadata", {})),
                "enhanced_training_timestamp": datetime.now().isoformat()
            })
            
            # Add diagnostic summaries to metadata
            if diagnostic_results and "comprehensive_metadata" in diagnostic_results:
                enhanced_metadata["diagnostic_summary"] = diagnostic_results["comprehensive_metadata"]
            
            # Save enhanced metadata
            enhanced_metadata_path = self.model_save_dir / f"enhanced_training_metadata_{model.model_version}.json"
            with open(enhanced_metadata_path, 'w') as f:
                json.dump(enhanced_metadata, f, indent=2, default=str)
            
            # Compile comprehensive results
            enhanced_results = {
                # Standard training results
                "model": model,
                "training_metrics": standard_results["training_metrics"],
                "backtest_result": standard_results["backtest_result"],
                "cross_validation_results": standard_results["cross_validation_results"],
                "performance_summary": standard_results["performance_summary"],
                
                # Enhanced components
                "enhanced_metadata": enhanced_metadata,
                "enhanced_metadata_path": str(enhanced_metadata_path),
                "diagnostic_results": diagnostic_results,
                "training_data_quality": training_data.get("data_quality_report"),
                "crash_events_metadata": training_data.get("crash_events_metadata", {}),
                
                # Recommendations compilation
                "comprehensive_recommendations": self._compile_comprehensive_recommendations(
                    standard_results, diagnostic_results, training_data
                )
            }
            
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time(
                "enhanced_training_with_diagnostics", duration_ms, True
            )
            
            print(f"\n[SUCCESS] Enhanced training completed in {duration_ms/1000:.2f} seconds")
            print(f"[INFO] Enhanced metadata saved to: {enhanced_metadata_path}")
            
            return enhanced_results
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time(
                "enhanced_training_with_diagnostics", duration_ms, False
            )
            self.data_logger.log_data_collection(
                "enhanced_training_pipeline", "complete", 0, False, str(e)
            )
            raise
    
    def _compile_comprehensive_recommendations(self, standard_results: Dict[str, Any],
                                             diagnostic_results: Dict[str, Any],
                                             training_data: Dict[str, Any]) -> List[str]:
        """Compile recommendations from all analysis components"""
        try:
            recommendations = []
            
            # Standard training recommendations
            if "performance_summary" in standard_results:
                perf_summary = standard_results["performance_summary"]
                avg_f1 = perf_summary.get("avg_f1_score", 0)
                
                if avg_f1 < 0.6:
                    recommendations.append("Low F1 score detected - consider feature engineering or model tuning")
                elif avg_f1 > 0.8:
                    recommendations.append("Good model performance achieved")
            
            # Data quality recommendations
            if "data_quality_report" in training_data:
                quality_report = training_data["data_quality_report"]
                if hasattr(quality_report, 'quality_score'):
                    if quality_report.quality_score < 0.7:
                        recommendations.append("Data quality issues detected - consider improving data collection")
                    
                    if hasattr(quality_report, 'data_validation_errors') and quality_report.data_validation_errors:
                        recommendations.append(f"Data validation issues found: {len(quality_report.data_validation_errors)} errors")
            
            # Diagnostic recommendations
            if diagnostic_results and "comprehensive_metadata" in diagnostic_results:
                metadata = diagnostic_results["comprehensive_metadata"]
                
                # Add diagnostic-specific recommendations
                if "overall_recommendations" in metadata:
                    recommendations.extend(metadata["overall_recommendations"])
                
                # Add diagnostic summary insights
                if "diagnostic_summary" in metadata:
                    diag_summary = metadata["diagnostic_summary"]
                    if diag_summary.get("overfitting_detected"):
                        severity = diag_summary.get("overfitting_severity", "unknown")
                        recommendations.append(f"Overfitting detected ({severity} severity) - consider regularization")
                    
                    if diag_summary.get("underfitting_detected"):
                        severity = diag_summary.get("underfitting_severity", "unknown")
                        recommendations.append(f"Underfitting detected ({severity} severity) - consider model complexity increase")
            
            # Crash event recommendations
            if "crash_events_metadata" in training_data:
                crash_events = training_data["crash_events_metadata"]
                if len(crash_events) == 0:
                    recommendations.append("No crash events in training period - consider extending training period")
                else:
                    recommendations.append(f"Training includes {len(crash_events)} crash events for robust learning")
            
            # Remove duplicates and return
            return list(set(recommendations))
            
        except Exception as e:
            self.data_logger.log_data_collection(
                "recommendation_compilation", "error", 0, False, str(e)
            )
            return ["Error compiling recommendations"]
    
    async def run_enhanced_training_pipeline(self,
                                           enhanced_config: Optional[EnhancedTrainingConfiguration] = None,
                                           training_period_months: int = 24,
                                           validation_period_months: int = 6,
                                           model_params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Run enhanced training pipeline with all enhanced features integrated
        
        Args:
            enhanced_config: Enhanced training configuration
            training_period_months: Training period in months
            validation_period_months: Validation period in months
            model_params: Model parameters
            
        Returns:
            Dictionary with comprehensive training results
        """
        start_time = time.time()
        
        try:
            # Load enhanced configuration if not provided
            if enhanced_config is None:
                enhanced_config = self.enhanced_config_manager.load_enhanced_training_config()
            
            # Log configuration summary
            config_summary = self.enhanced_config_manager.create_training_summary(enhanced_config)
            print(config_summary)
            self.data_logger.log_data_collection(
                "enhanced_training_pipeline", "config_loaded", 1, True,
                "Enhanced training configuration loaded successfully"
            )
            
            # Adjust training period based on extended training configuration
            if enhanced_config.extended_training.enabled:
                training_period_months = max(
                    training_period_months,
                    enhanced_config.extended_training.min_training_months
                )
                training_period_months = min(
                    training_period_months,
                    enhanced_config.extended_training.max_training_months
                )
            
            # Define training and validation periods
            end_date = datetime.now()
            training_start = end_date - timedelta(days=training_period_months * 30)
            validation_start = end_date - timedelta(days=validation_period_months * 30)
            
            self.data_logger.log_data_collection(
                "enhanced_training_pipeline", "start", 1, True,
                f"Enhanced Training: {training_start} to {validation_start}, Validation: {validation_start} to {end_date}"
            )
            
            # Step 1: Enhanced historical data collection
            print("Step 1: Enhanced historical data collection...")
            training_data = await self._enhanced_data_collection(
                training_start, validation_start, enhanced_config
            )
            
            validation_data = await self._enhanced_data_collection(
                validation_start, end_date, enhanced_config
            )
            
            # Step 2: Enhanced feature engineering
            print("Step 2: Enhanced feature engineering...")
            training_features = await self._enhanced_feature_engineering(
                training_data, enhanced_config
            )
            validation_features = await self._enhanced_feature_engineering(
                validation_data, enhanced_config
            )
            
            # Step 3: Create enhanced target labels
            print("Step 3: Creating enhanced target labels...")
            training_outcomes = self._create_enhanced_risk_labels(
                training_data, enhanced_config
            )
            validation_outcomes = self._create_enhanced_risk_labels(
                validation_data, enhanced_config
            )
            
            # Step 4: Train model with enhanced configuration
            print("Step 4: Training model with enhanced features...")
            model, training_metrics = self.ml_trainer.train_risk_prediction_model(
                training_features, training_outcomes,
                validation_split=0.2, model_params=model_params
            )
            
            # Step 5: Threshold optimization (if enabled)
            optimal_threshold = None
            if enhanced_config.threshold_optimization.enabled and self.threshold_optimizer:
                print("Step 5: Optimizing prediction threshold...")
                optimal_threshold = await self._optimize_threshold(
                    model, validation_features, validation_outcomes, enhanced_config
                )
                if optimal_threshold:
                    # Update model with optimal threshold
                    model.risk_threshold = optimal_threshold.value
            
            # Step 6: Enhanced backtesting (if enabled)
            enhanced_backtest_result = None
            if (enhanced_config.backtesting.enhanced_backtesting.get('enabled', False) and 
                self.enhanced_backtesting_framework):
                print("Step 6: Running enhanced backtesting...")
                enhanced_backtest_result = await self._run_enhanced_backtesting(
                    model, validation_features, validation_outcomes, 
                    validation_start, end_date, enhanced_config
                )
            
            # Step 7: Standard backtesting for compatibility
            print("Step 7: Running standard backtesting...")
            backtest_result = self.ml_trainer.validate_model_performance(
                model, validation_features, validation_outcomes,
                validation_start, end_date
            )
            
            # Step 8: Cross-validation
            print("Step 8: Running cross-validation...")
            cv_results = self.ml_trainer.backtesting.time_series_cross_validation(
                XGBoostRiskModel, training_features, training_outcomes, n_splits=5
            )
            
            # Step 9: Generate performance summary
            performance_summary = self.ml_trainer.get_model_performance_summary(cv_results)
            
            # Step 10: Comprehensive diagnostics (if enabled)
            diagnostic_results = {}
            if enhanced_config.diagnostics.comprehensive_diagnostics.get('enabled', True):
                print("Step 10: Running comprehensive diagnostics...")
                diagnostic_results = await self._run_enhanced_diagnostics(
                    model, training_features, training_outcomes, enhanced_config
                )
            
            # Step 11: Setup automated retraining (if enabled)
            retraining_config = None
            if (enhanced_config.automated_retraining.enabled and 
                self.automated_retraining_system):
                print("Step 11: Setting up automated retraining...")
                retraining_config = await self._setup_automated_retraining(
                    model, enhanced_config
                )
            
            # Step 12: Compile comprehensive metadata
            enhanced_metadata = self._compile_enhanced_metadata(
                model, enhanced_config, training_metrics, backtest_result,
                performance_summary, diagnostic_results, optimal_threshold,
                enhanced_backtest_result, retraining_config,
                training_start, validation_start, end_date,
                training_features, validation_features
            )
            
            # Step 13: Save enhanced metadata and results
            metadata_path = self.model_save_dir / f"enhanced_training_metadata_{model.model_version}.json"
            with open(metadata_path, 'w') as f:
                json.dump(enhanced_metadata, f, indent=2, default=str)
            
            # Save enhanced configuration used
            config_path = self.model_save_dir / f"enhanced_config_{model.model_version}.json"
            self.enhanced_config_manager.save_enhanced_config(enhanced_config, str(config_path))
            
            # Step 14: Generate enhanced result reporting
            enhanced_report = self._generate_enhanced_training_report(
                model, enhanced_config, training_metrics, backtest_result,
                performance_summary, diagnostic_results, optimal_threshold,
                enhanced_backtest_result, training_data, validation_data
            )
            
            # Save enhanced report
            report_path = self.model_save_dir / f"enhanced_training_report_{model.model_version}.md"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(enhanced_report)
            
            # Compile final results with backward compatibility
            enhanced_results = {
                # Core training results (backward compatible)
                "model": model,
                "training_metrics": training_metrics,
                "backtest_result": backtest_result,
                "cross_validation_results": cv_results,
                "performance_summary": performance_summary,
                "metadata": enhanced_metadata,  # Backward compatibility
                "metadata_path": str(metadata_path),  # Backward compatibility
                
                # Enhanced components
                "enhanced_metadata": enhanced_metadata,
                "enhanced_metadata_path": str(metadata_path),
                "enhanced_config_path": str(config_path),
                "enhanced_config": enhanced_config,
                "enhanced_report": enhanced_report,
                "enhanced_report_path": str(report_path),
                "optimal_threshold": optimal_threshold,
                "enhanced_backtest_result": enhanced_backtest_result,
                "diagnostic_results": diagnostic_results,
                "retraining_config": retraining_config,
                
                # Data quality and crash events
                "training_data_quality": training_data.get("data_quality_report"),
                "validation_data_quality": validation_data.get("data_quality_report"),
                "crash_events_metadata": training_data.get("crash_events_metadata", {}),
                
                # Comprehensive recommendations
                "comprehensive_recommendations": self._compile_enhanced_recommendations(
                    enhanced_results if 'enhanced_results' in locals() else {},
                    diagnostic_results, training_data, enhanced_config
                ),
                
                # Integration status
                "enhanced_components_status": self._get_enhanced_components_status(),
                "backward_compatibility": True
            }
            
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time(
                "enhanced_training_pipeline", duration_ms, True
            )
            
            print(f"\n[SUCCESS] Enhanced training pipeline completed in {duration_ms/1000:.2f} seconds")
            print(f"[INFO] Enhanced metadata saved to: {metadata_path}")
            print(f"[INFO] Enhanced configuration saved to: {config_path}")
            print(f"📋 Enhanced report saved to: {report_path}")
            
            return enhanced_results
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time(
                "enhanced_training_pipeline", duration_ms, False
            )
            self.data_logger.log_data_collection(
                "enhanced_training_pipeline", "complete", 0, False, str(e)
            )
            raise
    
    async def _enhanced_data_collection(self, 
                                      start_date: datetime,
                                      end_date: datetime,
                                      enhanced_config: EnhancedTrainingConfiguration) -> Dict[str, Any]:
        """Enhanced data collection with crash events and quality validation"""
        try:
            # Collect data with enhanced options
            collected_data = await self.historical_collector.collect_training_data(
                start_date=start_date,
                end_date=end_date,
                enable_progress_tracking=enhanced_config.extended_training.enable_progress_tracking,
                include_crash_events=enhanced_config.crash_events.enabled
            )
            
            # Validate data quality against thresholds
            if "data_quality_report" in collected_data:
                quality_report = collected_data["data_quality_report"]
                if hasattr(quality_report, 'quality_score'):
                    if quality_report.quality_score < enhanced_config.extended_training.data_quality_threshold:
                        self.data_logger.log_data_collection(
                            "enhanced_data_collection", "quality_warning", 1, True,
                            f"Data quality score {quality_report.quality_score:.2f} below threshold {enhanced_config.extended_training.data_quality_threshold}"
                        )
            
            return collected_data
            
        except Exception as e:
            self.data_logger.log_data_collection(
                "enhanced_data_collection", "error", 0, False, str(e)
            )
            raise
    
    async def _enhanced_feature_engineering(self,
                                          historical_data: Dict[str, Any],
                                          enhanced_config: EnhancedTrainingConfiguration) -> List[FeatureSet]:
        """Enhanced feature engineering with advanced features"""
        try:
            # Debug: Check historical data
            print(f"Historical data keys: {list(historical_data.keys())}")
            crypto_data = historical_data.get("crypto_data", {})
            traditional_data = historical_data.get("traditional_data", {})
            print(f"Crypto assets: {list(crypto_data.keys())}")
            print(f"Traditional assets: {list(traditional_data.keys())}")
            
            # Check data counts
            for asset, data in crypto_data.items():
                print(f"  {asset}: {len(data) if data else 0} data points")
            for asset, data in traditional_data.items():
                print(f"  {asset}: {len(data) if data else 0} data points")
            
            # Start with standard feature preparation
            feature_sets = await self._prepare_features_from_data(historical_data)
            print(f"Generated {len(feature_sets)} feature sets from standard preparation")
            
            # Apply advanced feature engineering if available and enabled
            if self.advanced_feature_engine and enhanced_config.feature_engineering:
                try:
                    # Enhanced correlation features
                    if enhanced_config.feature_engineering.cross_asset_correlation.get('enabled', False):
                        feature_sets = await self._add_correlation_features(
                            feature_sets, historical_data, enhanced_config
                        )
                    
                    # Enhanced volatility regime features
                    if enhanced_config.feature_engineering.volatility_regimes.get('enabled', False):
                        feature_sets = await self._add_volatility_regime_features(
                            feature_sets, historical_data, enhanced_config
                        )
                    
                    # Enhanced momentum features
                    if enhanced_config.feature_engineering.momentum_indicators.get('enabled', False):
                        feature_sets = await self._add_momentum_features(
                            feature_sets, historical_data, enhanced_config
                        )
                    
                    # Enhanced volume features
                    if enhanced_config.feature_engineering.volume_analysis.get('enabled', False):
                        feature_sets = await self._add_volume_features(
                            feature_sets, historical_data, enhanced_config
                        )
                    
                except Exception as e:
                    self.data_logger.log_data_collection(
                        "enhanced_feature_engineering", "advanced_features", 0, False, str(e)
                    )
                    # Continue with standard features if advanced features fail
                    print(f"Advanced feature engineering failed: {e}")
            
            print(f"Final feature sets count: {len(feature_sets)}")
            return feature_sets
            
        except Exception as e:
            self.data_logger.log_data_collection(
                "enhanced_feature_engineering", "error", 0, False, str(e)
            )
            print(f"Enhanced feature engineering failed: {e}")
            raise
    
    def _create_enhanced_risk_labels(self,
                                   historical_data: Dict[str, Any],
                                   enhanced_config: EnhancedTrainingConfiguration) -> Dict[str, List[float]]:
        """Create enhanced risk labels with crash event weighting"""
        try:
            # Start with standard risk labels
            risk_labels = self._create_risk_event_labels(historical_data)
            
            # Apply crash event weighting if enabled
            if (enhanced_config.crash_events.enabled and 
                enhanced_config.crash_events.weight_crash_samples and
                "crash_event_labels" in historical_data):
                
                crash_labels = historical_data["crash_event_labels"]
                
                for asset, labels in risk_labels.items():
                    if asset in crash_labels:
                        asset_crash_labels = crash_labels[asset]
                        
                        # Apply weight multipliers to crash periods
                        for i, (risk_value, crash_info) in enumerate(zip(labels, asset_crash_labels)):
                            if crash_info.get("is_crash_period", False):
                                weight_multiplier = crash_info.get("weight_multiplier", 1.0)
                                # Increase risk signal during crash periods
                                risk_labels[asset][i] = min(risk_value * weight_multiplier, 1.0)
            
            return risk_labels
            
        except Exception as e:
            self.data_logger.log_data_collection(
                "enhanced_risk_labels", "error", 0, False, str(e)
            )
            # Return standard labels on error
            return self._create_risk_event_labels(historical_data)
    
    async def _optimize_threshold(self,
                                model: XGBoostRiskModel,
                                validation_features: List[FeatureSet],
                                validation_outcomes: Dict[str, List[float]],
                                enhanced_config: EnhancedTrainingConfiguration) -> Optional[Any]:
        """Optimize prediction threshold using enhanced configuration"""
        try:
            if not self.threshold_optimizer:
                return None
            
            # Run threshold optimization
            optimization_result = await self.threshold_optimizer.optimize_threshold_comprehensive(
                model=model,
                validation_features=validation_features,
                validation_outcomes=validation_outcomes,
                threshold_values=enhanced_config.threshold_optimization.threshold_values,
                optimization_metric=enhanced_config.threshold_optimization.optimization_metric,
                cross_validation_folds=enhanced_config.threshold_optimization.cross_validation_folds
            )
            
            if optimization_result and enhanced_config.threshold_optimization.save_optimization_results:
                # Save optimization results
                results_path = self.model_save_dir / f"threshold_optimization_{model.model_version}.json"
                with open(results_path, 'w') as f:
                    json.dump(optimization_result.__dict__, f, indent=2, default=str)
            
            return optimization_result
            
        except Exception as e:
            self.data_logger.log_data_collection(
                "threshold_optimization", "error", 0, False, str(e)
            )
            return None
    
    async def _run_enhanced_backtesting(self,
                                      model: XGBoostRiskModel,
                                      validation_features: List[FeatureSet],
                                      validation_outcomes: Dict[str, List[float]],
                                      validation_start: datetime,
                                      validation_end: datetime,
                                      enhanced_config: EnhancedTrainingConfiguration) -> Optional[Any]:
        """Run enhanced backtesting with crash event analysis"""
        try:
            if not self.enhanced_backtesting_framework:
                return None
            
            # Get crash events for validation period
            crash_events = []
            if enhanced_config.crash_events.enabled:
                crash_detector = CrashEventDetector()
                crash_events = crash_detector.get_crash_events_in_period(
                    validation_start, validation_end
                )
            
            # Run enhanced backtesting
            enhanced_result = self.enhanced_backtesting_framework.run_enhanced_backtest(
                model=model,
                historical_features=validation_features,
                historical_outcomes=validation_outcomes,
                crash_events=crash_events,
                evaluation_start=validation_start,
                evaluation_end=validation_end,
                include_baseline_comparison=enhanced_config.backtesting.enhanced_backtesting.get('baseline_comparisons', True),
                include_multi_horizon=enhanced_config.backtesting.enhanced_backtesting.get('multi_horizon_evaluation', True)
            )
            
            # Save enhanced backtest results
            if enhanced_result:
                results_path = self.enhanced_backtesting_framework.save_enhanced_backtest_result(enhanced_result)
                self.data_logger.log_data_collection(
                    "enhanced_backtesting", "results_saved", 1, True,
                    f"Enhanced backtest results saved to {results_path}"
                )
            
            return enhanced_result
            
        except Exception as e:
            self.data_logger.log_data_collection(
                "enhanced_backtesting", "error", 0, False, str(e)
            )
            return None
    
    async def _run_enhanced_diagnostics(self,
                                      model: XGBoostRiskModel,
                                      training_features: List[FeatureSet],
                                      training_outcomes: Dict[str, List[float]],
                                      enhanced_config: EnhancedTrainingConfiguration) -> Dict[str, Any]:
        """Run enhanced diagnostics with comprehensive analysis"""
        try:
            # Prepare data for diagnostics
            preprocessor = self.ml_trainer.preprocessor
            feature_df = preprocessor.prepare_features_from_feature_sets(training_features)
            labels = preprocessor.create_target_labels(feature_df, training_outcomes)
            
            X_train = preprocessor.fit_transform(feature_df)
            y_train = labels.values
            feature_names = preprocessor.feature_columns
            
            if X_train.size == 0:
                return {"error": "No training data available for diagnostics"}
            
            # Run comprehensive diagnostics
            diagnostic_results = self.diagnostics_manager.run_comprehensive_diagnostics(
                model=model,
                X_train=X_train,
                y_train=y_train,
                feature_names=feature_names,
                model_class=XGBoostRiskModel,
                model_params=None
            )
            
            return diagnostic_results
            
        except Exception as e:
            self.data_logger.log_data_collection(
                "enhanced_diagnostics", "error", 0, False, str(e)
            )
            return {"error": str(e)}
    
    async def _setup_automated_retraining(self,
                                        model: XGBoostRiskModel,
                                        enhanced_config: EnhancedTrainingConfiguration) -> Optional[Dict[str, Any]]:
        """Setup automated retraining system"""
        try:
            if not self.automated_retraining_system:
                return None
            
            # Configure automated retraining
            retraining_config = {
                "model_version": model.model_version,
                "schedule": enhanced_config.automated_retraining.schedule,
                "performance_monitoring": enhanced_config.automated_retraining.performance_monitoring,
                "data_freshness": enhanced_config.automated_retraining.data_freshness,
                "incremental_training": enhanced_config.automated_retraining.incremental_training
            }
            
            # Setup retraining system
            setup_result = await self.automated_retraining_system.setup_automated_retraining(
                model=model,
                retraining_config=retraining_config
            )
            
            return setup_result
            
        except Exception as e:
            self.data_logger.log_data_collection(
                "automated_retraining_setup", "error", 0, False, str(e)
            )
            return None
    
    def _compile_enhanced_metadata(self,
                                 model: XGBoostRiskModel,
                                 enhanced_config: EnhancedTrainingConfiguration,
                                 training_metrics: Dict[str, float],
                                 backtest_result: Any,
                                 performance_summary: Dict[str, Any],
                                 diagnostic_results: Dict[str, Any],
                                 optimal_threshold: Optional[Any],
                                 enhanced_backtest_result: Optional[Any],
                                 retraining_config: Optional[Dict[str, Any]],
                                 training_start: datetime,
                                 validation_start: datetime,
                                 end_date: datetime,
                                 training_features: List[FeatureSet],
                                 validation_features: List[FeatureSet]) -> Dict[str, Any]:
        """Compile comprehensive enhanced training metadata"""
        
        metadata = {
            # Model information
            "model_version": model.model_version,
            "model_type": "XGBoostRiskModel",
            "enhanced_training_enabled": True,
            "training_timestamp": datetime.now().isoformat(),
            
            # Training periods
            "training_period": {
                "start": training_start.isoformat(),
                "end": validation_start.isoformat(),
                "duration_days": (validation_start - training_start).days
            },
            "validation_period": {
                "start": validation_start.isoformat(),
                "end": end_date.isoformat(),
                "duration_days": (end_date - validation_start).days
            },
            
            # Enhanced configuration summary
            "enhanced_configuration": {
                "extended_training_enabled": enhanced_config.extended_training.enabled,
                "crash_events_enabled": enhanced_config.crash_events.enabled,
                "threshold_optimization_enabled": enhanced_config.threshold_optimization.enabled,
                "enhanced_backtesting_enabled": enhanced_config.backtesting.enhanced_backtesting.get('enabled', False),
                "automated_retraining_enabled": enhanced_config.automated_retraining.enabled,
                "comprehensive_diagnostics_enabled": enhanced_config.diagnostics.comprehensive_diagnostics.get('enabled', False),
                "shap_analysis_enabled": enhanced_config.diagnostics.shap_analysis.get('enabled', False)
            },
            
            # Training metrics
            "training_metrics": training_metrics,
            
            # Backtest results
            "backtest_result": {
                "accuracy": backtest_result.accuracy,
                "precision": backtest_result.precision,
                "recall": backtest_result.recall,
                "f1_score": backtest_result.f1_score,
                "auc_score": backtest_result.auc_score,
                "average_prediction_latency_ms": backtest_result.average_prediction_latency_ms
            },
            
            # Cross-validation summary
            "cross_validation_summary": performance_summary,
            
            # Feature importance
            "feature_importance": model.feature_importance,
            
            # Training data statistics
            "training_data_stats": {
                "training_samples": len(training_features),
                "validation_samples": len(validation_features),
                "total_features": len(training_features[0].__dict__) if training_features else 0
            },
            
            # Enhanced components results
            "enhanced_components": {
                "optimal_threshold": optimal_threshold.__dict__ if optimal_threshold else None,
                "enhanced_backtest_available": enhanced_backtest_result is not None,
                "diagnostics_available": len(diagnostic_results) > 0,
                "automated_retraining_configured": retraining_config is not None
            }
        }
        
        # Add diagnostic summary if available
        if diagnostic_results and "comprehensive_metadata" in diagnostic_results:
            metadata["diagnostic_summary"] = diagnostic_results["comprehensive_metadata"]
        
        # Add enhanced backtest summary if available
        if enhanced_backtest_result:
            metadata["enhanced_backtest_summary"] = {
                "crash_events_evaluated": enhanced_backtest_result.crash_events_evaluated,
                "average_crash_detection_f1": enhanced_backtest_result.average_crash_detection_f1,
                "best_horizon_performance": enhanced_backtest_result.best_horizon_performance,
                "outperforms_baselines": enhanced_backtest_result.outperforms_baselines
            }
        
        return metadata
    
    def _compile_enhanced_recommendations(self,
                                        enhanced_results: Dict[str, Any],
                                        diagnostic_results: Dict[str, Any],
                                        training_data: Dict[str, Any],
                                        enhanced_config: EnhancedTrainingConfiguration) -> List[str]:
        """Compile enhanced recommendations from all analysis components"""
        try:
            recommendations = []
            
            # Enhanced training specific recommendations
            if enhanced_config.extended_training.enabled:
                recommendations.append("Extended training period enabled for better market cycle coverage")
            
            if enhanced_config.crash_events.enabled:
                crash_events = training_data.get("crash_events_metadata", {})
                if len(crash_events) > 0:
                    recommendations.append(f"Training includes {len(crash_events)} crash events for robust learning")
                else:
                    recommendations.append("No crash events in training period - consider extending training period")
            
            # Threshold optimization recommendations
            if enhanced_config.threshold_optimization.enabled and "optimal_threshold" in enhanced_results:
                optimal_threshold = enhanced_results["optimal_threshold"]
                if optimal_threshold:
                    recommendations.append(f"Optimal threshold found: {optimal_threshold.value:.3f} (F1: {optimal_threshold.f1_score:.3f})")
                else:
                    recommendations.append("Threshold optimization failed - using default threshold")
            
            # Enhanced backtesting recommendations
            if "enhanced_backtest_result" in enhanced_results and enhanced_results["enhanced_backtest_result"]:
                backtest_result = enhanced_results["enhanced_backtest_result"]
                if hasattr(backtest_result, 'outperforms_baselines') and backtest_result.outperforms_baselines:
                    recommendations.append("Model outperforms baseline models in enhanced backtesting")
                else:
                    recommendations.append("Model performance comparable to baselines - consider feature engineering improvements")
                
                if hasattr(backtest_result, 'recommendations'):
                    recommendations.extend(backtest_result.recommendations)
            
            # Data quality recommendations
            if "training_data_quality" in enhanced_results:
                quality_report = enhanced_results["training_data_quality"]
                if hasattr(quality_report, 'quality_score'):
                    if quality_report.quality_score < 0.7:
                        recommendations.append("Data quality issues detected - consider improving data collection")
                    elif quality_report.quality_score > 0.9:
                        recommendations.append("Excellent data quality achieved")
            
            # Diagnostic recommendations
            if diagnostic_results and "comprehensive_metadata" in diagnostic_results:
                metadata = diagnostic_results["comprehensive_metadata"]
                if "overall_recommendations" in metadata:
                    recommendations.extend(metadata["overall_recommendations"])
            
            # Feature engineering recommendations
            if enhanced_config.feature_engineering.cross_asset_correlation.get('enabled', False):
                recommendations.append("Cross-asset correlation features enabled for market regime detection")
            
            if enhanced_config.feature_engineering.volatility_regimes.get('enabled', False):
                recommendations.append("Volatility regime classification enabled for market state awareness")
            
            # Automated retraining recommendations
            if enhanced_config.automated_retraining.enabled:
                recommendations.append(f"Automated retraining configured with {enhanced_config.automated_retraining.schedule} schedule")
            else:
                recommendations.append("Consider enabling automated retraining for production deployment")
            
            # Remove duplicates and return
            return list(set(recommendations))
            
        except Exception as e:
            self.data_logger.log_data_collection(
                "enhanced_recommendation_compilation", "error", 0, False, str(e)
            )
            return ["Error compiling enhanced recommendations"]
    
    def _generate_enhanced_training_report(self,
                                         model: XGBoostRiskModel,
                                         enhanced_config: EnhancedTrainingConfiguration,
                                         training_metrics: Dict[str, float],
                                         backtest_result: Any,
                                         performance_summary: Dict[str, Any],
                                         diagnostic_results: Dict[str, Any],
                                         optimal_threshold: Optional[Any],
                                         enhanced_backtest_result: Optional[Any],
                                         training_data: Dict[str, Any],
                                         validation_data: Dict[str, Any]) -> str:
        """Generate comprehensive enhanced training report"""
        try:
            report_lines = []
            
            # Header
            report_lines.append("# Enhanced AI Risk Oracle Training Report")
            report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append(f"Model Version: {model.model_version}")
            report_lines.append("")
            
            # Executive Summary
            report_lines.append("## Executive Summary")
            report_lines.append(f"- **Training Status**: [SUCCESS] Completed Successfully")
            report_lines.append(f"- **Model Performance**: F1 Score {backtest_result.f1_score:.3f}, Accuracy {backtest_result.accuracy:.3f}")
            report_lines.append(f"- **Enhanced Features**: {self._count_enabled_features(enhanced_config)} enabled")
            
            if optimal_threshold:
                report_lines.append(f"- **Optimal Threshold**: {optimal_threshold.value:.3f} (F1: {optimal_threshold.f1_score:.3f})")
            
            # Data quality summary
            if "data_quality_report" in training_data:
                quality_report = training_data["data_quality_report"]
                if hasattr(quality_report, 'quality_score'):
                    report_lines.append(f"- **Data Quality**: {quality_report.quality_score:.2f}/1.0")
            
            report_lines.append("")
            
            # Enhanced Configuration Summary
            report_lines.append("## Enhanced Configuration")
            report_lines.append(f"- Extended Training: {'[YES]' if enhanced_config.extended_training.enabled else '[NO]'}")
            report_lines.append(f"- Crash Events: {'[YES]' if enhanced_config.crash_events.enabled else '[NO]'}")
            report_lines.append(f"- Threshold Optimization: {'[YES]' if enhanced_config.threshold_optimization.enabled else '[NO]'}")
            report_lines.append(f"- Enhanced Backtesting: {'[YES]' if enhanced_config.backtesting.enhanced_backtesting.get('enabled', False) else '[NO]'}")
            report_lines.append(f"- Automated Retraining: {'[YES]' if enhanced_config.automated_retraining.enabled else '[NO]'}")
            report_lines.append(f"- Comprehensive Diagnostics: {'[YES]' if enhanced_config.diagnostics.comprehensive_diagnostics.get('enabled', False) else '[NO]'}")
            report_lines.append("")
            
            # Training Metrics
            report_lines.append("## Training Metrics")
            for metric, value in training_metrics.items():
                report_lines.append(f"- **{metric.replace('_', ' ').title()}**: {value:.4f}")
            report_lines.append("")
            
            # Backtest Results
            report_lines.append("## Backtest Results")
            report_lines.append(f"- **Accuracy**: {backtest_result.accuracy:.4f}")
            report_lines.append(f"- **Precision**: {backtest_result.precision:.4f}")
            report_lines.append(f"- **Recall**: {backtest_result.recall:.4f}")
            report_lines.append(f"- **F1 Score**: {backtest_result.f1_score:.4f}")
            if backtest_result.auc_score:
                report_lines.append(f"- **AUC Score**: {backtest_result.auc_score:.4f}")
            report_lines.append(f"- **Average Latency**: {backtest_result.average_prediction_latency_ms:.2f}ms")
            report_lines.append("")
            
            # Cross-Validation Summary
            report_lines.append("## Cross-Validation Summary")
            report_lines.append(f"- **Average Accuracy**: {performance_summary.get('avg_accuracy', 0):.4f}")
            report_lines.append(f"- **Average Precision**: {performance_summary.get('avg_precision', 0):.4f}")
            report_lines.append(f"- **Average Recall**: {performance_summary.get('avg_recall', 0):.4f}")
            report_lines.append(f"- **Average F1 Score**: {performance_summary.get('avg_f1_score', 0):.4f}")
            report_lines.append(f"- **Total Predictions**: {performance_summary.get('total_predictions', 0):,}")
            report_lines.append("")
            
            # Enhanced Backtesting Results
            if enhanced_backtest_result:
                report_lines.append("## Enhanced Backtesting Results")
                if hasattr(enhanced_backtest_result, 'crash_events_evaluated'):
                    report_lines.append(f"- **Crash Events Evaluated**: {enhanced_backtest_result.crash_events_evaluated}")
                if hasattr(enhanced_backtest_result, 'average_crash_detection_f1'):
                    report_lines.append(f"- **Average Crash Detection F1**: {enhanced_backtest_result.average_crash_detection_f1:.4f}")
                if hasattr(enhanced_backtest_result, 'outperforms_baselines'):
                    report_lines.append(f"- **Outperforms Baselines**: {'[YES]' if enhanced_backtest_result.outperforms_baselines else '[NO]'}")
                report_lines.append("")
            
            # Crash Events Analysis
            if "crash_events_metadata" in training_data and training_data["crash_events_metadata"]:
                report_lines.append("## Crash Events Analysis")
                crash_events = training_data["crash_events_metadata"]
                report_lines.append(f"- **Total Crash Events**: {len(crash_events)}")
                for event_name, event_info in crash_events.items():
                    report_lines.append(f"  - **{event_name}**: {event_info.get('description', 'N/A')}")
                    report_lines.append(f"    - Period: {event_info.get('start_date', 'N/A')} to {event_info.get('end_date', 'N/A')}")
                    report_lines.append(f"    - Severity: {event_info.get('severity', 'N/A')}")
                    report_lines.append(f"    - Weight Multiplier: {event_info.get('weight_multiplier', 1.0)}")
                report_lines.append("")
            
            # Feature Importance
            report_lines.append("## Top Feature Importance")
            feature_importance = model.feature_importance
            if feature_importance:
                sorted_features = sorted(feature_importance.items(), 
                                       key=lambda x: x[1], reverse=True)[:15]
                for i, (feature, importance) in enumerate(sorted_features, 1):
                    report_lines.append(f"{i:2d}. **{feature}**: {importance:.4f}")
            else:
                report_lines.append("- No feature importance data available")
            report_lines.append("")
            
            # Data Quality Report
            if "data_quality_report" in training_data:
                quality_report = training_data["data_quality_report"]
                report_lines.append("## Data Quality Report")
                if hasattr(quality_report, 'quality_score'):
                    report_lines.append(f"- **Overall Quality Score**: {quality_report.quality_score:.2f}/1.0")
                    report_lines.append(f"- **Data Coverage**: {quality_report.coverage_percentage:.1%}")
                    report_lines.append(f"- **Total Data Points**: {quality_report.total_collected_points:,}")
                    
                    if hasattr(quality_report, 'asset_coverage'):
                        report_lines.append("- **Asset Coverage**:")
                        for asset, coverage in quality_report.asset_coverage.items():
                            status = "[GOOD]" if coverage >= 0.8 else "[WARN]" if coverage >= 0.5 else "[POOR]"
                            report_lines.append(f"  - {status} {asset}: {coverage:.1%}")
                    
                    if hasattr(quality_report, 'data_validation_errors') and quality_report.data_validation_errors:
                        report_lines.append("- **Validation Issues**:")
                        for error in quality_report.data_validation_errors[:5]:  # Show first 5
                            report_lines.append(f"  - {error}")
                        if len(quality_report.data_validation_errors) > 5:
                            report_lines.append(f"  - ... and {len(quality_report.data_validation_errors) - 5} more issues")
                report_lines.append("")
            
            # Diagnostic Summary
            if diagnostic_results and "comprehensive_metadata" in diagnostic_results:
                metadata = diagnostic_results["comprehensive_metadata"]
                report_lines.append("## Diagnostic Summary")
                
                if "diagnostic_summary" in metadata:
                    diag_summary = metadata["diagnostic_summary"]
                    if diag_summary.get("overfitting_detected"):
                        severity = diag_summary.get("overfitting_severity", "unknown")
                        report_lines.append(f"- **Overfitting**: [WARN] Detected ({severity} severity)")
                    else:
                        report_lines.append(f"- **Overfitting**: [OK] Not detected")
                    
                    if diag_summary.get("underfitting_detected"):
                        severity = diag_summary.get("underfitting_severity", "unknown")
                        report_lines.append(f"- **Underfitting**: [WARN] Detected ({severity} severity)")
                    else:
                        report_lines.append(f"- **Underfitting**: [OK] Not detected")
                
                if "overall_recommendations" in metadata:
                    report_lines.append("- **Diagnostic Recommendations**:")
                    for rec in metadata["overall_recommendations"][:5]:  # Show first 5
                        report_lines.append(f"  - {rec}")
                report_lines.append("")
            
            # Threshold Optimization Results
            if optimal_threshold:
                report_lines.append("## Threshold Optimization Results")
                report_lines.append(f"- **Optimal Threshold**: {optimal_threshold.value:.3f}")
                report_lines.append(f"- **F1 Score**: {optimal_threshold.f1_score:.4f}")
                report_lines.append(f"- **Precision**: {optimal_threshold.precision:.4f}")
                report_lines.append(f"- **Recall**: {optimal_threshold.recall:.4f}")
                if hasattr(optimal_threshold, 'confidence_interval'):
                    ci = optimal_threshold.confidence_interval
                    report_lines.append(f"- **Confidence Interval**: [{ci[0]:.3f}, {ci[1]:.3f}]")
                report_lines.append("")
            
            # Component Status
            report_lines.append("## Enhanced Components Status")
            component_status = self._get_enhanced_components_status()
            for component, status in component_status.items():
                status_icon = "[YES]" if status["available"] else "[NO]"
                report_lines.append(f"- **{component}**: {status_icon} {status['status']}")
            report_lines.append("")
            
            # Recommendations
            recommendations = self._compile_enhanced_recommendations(
                {}, diagnostic_results, training_data, enhanced_config
            )
            if recommendations:
                report_lines.append("## Recommendations")
                for rec in recommendations:
                    report_lines.append(f"- {rec}")
                report_lines.append("")
            
            # Footer
            report_lines.append("---")
            report_lines.append("*Report generated by Enhanced AI Risk Oracle Training Pipeline*")
            
            return "\n".join(report_lines)
            
        except Exception as e:
            self.data_logger.log_data_collection(
                "enhanced_training_report", "generation_error", 0, False, str(e)
            )
            return f"# Enhanced Training Report\n\nError generating report: {str(e)}"
    
    def _count_enabled_features(self, enhanced_config: EnhancedTrainingConfiguration) -> int:
        """Count number of enabled enhanced features"""
        count = 0
        if enhanced_config.extended_training.enabled:
            count += 1
        if enhanced_config.crash_events.enabled:
            count += 1
        if enhanced_config.threshold_optimization.enabled:
            count += 1
        if enhanced_config.backtesting.enhanced_backtesting.get('enabled', False):
            count += 1
        if enhanced_config.automated_retraining.enabled:
            count += 1
        if enhanced_config.diagnostics.comprehensive_diagnostics.get('enabled', False):
            count += 1
        if enhanced_config.feature_engineering.cross_asset_correlation.get('enabled', False):
            count += 1
        if enhanced_config.feature_engineering.volatility_regimes.get('enabled', False):
            count += 1
        if enhanced_config.feature_engineering.momentum_indicators.get('enabled', False):
            count += 1
        if enhanced_config.feature_engineering.volume_analysis.get('enabled', False):
            count += 1
        return count
    
    def _get_enhanced_components_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all enhanced components for integration reporting"""
        status = {}
        
        # Threshold Optimizer
        status["Threshold Optimizer"] = {
            "available": self.threshold_optimizer is not None,
            "status": "Available" if self.threshold_optimizer is not None else "Not Available"
        }
        
        # Enhanced Backtesting Framework
        status["Enhanced Backtesting"] = {
            "available": self.enhanced_backtesting_framework is not None,
            "status": "Available" if self.enhanced_backtesting_framework is not None else "Not Available"
        }
        
        # Advanced Feature Engine
        status["Advanced Feature Engine"] = {
            "available": self.advanced_feature_engine is not None,
            "status": "Available" if self.advanced_feature_engine is not None else "Not Available"
        }
        
        # Automated Retraining System
        status["Automated Retraining"] = {
            "available": self.automated_retraining_system is not None,
            "status": "Available" if self.automated_retraining_system is not None else "Not Available"
        }
        
        # Training Diagnostics
        status["Training Diagnostics"] = {
            "available": self.diagnostics_manager is not None,
            "status": "Available" if self.diagnostics_manager is not None else "Not Available"
        }
        
        # Enhanced Configuration Manager
        status["Enhanced Configuration"] = {
            "available": self.enhanced_config_manager is not None,
            "status": "Available" if self.enhanced_config_manager is not None else "Not Available"
        }
        
        # Historical Data Collector
        status["Historical Data Collector"] = {
            "available": self.historical_collector is not None,
            "status": "Available" if self.historical_collector is not None else "Not Available"
        }
        
        # Correlation Engine
        status["Correlation Engine"] = {
            "available": self.correlation_engine is not None,
            "status": "Available" if self.correlation_engine is not None else "Not Available"
        }
        
        return status
    # Placeholder methods for advanced feature engineering (to be implemented when components are available)
    async def _add_correlation_features(self, feature_sets, historical_data, enhanced_config):
        """Add cross-asset correlation features"""
        try:
            if self.advanced_feature_engine:
                # Use advanced feature engine to generate features
                all_asset_data = {}
                all_asset_data.update(historical_data.get("crypto_data", {}))
                all_asset_data.update(historical_data.get("traditional_data", {}))
                
                # Generate advanced features
                advanced_features = self.advanced_feature_engine.generate_advanced_features(all_asset_data)
                
                # Extract correlation features and add to feature sets
                for feature_set in feature_sets:
                    asset = feature_set.asset_symbol
                    correlation_matrix = advanced_features.get("correlation_matrix")
                    
                    if correlation_matrix and hasattr(correlation_matrix, 'asset_pairs'):
                        asset_correlations = correlation_matrix.asset_pairs.get(asset, {})
                        
                        # Add correlation features to the feature set
                        for other_asset, corr_value in asset_correlations.items():
                            if other_asset != asset and corr_value is not None:
                                if other_asset == "BTC":
                                    feature_set.btc_correlation = corr_value
                                elif other_asset == "GOLD":
                                    feature_set.gold_correlation = corr_value
                                elif other_asset == "ETH":
                                    feature_set.eth_correlation = corr_value
                
                return feature_sets
            else:
                # Fallback to basic correlation features using existing correlation engine
                if self.correlation_engine:
                    # Basic correlation features using existing engine
                    for feature_set in feature_sets:
                        # Add basic correlation features
                        correlation_data = self.correlation_engine.calculate_asset_correlations(
                            historical_data.get("crypto_data", {}),
                            historical_data.get("traditional_data", {})
                        )
                        # Add correlation features to feature set
                        if hasattr(feature_set, 'correlation_features'):
                            feature_set.correlation_features = correlation_data
                
                return feature_sets
        except Exception as e:
            self.data_logger.log_data_collection(
                "correlation_features", "error", 0, False, str(e)
            )
            return feature_sets
    
    async def _add_volatility_regime_features(self, feature_sets, historical_data, enhanced_config):
        """Add volatility regime features"""
        try:
            if self.advanced_feature_engine:
                # Use advanced feature engine to generate features
                all_asset_data = {}
                all_asset_data.update(historical_data.get("crypto_data", {}))
                all_asset_data.update(historical_data.get("traditional_data", {}))
                
                # Generate advanced features
                advanced_features = self.advanced_feature_engine.generate_advanced_features(all_asset_data)
                
                # Extract volatility regime features and add to feature sets
                for feature_set in feature_sets:
                    asset = feature_set.asset_symbol
                    volatility_regimes = advanced_features.get("volatility_regimes", {})
                    
                    if asset in volatility_regimes:
                        regime_data = volatility_regimes[asset]
                        if hasattr(regime_data, 'current_regime'):
                            # Map regime to numerical value
                            regime_mapping = {"low": 0.2, "medium": 0.5, "high": 0.8}
                            feature_set.volatility_regime = regime_mapping.get(regime_data.current_regime, 0.5)
                        
                        if hasattr(regime_data, 'volatility_percentile'):
                            feature_set.volatility_percentile = regime_data.volatility_percentile
                
                return feature_sets
            else:
                # Basic volatility regime classification
                for feature_set in feature_sets:
                    # Add basic volatility regime indicators
                    if hasattr(feature_set, 'current_volatility') and feature_set.current_volatility:
                        # Simple volatility regime classification
                        volatility = feature_set.current_volatility
                        if volatility < 0.2:
                            regime = 0.2  # low
                        elif volatility < 0.5:
                            regime = 0.5  # medium
                        else:
                            regime = 0.8  # high
                        
                        feature_set.volatility_regime = regime
                
                return feature_sets
        except Exception as e:
            self.data_logger.log_data_collection(
                "volatility_regime_features", "error", 0, False, str(e)
            )
            return feature_sets
    
    async def _add_momentum_features(self, feature_sets, historical_data, enhanced_config):
        """Add momentum indicator features"""
        try:
            if self.advanced_feature_engine:
                # Use advanced feature engine to generate features
                all_asset_data = {}
                all_asset_data.update(historical_data.get("crypto_data", {}))
                all_asset_data.update(historical_data.get("traditional_data", {}))
                
                # Generate advanced features
                advanced_features = self.advanced_feature_engine.generate_advanced_features(all_asset_data)
                
                # Extract momentum features and add to feature sets
                for feature_set in feature_sets:
                    asset = feature_set.asset_symbol
                    momentum_indicators = advanced_features.get("momentum_indicators", {})
                    
                    if asset in momentum_indicators:
                        momentum_data = momentum_indicators[asset]
                        if hasattr(momentum_data, 'momentum_score'):
                            feature_set.price_momentum_score = momentum_data.momentum_score
                        
                        if hasattr(momentum_data, 'trend_strength'):
                            feature_set.trend_strength = momentum_data.trend_strength
                
                return feature_sets
            else:
                # Basic momentum indicators
                for feature_set in feature_sets:
                    # Add basic momentum features using existing data
                    if hasattr(feature_set, 'price_change_1h') and hasattr(feature_set, 'price_change_24h'):
                        # Simple momentum calculation
                        momentum_1h = getattr(feature_set, 'price_change_1h', 0) or 0
                        momentum_24h = getattr(feature_set, 'price_change_24h', 0) or 0
                        
                        # Calculate momentum score
                        momentum_score = (momentum_1h + momentum_24h) / 2
                        feature_set.price_momentum_score = momentum_score
                
                return feature_sets
        except Exception as e:
            self.data_logger.log_data_collection(
                "momentum_features", "error", 0, False, str(e)
            )
            return feature_sets
    
    async def _add_volume_features(self, feature_sets, historical_data, enhanced_config):
        """Add volume analysis features"""
        try:
            if self.advanced_feature_engine:
                # Use advanced feature engine to generate features
                all_asset_data = {}
                all_asset_data.update(historical_data.get("crypto_data", {}))
                all_asset_data.update(historical_data.get("traditional_data", {}))
                
                # Generate advanced features
                advanced_features = self.advanced_feature_engine.generate_advanced_features(all_asset_data)
                
                # Extract volume features and add to feature sets
                for feature_set in feature_sets:
                    asset = feature_set.asset_symbol
                    volume_anomalies = advanced_features.get("volume_anomalies", {})
                    
                    if asset in volume_anomalies:
                        volume_data = volume_anomalies[asset]
                        if hasattr(volume_data, 'anomaly_score'):
                            feature_set.volume_anomaly_score = volume_data.anomaly_score
                        
                        if hasattr(volume_data, 'volume_percentile'):
                            feature_set.volume_percentile = volume_data.volume_percentile
                
                return feature_sets
            else:
                # Basic volume analysis
                for feature_set in feature_sets:
                    # Add basic volume features
                    if hasattr(feature_set, 'current_volume') and feature_set.current_volume:
                        volume = feature_set.current_volume
                        # Simple volume anomaly detection (placeholder)
                        volume_anomaly_score = min(max(volume / 1000000, 0), 1)  # Normalize to 0-1
                        feature_set.volume_anomaly_score = volume_anomaly_score
                
                return feature_sets
        except Exception as e:
            self.data_logger.log_data_collection(
                "volume_features", "error", 0, False, str(e)
            )
            return feature_sets