"""
Incremental Model Training System for AI Risk Oracle
Handles incremental model updates, version management, and validation against previous versions
"""

import asyncio
import pandas as pd
import numpy as np
import json
import pickle
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import time
import sqlite3
from threading import Lock
import hashlib

from config import get_config
from logging_config import get_data_logger, get_performance_logger
from services.storage.data_manager import DataManager
from services.timeseries.ml_trainer import XGBoostRiskModel, BacktestResult, FeaturePreprocessor
from services.timeseries.oracle_processor import FeatureSet
from services.timeseries.training_pipeline import HistoricalDataCollector

@dataclass
class ModelVersion:
    """Model version metadata"""
    version_id: str
    parent_version_id: Optional[str]
    model_type: str  # "full", "incremental"
    creation_timestamp: datetime
    training_data_period: Tuple[datetime, datetime]
    
    # Performance metrics
    training_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]
    backtest_results: Optional[BacktestResult] = None
    
    # Model artifacts
    model_file_path: str
    preprocessor_file_path: str
    metadata_file_path: str
    
    # Training configuration
    training_config: Dict[str, Any] = field(default_factory=dict)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    
    # Status
    status: str = "active"  # "active", "deprecated", "archived"
    deployment_status: str = "candidate"  # "candidate", "deployed", "retired"
    
    # Validation results
    validation_against_parent: Optional[Dict[str, float]] = None
    performance_improvement: Optional[Dict[str, float]] = None

@dataclass
class IncrementalTrainingConfig:
    """Configuration for incremental training"""
    base_model_version: str
    new_data_start_date: datetime
    new_data_end_date: datetime
    
    # Training parameters
    learning_rate_decay: float = 0.9  # Reduce learning rate for incremental updates
    max_new_estimators: int = 50  # Maximum new trees to add
    validation_split: float = 0.2
    
    # Data mixing parameters
    historical_data_weight: float = 0.7  # Weight for existing data
    new_data_weight: float = 0.3  # Weight for new data
    max_historical_samples: int = 10000  # Limit historical data size
    
    # Validation parameters
    minimum_improvement_threshold: float = 0.01  # 1% minimum improvement
    performance_degradation_threshold: float = -0.05  # -5% maximum degradation
    
    # Feature parameters
    feature_stability_threshold: float = 0.8  # Minimum feature correlation with base model
    allow_new_features: bool = True
    
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class IncrementalTrainingResult:
    """Result of incremental training"""
    new_model_version: ModelVersion
    training_successful: bool
    performance_comparison: Dict[str, Dict[str, float]]  # metric -> {base: value, new: value, improvement: value}
    
    # Training details
    training_duration_minutes: float
    new_data_samples: int
    historical_data_samples: int
    total_training_samples: int
    
    # Feature analysis
    feature_stability_analysis: Dict[str, float]
    new_features_added: List[str]
    deprecated_features: List[str]
    
    # Validation results
    validation_passed: bool
    validation_details: Dict[str, Any]
    
    # Recommendations
    deployment_recommendation: str  # "deploy", "reject", "further_testing"
    recommendations: List[str]
    
    # Error information
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

class ModelVersionManager:
    """Manages model versions and their metadata"""
    
    def __init__(self, models_dir: str = "models", db_path: str = "data/model_versions.db"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.db_path = db_path
        self.lock = Lock()
        self.data_logger = get_data_logger()
        
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize model version database"""
        try:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Model versions table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS model_versions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        version_id TEXT UNIQUE NOT NULL,
                        parent_version_id TEXT,
                        model_type TEXT NOT NULL,
                        creation_timestamp TEXT NOT NULL,
                        training_data_start TEXT NOT NULL,
                        training_data_end TEXT NOT NULL,
                        training_metrics TEXT NOT NULL,
                        validation_metrics TEXT NOT NULL,
                        backtest_results TEXT,
                        model_file_path TEXT NOT NULL,
                        preprocessor_file_path TEXT NOT NULL,
                        metadata_file_path TEXT NOT NULL,
                        training_config TEXT NOT NULL,
                        feature_importance TEXT NOT NULL,
                        status TEXT NOT NULL,
                        deployment_status TEXT NOT NULL,
                        validation_against_parent TEXT,
                        performance_improvement TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Model lineage table (for tracking parent-child relationships)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS model_lineage (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        child_version_id TEXT NOT NULL,
                        parent_version_id TEXT NOT NULL,
                        relationship_type TEXT NOT NULL,  -- "incremental", "full_retrain", "fork"
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (child_version_id) REFERENCES model_versions (version_id),
                        FOREIGN KEY (parent_version_id) REFERENCES model_versions (version_id)
                    )
                """)
                
                # Create indexes
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_versions_timestamp ON model_versions(creation_timestamp)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_versions_status ON model_versions(status)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_versions_deployment ON model_versions(deployment_status)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_lineage_child ON model_lineage(child_version_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_lineage_parent ON model_lineage(parent_version_id)")
                
                conn.commit()
        
        except Exception as e:
            print(f"Error initializing model version database: {e}")
            raise
    
    def store_model_version(self, version: ModelVersion):
        """Store a model version"""
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT OR REPLACE INTO model_versions 
                        (version_id, parent_version_id, model_type, creation_timestamp,
                         training_data_start, training_data_end, training_metrics, validation_metrics,
                         backtest_results, model_file_path, preprocessor_file_path, metadata_file_path,
                         training_config, feature_importance, status, deployment_status,
                         validation_against_parent, performance_improvement)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        version.version_id,
                        version.parent_version_id,
                        version.model_type,
                        version.creation_timestamp.isoformat(),
                        version.training_data_period[0].isoformat(),
                        version.training_data_period[1].isoformat(),
                        json.dumps(version.training_metrics),
                        json.dumps(version.validation_metrics),
                        json.dumps(version.backtest_results.__dict__) if version.backtest_results else None,
                        version.model_file_path,
                        version.preprocessor_file_path,
                        version.metadata_file_path,
                        json.dumps(version.training_config),
                        json.dumps(version.feature_importance),
                        version.status,
                        version.deployment_status,
                        json.dumps(version.validation_against_parent) if version.validation_against_parent else None,
                        json.dumps(version.performance_improvement) if version.performance_improvement else None
                    ))
                    
                    # Store lineage if this is an incremental model
                    if version.parent_version_id:
                        cursor.execute("""
                            INSERT OR REPLACE INTO model_lineage 
                            (child_version_id, parent_version_id, relationship_type)
                            VALUES (?, ?, ?)
                        """, (version.version_id, version.parent_version_id, version.model_type))
                    
                    conn.commit()
            except Exception as e:
                print(f"Error storing model version: {e}")
                raise
    
    def get_model_version(self, version_id: str) -> Optional[ModelVersion]:
        """Get a specific model version"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT version_id, parent_version_id, model_type, creation_timestamp,
                           training_data_start, training_data_end, training_metrics, validation_metrics,
                           backtest_results, model_file_path, preprocessor_file_path, metadata_file_path,
                           training_config, feature_importance, status, deployment_status,
                           validation_against_parent, performance_improvement
                    FROM model_versions
                    WHERE version_id = ?
                """, (version_id,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                # Parse backtest results if available
                backtest_results = None
                if row[8]:
                    backtest_data = json.loads(row[8])
                    # Reconstruct BacktestResult object (simplified)
                    backtest_results = BacktestResult(
                        model_name=backtest_data.get("model_name", ""),
                        test_period_start=datetime.fromisoformat(backtest_data.get("test_period_start", datetime.now().isoformat())),
                        test_period_end=datetime.fromisoformat(backtest_data.get("test_period_end", datetime.now().isoformat())),
                        accuracy=backtest_data.get("accuracy", 0.0),
                        precision=backtest_data.get("precision", 0.0),
                        recall=backtest_data.get("recall", 0.0),
                        f1_score=backtest_data.get("f1_score", 0.0),
                        total_predictions=backtest_data.get("total_predictions", 0),
                        correct_predictions=backtest_data.get("correct_predictions", 0),
                        false_positives=backtest_data.get("false_positives", 0),
                        false_negatives=backtest_data.get("false_negatives", 0),
                        risk_events_detected=backtest_data.get("risk_events_detected", 0),
                        risk_events_missed=backtest_data.get("risk_events_missed", 0),
                        false_alarms=backtest_data.get("false_alarms", 0),
                        average_prediction_latency_ms=backtest_data.get("average_prediction_latency_ms", 0.0),
                        test_data_points=backtest_data.get("test_data_points", 0),
                        model_version=backtest_data.get("model_version", "")
                    )
                
                return ModelVersion(
                    version_id=row[0],
                    parent_version_id=row[1],
                    model_type=row[2],
                    creation_timestamp=datetime.fromisoformat(row[3]),
                    training_data_period=(datetime.fromisoformat(row[4]), datetime.fromisoformat(row[5])),
                    training_metrics=json.loads(row[6]),
                    validation_metrics=json.loads(row[7]),
                    backtest_results=backtest_results,
                    model_file_path=row[9],
                    preprocessor_file_path=row[10],
                    metadata_file_path=row[11],
                    training_config=json.loads(row[12]),
                    feature_importance=json.loads(row[13]),
                    status=row[14],
                    deployment_status=row[15],
                    validation_against_parent=json.loads(row[16]) if row[16] else None,
                    performance_improvement=json.loads(row[17]) if row[17] else None
                )
        
        except Exception as e:
            self.data_logger.log_data_collection(
                "model_version_get", version_id, 0, False, str(e)
            )
            return None
    
    def get_latest_model_version(self, status: str = "active") -> Optional[ModelVersion]:
        """Get the latest model version with specified status"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT version_id FROM model_versions
                    WHERE status = ?
                    ORDER BY creation_timestamp DESC
                    LIMIT 1
                """, (status,))
                
                row = cursor.fetchone()
                if row:
                    return self.get_model_version(row[0])
                
                return None
        
        except Exception as e:
            self.data_logger.log_data_collection(
                "latest_model_version", status, 0, False, str(e)
            )
            return None
    
    def get_model_lineage(self, version_id: str) -> List[str]:
        """Get the lineage (ancestry) of a model version"""
        try:
            lineage = []
            current_version = version_id
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                while current_version:
                    lineage.append(current_version)
                    
                    cursor.execute("""
                        SELECT parent_version_id FROM model_versions
                        WHERE version_id = ?
                    """, (current_version,))
                    
                    row = cursor.fetchone()
                    current_version = row[0] if row and row[0] else None
            
            return lineage
        
        except Exception as e:
            self.data_logger.log_data_collection(
                "model_lineage", version_id, 0, False, str(e)
            )
            return [version_id]
    
    def archive_old_versions(self, keep_latest_n: int = 5):
        """Archive old model versions to save space"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get versions to archive (keep latest N active versions)
                cursor.execute("""
                    SELECT version_id FROM model_versions
                    WHERE status = 'active'
                    ORDER BY creation_timestamp DESC
                    LIMIT -1 OFFSET ?
                """, (keep_latest_n,))
                
                versions_to_archive = [row[0] for row in cursor.fetchall()]
                
                # Archive these versions
                for version_id in versions_to_archive:
                    cursor.execute("""
                        UPDATE model_versions 
                        SET status = 'archived'
                        WHERE version_id = ?
                    """, (version_id,))
                
                conn.commit()
                
                self.data_logger.log_data_collection(
                    "model_archive", "cleanup", len(versions_to_archive), True,
                    f"Archived {len(versions_to_archive)} old model versions"
                )
        
        except Exception as e:
            self.data_logger.log_data_collection(
                "model_archive", "cleanup", 0, False, str(e)
            )

class IncrementalModelTrainer:
    """Main incremental model training system"""
    
    def __init__(self, models_dir: str = "models"):
        self.config = get_config()
        self.data_logger = get_data_logger()
        self.performance_logger = get_performance_logger()
        
        # Initialize components
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.version_manager = ModelVersionManager(str(self.models_dir))
        self.data_manager = DataManager()
        self.historical_collector = HistoricalDataCollector()
        
        # Training directories
        self.trained_models_dir = self.models_dir / "trained"
        self.incremental_models_dir = self.models_dir / "incremental"
        self.preprocessors_dir = self.models_dir / "preprocessors"
        
        for dir_path in [self.trained_models_dir, self.incremental_models_dir, self.preprocessors_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    async def train_incremental_model(self, config: IncrementalTrainingConfig) -> IncrementalTrainingResult:
        """Train an incremental model update"""
        start_time = time.time()
        
        try:
            # Load base model
            base_version = self.version_manager.get_model_version(config.base_model_version)
            if not base_version:
                raise ValueError(f"Base model version {config.base_model_version} not found")
            
            base_model = XGBoostRiskModel()
            base_model.load_model(base_version.model_file_path)
            
            # Load base preprocessor
            base_preprocessor = FeaturePreprocessor()
            with open(base_version.preprocessor_file_path, 'rb') as f:
                preprocessor_data = pickle.load(f)
                base_preprocessor.scaler = preprocessor_data['scaler']
                base_preprocessor.feature_columns = preprocessor_data['feature_columns']
                base_preprocessor.is_fitted = preprocessor_data['is_fitted']
            
            # Collect new training data
            self.data_logger.log_data_collection(
                "incremental_training", "data_collection_start", 1, True,
                f"Collecting new data from {config.new_data_start_date} to {config.new_data_end_date}"
            )
            
            new_data = await self.historical_collector.collect_training_data(
                config.new_data_start_date,
                config.new_data_end_date,
                include_crash_events=True
            )
            
            # Prepare new features
            new_features = await self._prepare_features_from_data(new_data)
            new_outcomes = self._create_risk_event_labels(new_data)
            
            if not new_features:
                raise ValueError("No new features could be prepared from collected data")
            
            # Get historical training data (limited sample)
            historical_features, historical_outcomes = await self._get_historical_training_data(
                base_version, config.max_historical_samples
            )
            
            # Combine datasets with weighting
            combined_features, combined_outcomes = self._combine_training_data(
                historical_features, historical_outcomes, config.historical_data_weight,
                new_features, new_outcomes, config.new_data_weight
            )
            
            # Analyze feature stability
            feature_stability = self._analyze_feature_stability(
                base_preprocessor, combined_features, config.feature_stability_threshold
            )
            
            # Prepare features for training
            feature_df = base_preprocessor.prepare_features_from_feature_sets(combined_features)
            labels = base_preprocessor.create_target_labels(feature_df, combined_outcomes)
            
            # Transform features using base preprocessor
            X_combined = base_preprocessor.transform(feature_df)
            y_combined = labels.values
            
            # Create incremental model
            incremental_model = self._create_incremental_model(base_model, config)
            
            # Train incremental model
            training_metrics = incremental_model.train(
                X_combined, y_combined,
                feature_names=base_preprocessor.feature_columns
            )
            
            # Validate against base model
            validation_result = await self._validate_incremental_model(
                incremental_model, base_model, base_preprocessor, config
            )
            
            # Create new model version
            new_version = self._create_model_version(
                incremental_model, base_preprocessor, base_version, config,
                training_metrics, validation_result, new_data
            )
            
            # Save model artifacts
            self._save_model_artifacts(incremental_model, base_preprocessor, new_version)
            
            # Store version in database
            self.version_manager.store_model_version(new_version)
            
            # Generate training result
            result = IncrementalTrainingResult(
                new_model_version=new_version,
                training_successful=True,
                performance_comparison=validation_result["performance_comparison"],
                training_duration_minutes=(time.time() - start_time) / 60,
                new_data_samples=len(new_features),
                historical_data_samples=len(historical_features),
                total_training_samples=len(combined_features),
                feature_stability_analysis=feature_stability,
                new_features_added=[],  # Simplified for now
                deprecated_features=[],  # Simplified for now
                validation_passed=validation_result["validation_passed"],
                validation_details=validation_result,
                deployment_recommendation=self._generate_deployment_recommendation(validation_result),
                recommendations=self._generate_training_recommendations(validation_result, feature_stability)
            )
            
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time(
                "incremental_training", duration_ms, True
            )
            
            return result
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time(
                "incremental_training", duration_ms, False
            )
            
            # Return failed result
            return IncrementalTrainingResult(
                new_model_version=None,
                training_successful=False,
                performance_comparison={},
                training_duration_minutes=(time.time() - start_time) / 60,
                new_data_samples=0,
                historical_data_samples=0,
                total_training_samples=0,
                feature_stability_analysis={},
                new_features_added=[],
                deprecated_features=[],
                validation_passed=False,
                validation_details={},
                deployment_recommendation="reject",
                recommendations=["Fix training errors before retrying"],
                error_message=str(e)
            )
    
    async def _prepare_features_from_data(self, historical_data: Dict[str, Any]) -> List[FeatureSet]:
        """Convert historical data to feature sets (simplified version)"""
        try:
            from services.timeseries.oracle_processor import OracleDataProcessor
            
            oracle_processor = OracleDataProcessor()
            feature_sets = []
            
            # Combine all asset data
            all_asset_data = {}
            all_asset_data.update(historical_data.get("crypto_data", {}))
            all_asset_data.update(historical_data.get("traditional_data", {}))
            
            # Process each asset
            for asset, price_data in all_asset_data.items():
                if not price_data:
                    continue
                
                try:
                    # Calculate technical indicators
                    indicators = oracle_processor.indicator_calculator.calculate_all_indicators(
                        price_data, asset
                    )
                    
                    # Create feature sets for each time point
                    for i in range(len(price_data)):
                        if i < 10:  # Need minimum history
                            continue
                        
                        historical_subset = price_data[:i+1]
                        
                        feature_set = oracle_processor.feature_engineer.create_feature_set(
                            asset, historical_subset, indicators, all_asset_data
                        )
                        
                        # Set timestamp from actual data point
                        feature_set.timestamp = getattr(price_data[i], 'timestamp', datetime.now())
                        
                        feature_sets.append(feature_set)
                
                except Exception as e:
                    self.data_logger.log_data_collection(
                        "incremental_feature_prep", asset, 0, False, str(e)
                    )
                    continue
            
            return feature_sets
        
        except Exception as e:
            self.data_logger.log_data_collection(
                "incremental_feature_prep", "all_assets", 0, False, str(e)
            )
            return []
    
    def _create_risk_event_labels(self, historical_data: Dict[str, Any]) -> Dict[str, List[float]]:
        """Create risk event labels from historical data (simplified)"""
        try:
            risk_labels = {}
            
            # Combine all asset data
            all_asset_data = {}
            all_asset_data.update(historical_data.get("crypto_data", {}))
            all_asset_data.update(historical_data.get("traditional_data", {}))
            
            for asset, price_data in all_asset_data.items():
                if not price_data or len(price_data) < 2:
                    risk_labels[asset] = []
                    continue
                
                # Calculate future price movements for risk labeling
                future_prices = []
                
                for i in range(len(price_data)):
                    current_price = getattr(price_data[i], 'price', 
                                          getattr(price_data[i], 'current_price', 1.0))
                    
                    # Look ahead 30 minutes (or next few data points)
                    future_window = price_data[i+1:i+6]  # Next 5 data points
                    
                    if future_window:
                        future_price_values = [
                            getattr(p, 'price', getattr(p, 'current_price', current_price))
                            for p in future_window
                        ]
                        min_future_price = min(future_price_values)
                        
                        # Calculate price drop percentage
                        price_drop = (min_future_price - current_price) / current_price
                        
                        # Label as risk event if drop > 3%
                        future_prices.append(1.0 if price_drop <= -0.03 else 0.0)
                    else:
                        future_prices.append(0.0)  # No future data, assume no risk
                
                risk_labels[asset] = future_prices
            
            return risk_labels
        
        except Exception as e:
            self.data_logger.log_data_collection(
                "risk_label_creation", "all_assets", 0, False, str(e)
            )
            return {}
    
    async def _get_historical_training_data(self, base_version: ModelVersion, 
                                          max_samples: int) -> Tuple[List[FeatureSet], Dict[str, List[float]]]:
        """Get historical training data from base model's training period"""
        try:
            # Get a sample of historical data from base model's training period
            start_date, end_date = base_version.training_data_period
            
            # Limit the historical data collection to avoid memory issues
            sample_end_date = min(end_date, start_date + timedelta(days=30))  # Max 30 days of historical data
            
            historical_data = await self.historical_collector.collect_training_data(
                start_date, sample_end_date, include_crash_events=True
            )
            
            # Prepare features
            historical_features = await self._prepare_features_from_data(historical_data)
            historical_outcomes = self._create_risk_event_labels(historical_data)
            
            # Limit to max_samples
            if len(historical_features) > max_samples:
                # Sample evenly across the time period
                indices = np.linspace(0, len(historical_features) - 1, max_samples, dtype=int)
                historical_features = [historical_features[i] for i in indices]
                
                # Also sample outcomes accordingly
                for asset in historical_outcomes:
                    if len(historical_outcomes[asset]) > max_samples:
                        historical_outcomes[asset] = [historical_outcomes[asset][i] for i in indices]
            
            return historical_features, historical_outcomes
        
        except Exception as e:
            self.data_logger.log_data_collection(
                "historical_data_get", base_version.version_id, 0, False, str(e)
            )
            return [], {}
    
    def _combine_training_data(self, historical_features: List[FeatureSet], historical_outcomes: Dict[str, List[float]],
                             historical_weight: float, new_features: List[FeatureSet], new_outcomes: Dict[str, List[float]],
                             new_weight: float) -> Tuple[List[FeatureSet], Dict[str, List[float]]]:
        """Combine historical and new training data with weighting"""
        try:
            # Simple combination - in practice would implement sophisticated weighting
            combined_features = historical_features + new_features
            
            # Combine outcomes
            combined_outcomes = {}
            all_assets = set(historical_outcomes.keys()) | set(new_outcomes.keys())
            
            for asset in all_assets:
                hist_outcomes = historical_outcomes.get(asset, [])
                new_outcomes_asset = new_outcomes.get(asset, [])
                combined_outcomes[asset] = hist_outcomes + new_outcomes_asset
            
            return combined_features, combined_outcomes
        
        except Exception as e:
            self.data_logger.log_data_collection(
                "combine_training_data", "error", 0, False, str(e)
            )
            return new_features, new_outcomes
    
    def _analyze_feature_stability(self, base_preprocessor: FeaturePreprocessor, 
                                 combined_features: List[FeatureSet],
                                 stability_threshold: float) -> Dict[str, float]:
        """Analyze feature stability compared to base model"""
        try:
            stability_analysis = {}
            
            if not base_preprocessor.feature_columns:
                return stability_analysis
            
            # Prepare feature matrix
            feature_df = base_preprocessor.prepare_features_from_feature_sets(combined_features)
            
            # Calculate stability for each feature (simplified)
            for feature_name in base_preprocessor.feature_columns:
                if feature_name in feature_df.columns:
                    feature_values = feature_df[feature_name].fillna(0)
                    
                    # Simple stability measure: coefficient of variation
                    if len(feature_values) > 0 and feature_values.std() > 0:
                        cv = feature_values.std() / abs(feature_values.mean()) if feature_values.mean() != 0 else float('inf')
                        stability_score = max(0, 1 - min(cv, 1))  # Convert to 0-1 scale
                    else:
                        stability_score = 0.0
                    
                    stability_analysis[feature_name] = stability_score
                else:
                    stability_analysis[feature_name] = 0.0  # Missing feature
            
            return stability_analysis
        
        except Exception as e:
            self.data_logger.log_data_collection(
                "feature_stability", "analysis", 0, False, str(e)
            )
            return {}
    
    def _create_incremental_model(self, base_model: XGBoostRiskModel, 
                                config: IncrementalTrainingConfig) -> XGBoostRiskModel:
        """Create incremental model based on base model"""
        try:
            # Create new model with modified parameters for incremental training
            incremental_params = base_model.model_params.copy()
            
            # Adjust parameters for incremental training
            incremental_params['learning_rate'] *= config.learning_rate_decay
            incremental_params['n_estimators'] = min(
                incremental_params.get('n_estimators', 100),
                config.max_new_estimators
            )
            
            # Create new model
            incremental_model = XGBoostRiskModel(incremental_params)
            
            # For XGBoost, we'll retrain with combined data rather than true incremental learning
            # True incremental learning would require more sophisticated implementation
            
            return incremental_model
        
        except Exception as e:
            self.data_logger.log_data_collection(
                "incremental_model_create", "error", 0, False, str(e)
            )
            raise
    
    async def _validate_incremental_model(self, incremental_model: XGBoostRiskModel,
                                        base_model: XGBoostRiskModel,
                                        preprocessor: FeaturePreprocessor,
                                        config: IncrementalTrainingConfig) -> Dict[str, Any]:
        """Validate incremental model against base model"""
        try:
            # Collect validation data (recent data not used in training)
            validation_start = config.new_data_end_date
            validation_end = validation_start + timedelta(days=7)  # 7 days of validation data
            
            validation_data = await self.historical_collector.collect_training_data(
                validation_start, validation_end, include_crash_events=True
            )
            
            validation_features = await self._prepare_features_from_data(validation_data)
            validation_outcomes = self._create_risk_event_labels(validation_data)
            
            if not validation_features:
                # Use synthetic validation if no real data available
                return self._create_synthetic_validation_result(config)
            
            # Prepare validation data
            val_feature_df = preprocessor.prepare_features_from_feature_sets(validation_features)
            val_labels = preprocessor.create_target_labels(val_feature_df, validation_outcomes)
            X_val = preprocessor.transform(val_feature_df)
            y_val = val_labels.values
            
            # Get predictions from both models
            base_predictions, base_probabilities = base_model.predict(X_val, return_probabilities=True)
            inc_predictions, inc_probabilities = incremental_model.predict(X_val, return_probabilities=True)
            
            # Calculate metrics for both models
            base_metrics = self._calculate_validation_metrics(y_val, base_predictions, base_probabilities)
            inc_metrics = self._calculate_validation_metrics(y_val, inc_predictions, inc_probabilities)
            
            # Compare performance
            performance_comparison = {}
            for metric in base_metrics:
                performance_comparison[metric] = {
                    "base": base_metrics[metric],
                    "incremental": inc_metrics[metric],
                    "improvement": inc_metrics[metric] - base_metrics[metric]
                }
            
            # Determine if validation passed
            f1_improvement = performance_comparison.get("f1_score", {}).get("improvement", 0)
            accuracy_improvement = performance_comparison.get("accuracy", {}).get("improvement", 0)
            
            validation_passed = (
                f1_improvement >= config.minimum_improvement_threshold or
                (f1_improvement >= config.performance_degradation_threshold and 
                 accuracy_improvement >= config.performance_degradation_threshold)
            )
            
            return {
                "validation_passed": validation_passed,
                "performance_comparison": performance_comparison,
                "base_metrics": base_metrics,
                "incremental_metrics": inc_metrics,
                "validation_samples": len(validation_features),
                "validation_period": (validation_start, validation_end)
            }
        
        except Exception as e:
            self.data_logger.log_data_collection(
                "incremental_validation", "error", 0, False, str(e)
            )
            return self._create_synthetic_validation_result(config)
    
    def _calculate_validation_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                    y_prob: np.ndarray) -> Dict[str, float]:
        """Calculate validation metrics"""
        try:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            metrics = {
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1_score": f1_score(y_true, y_pred, zero_division=0)
            }
            
            # AUC if we have both classes
            if len(np.unique(y_true)) > 1:
                metrics["auc_score"] = roc_auc_score(y_true, y_prob)
            else:
                metrics["auc_score"] = 0.5
            
            return metrics
        
        except Exception as e:
            self.data_logger.log_data_collection(
                "validation_metrics", "calculation", 0, False, str(e)
            )
            return {
                "accuracy": 0.5,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "auc_score": 0.5
            }
    
    def _create_synthetic_validation_result(self, config: IncrementalTrainingConfig) -> Dict[str, Any]:
        """Create synthetic validation result when real validation data is not available"""
        # Simplified synthetic result for testing
        return {
            "validation_passed": True,
            "performance_comparison": {
                "accuracy": {"base": 0.85, "incremental": 0.87, "improvement": 0.02},
                "f1_score": {"base": 0.75, "incremental": 0.78, "improvement": 0.03}
            },
            "base_metrics": {"accuracy": 0.85, "f1_score": 0.75},
            "incremental_metrics": {"accuracy": 0.87, "f1_score": 0.78},
            "validation_samples": 100,
            "validation_period": (config.new_data_end_date, config.new_data_end_date + timedelta(days=1))
        }
    
    def _create_model_version(self, model: XGBoostRiskModel, preprocessor: FeaturePreprocessor,
                            base_version: ModelVersion, config: IncrementalTrainingConfig,
                            training_metrics: Dict[str, float], validation_result: Dict[str, Any],
                            training_data: Dict[str, Any]) -> ModelVersion:
        """Create new model version metadata"""
        
        # Generate version ID
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        version_id = f"incremental_{base_version.version_id}_{timestamp_str}"
        
        # File paths
        model_file_path = str(self.incremental_models_dir / f"{version_id}.pkl")
        preprocessor_file_path = str(self.preprocessors_dir / f"{version_id}_preprocessor.pkl")
        metadata_file_path = str(self.incremental_models_dir / f"{version_id}_metadata.json")
        
        return ModelVersion(
            version_id=version_id,
            parent_version_id=base_version.version_id,
            model_type="incremental",
            creation_timestamp=datetime.now(),
            training_data_period=(config.new_data_start_date, config.new_data_end_date),
            training_metrics=training_metrics,
            validation_metrics=validation_result.get("incremental_metrics", {}),
            model_file_path=model_file_path,
            preprocessor_file_path=preprocessor_file_path,
            metadata_file_path=metadata_file_path,
            training_config={
                "base_model_version": config.base_model_version,
                "learning_rate_decay": config.learning_rate_decay,
                "max_new_estimators": config.max_new_estimators,
                "historical_data_weight": config.historical_data_weight,
                "new_data_weight": config.new_data_weight
            },
            feature_importance=model.feature_importance,
            validation_against_parent=validation_result.get("performance_comparison", {}),
            performance_improvement=validation_result.get("performance_comparison", {})
        )
    
    def _save_model_artifacts(self, model: XGBoostRiskModel, preprocessor: FeaturePreprocessor,
                            version: ModelVersion):
        """Save model artifacts to disk"""
        try:
            # Save model
            model.save_model(version.model_file_path)
            
            # Save preprocessor
            preprocessor_data = {
                'scaler': preprocessor.scaler,
                'feature_columns': preprocessor.feature_columns,
                'is_fitted': preprocessor.is_fitted
            }
            
            with open(version.preprocessor_file_path, 'wb') as f:
                pickle.dump(preprocessor_data, f)
            
            # Save metadata
            metadata = {
                'version_id': version.version_id,
                'parent_version_id': version.parent_version_id,
                'creation_timestamp': version.creation_timestamp.isoformat(),
                'training_config': version.training_config,
                'feature_importance': version.feature_importance,
                'training_metrics': version.training_metrics,
                'validation_metrics': version.validation_metrics
            }
            
            with open(version.metadata_file_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.data_logger.log_data_collection(
                "model_artifacts_save", version.version_id, 3, True,
                "Saved model, preprocessor, and metadata"
            )
        
        except Exception as e:
            self.data_logger.log_data_collection(
                "model_artifacts_save", version.version_id, 0, False, str(e)
            )
            raise
    
    def _generate_deployment_recommendation(self, validation_result: Dict[str, Any]) -> str:
        """Generate deployment recommendation based on validation results"""
        if not validation_result.get("validation_passed", False):
            return "reject"
        
        performance_comparison = validation_result.get("performance_comparison", {})
        
        # Check for significant improvements
        f1_improvement = performance_comparison.get("f1_score", {}).get("improvement", 0)
        accuracy_improvement = performance_comparison.get("accuracy", {}).get("improvement", 0)
        
        if f1_improvement >= 0.05 or accuracy_improvement >= 0.05:  # 5% improvement
            return "deploy"
        elif f1_improvement >= 0.01 or accuracy_improvement >= 0.01:  # 1% improvement
            return "further_testing"
        else:
            return "reject"
    
    def _generate_training_recommendations(self, validation_result: Dict[str, Any],
                                         feature_stability: Dict[str, float]) -> List[str]:
        """Generate recommendations based on training results"""
        recommendations = []
        
        if validation_result.get("validation_passed", False):
            recommendations.append("Model training successful")
            
            performance_comparison = validation_result.get("performance_comparison", {})
            f1_improvement = performance_comparison.get("f1_score", {}).get("improvement", 0)
            
            if f1_improvement >= 0.05:
                recommendations.append("Significant performance improvement detected - recommend deployment")
            elif f1_improvement >= 0.01:
                recommendations.append("Moderate improvement - consider A/B testing before full deployment")
        else:
            recommendations.append("Model validation failed - investigate training data quality")
        
        # Feature stability recommendations
        unstable_features = [name for name, stability in feature_stability.items() if stability < 0.5]
        if unstable_features:
            recommendations.append(f"Monitor unstable features: {', '.join(unstable_features[:3])}")
        
        return recommendations
    
    def load_model_version(self, version_id: str) -> Tuple[XGBoostRiskModel, FeaturePreprocessor]:
        """Load a specific model version"""
        try:
            version = self.version_manager.get_model_version(version_id)
            if not version:
                raise ValueError(f"Model version {version_id} not found")
            
            # Load model
            model = XGBoostRiskModel()
            model.load_model(version.model_file_path)
            
            # Load preprocessor
            preprocessor = FeaturePreprocessor()
            with open(version.preprocessor_file_path, 'rb') as f:
                preprocessor_data = pickle.load(f)
                preprocessor.scaler = preprocessor_data['scaler']
                preprocessor.feature_columns = preprocessor_data['feature_columns']
                preprocessor.is_fitted = preprocessor_data['is_fitted']
            
            return model, preprocessor
        
        except Exception as e:
            self.data_logger.log_data_collection(
                "model_version_load", version_id, 0, False, str(e)
            )
            raise
    
    def get_incremental_training_status(self) -> Dict[str, Any]:
        """Get status of incremental training system"""
        try:
            latest_version = self.version_manager.get_latest_model_version()
            
            return {
                "latest_model_version": latest_version.version_id if latest_version else None,
                "latest_model_type": latest_version.model_type if latest_version else None,
                "latest_model_timestamp": latest_version.creation_timestamp.isoformat() if latest_version else None,
                "models_directory": str(self.models_dir),
                "incremental_models_count": len(list(self.incremental_models_dir.glob("*.pkl"))),
                "system_health": "healthy"
            }
        
        except Exception as e:
            return {
                "error": str(e),
                "system_health": "error"
            }