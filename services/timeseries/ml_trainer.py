"""
Machine Learning Model Training for Time-Series Analysis
Implements XGBoost model for fast inference and backtesting framework for model validation
"""

import asyncio
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import time
from pathlib import Path

# ML libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from config import get_config
from logging_config import get_data_logger, get_performance_logger
from services.timeseries.oracle_processor import FeatureSet, TechnicalIndicators
from services.timeseries.correlation_engine import CorrelationPattern

@dataclass
class ModelPrediction:
    """Model prediction result"""
    asset_symbol: str
    prediction_timestamp: datetime
    
    # Risk predictions
    risk_score: float  # 0.0 to 1.0
    price_drop_probability: float  # Probability of 3% drop in 30 minutes
    volatility_prediction: float  # Expected volatility
    
    # Confidence and metadata
    confidence: float  # 0.0 to 1.0
    model_version: str
    feature_importance: Optional[Dict[str, float]] = None
    
    # Supporting data
    input_features: Optional[Dict[str, float]] = None
    correlation_signals: Optional[List[str]] = None
    market_regime: Optional[str] = None  # "normal", "high_volatility", "crisis"

@dataclass
class BacktestResult:
    """Backtesting result for model validation"""
    model_name: str
    test_period_start: datetime
    test_period_end: datetime
    
    # Performance metrics
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    
    # Financial metrics
    total_predictions: int
    correct_predictions: int
    false_positives: int
    false_negatives: int
    
    # Risk-specific metrics
    risk_events_detected: int
    risk_events_missed: int
    false_alarms: int
    
    # Timing metrics
    average_prediction_latency_ms: float
    
    # Additional metadata
    test_data_points: int
    model_version: str
    
    # Optional fields with defaults
    auc_score: Optional[float] = None
    backtest_timestamp: datetime = field(default_factory=datetime.now)

class FeaturePreprocessor:
    """Preprocess features for ML model training"""
    
    def __init__(self):
        self.data_logger = get_data_logger()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = []
        self.is_fitted = False
    
    def prepare_features_from_feature_sets(self, feature_sets: List[FeatureSet]) -> pd.DataFrame:
        """Convert FeatureSet objects to ML-ready DataFrame"""
        try:
            if not feature_sets:
                return pd.DataFrame()
            
            feature_data = []
            
            for fs in feature_sets:
                feature_dict = {
                    'asset_symbol': fs.asset_symbol,
                    'timestamp': fs.timestamp,
                    'current_price': fs.current_price,
                    'current_volume': fs.current_volume,
                    'current_volatility': fs.current_volatility,
                    
                    # Lag features
                    'price_lag_1': fs.price_lag_1 or 0.0,
                    'price_lag_3': fs.price_lag_3 or 0.0,
                    'price_lag_7': fs.price_lag_7 or 0.0,
                    'volume_lag_1': fs.volume_lag_1 or 0.0,
                    'volume_lag_3': fs.volume_lag_3 or 0.0,
                    
                    # Price change features
                    'price_change_1h': fs.price_change_1h or 0.0,
                    'price_change_4h': fs.price_change_4h or 0.0,
                    'price_change_24h': fs.price_change_24h or 0.0,
                    'price_change_7d': fs.price_change_7d or 0.0,
                    
                    # Technical indicators
                    'rsi_normalized': fs.rsi_normalized or 0.5,
                    'macd_normalized': fs.macd_normalized or 0.0,
                    'bb_position': fs.bb_position or 0.5,
                    
                    # Correlation features
                    'btc_correlation': fs.btc_correlation or 0.0,
                    'gold_correlation': fs.gold_correlation or 0.0,
                    'market_correlation': fs.market_correlation or 0.0,
                    
                    # Risk features
                    'volatility_percentile': fs.volatility_percentile or 0.5,
                    'volume_anomaly_score': fs.volume_anomaly_score or 0.0,
                    'price_momentum_score': fs.price_momentum_score or 0.0
                }
                
                feature_data.append(feature_dict)
            
            df = pd.DataFrame(feature_data)
            return df
        
        except Exception as e:
            self.data_logger.log_data_collection("feature_preparation", "feature_sets", 0, False, str(e))
            return pd.DataFrame()
    
    def create_target_labels(self, feature_df: pd.DataFrame, 
                           future_price_data: Dict[str, List[float]],
                           drop_threshold: float = -0.03,
                           time_horizon_minutes: int = 30) -> pd.Series:
        """Create target labels for risk prediction (1 = risk event, 0 = normal)"""
        try:
            labels = []
            
            for _, row in feature_df.iterrows():
                asset = row['asset_symbol']
                timestamp = row['timestamp']
                current_price = row['current_price']
                
                # Look for price drops in the future
                if asset in future_price_data:
                    future_prices = future_price_data[asset]
                    
                    # Check if there's a significant drop within the time horizon
                    min_future_price = min(future_prices) if future_prices else current_price
                    price_drop = (min_future_price - current_price) / current_price
                    
                    # Label as 1 if drop exceeds threshold, 0 otherwise
                    label = 1 if price_drop <= drop_threshold else 0
                else:
                    # Default to no risk if no future data
                    label = 0
                
                labels.append(label)
            
            return pd.Series(labels)
        
        except Exception as e:
            self.data_logger.log_data_collection("label_creation", "target_labels", 0, False, str(e))
            return pd.Series([0] * len(feature_df))
    
    def fit_transform(self, feature_df: pd.DataFrame) -> np.ndarray:
        """Fit preprocessor and transform features"""
        try:
            # Select numeric features only
            numeric_features = feature_df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Remove timestamp and identifier columns
            exclude_cols = ['timestamp', 'asset_symbol']
            numeric_features = [col for col in numeric_features if col not in exclude_cols]
            
            self.feature_columns = numeric_features
            
            if not numeric_features:
                return np.array([])
            
            # Fit and transform
            feature_matrix = feature_df[numeric_features].fillna(0)
            scaled_features = self.scaler.fit_transform(feature_matrix)
            
            self.is_fitted = True
            return scaled_features
        
        except Exception as e:
            self.data_logger.log_data_collection("feature_preprocessing", "fit_transform", 0, False, str(e))
            return np.array([])
    
    def transform(self, feature_df: pd.DataFrame) -> np.ndarray:
        """Transform features using fitted preprocessor"""
        try:
            if not self.is_fitted:
                raise ValueError("Preprocessor not fitted. Call fit_transform first.")
            
            if not self.feature_columns:
                return np.array([])
            
            # Transform using same features as training
            feature_matrix = feature_df[self.feature_columns].fillna(0)
            scaled_features = self.scaler.transform(feature_matrix)
            
            return scaled_features
        
        except Exception as e:
            self.data_logger.log_data_collection("feature_preprocessing", "transform", 0, False, str(e))
            return np.array([])

class XGBoostRiskModel:
    """XGBoost model for fast risk prediction inference"""
    
    def __init__(self, model_params: Optional[Dict] = None):
        self.data_logger = get_data_logger()
        self.performance_logger = get_performance_logger()
        
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not available. Install with: pip install xgboost")
        
        # Default XGBoost parameters optimized for risk prediction
        self.model_params = model_params or {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }
        
        self.model = None
        self.feature_importance = {}
        self.is_trained = False
        self.model_version = f"xgb_risk_v1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        """Train XGBoost model"""
        start_time = time.time()
        
        try:
            if X_train.size == 0 or len(y_train) == 0:
                raise ValueError("Empty training data")
            
            # Create XGBoost model
            self.model = xgb.XGBClassifier(**self.model_params)
            
            # Prepare validation data if provided
            eval_set = None
            if X_val is not None and y_val is not None:
                eval_set = [(X_train, y_train), (X_val, y_val)]
            
            # Train model
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=False
            )
            
            # Store feature importance
            if feature_names and hasattr(self.model, 'feature_importances_'):
                self.feature_importance = dict(zip(feature_names, self.model.feature_importances_))
            
            self.is_trained = True
            
            # Calculate training metrics
            train_pred = self.model.predict(X_train)
            train_pred_proba = self.model.predict_proba(X_train)[:, 1]
            
            metrics = {
                'train_accuracy': accuracy_score(y_train, train_pred),
                'train_precision': precision_score(y_train, train_pred, zero_division=0),
                'train_recall': recall_score(y_train, train_pred, zero_division=0),
                'train_f1': f1_score(y_train, train_pred, zero_division=0),
                'train_auc': roc_auc_score(y_train, train_pred_proba) if len(np.unique(y_train)) > 1 else 0.5
            }
            
            # Validation metrics if available
            if X_val is not None and y_val is not None:
                val_pred = self.model.predict(X_val)
                val_pred_proba = self.model.predict_proba(X_val)[:, 1]
                
                metrics.update({
                    'val_accuracy': accuracy_score(y_val, val_pred),
                    'val_precision': precision_score(y_val, val_pred, zero_division=0),
                    'val_recall': recall_score(y_val, val_pred, zero_division=0),
                    'val_f1': f1_score(y_val, val_pred, zero_division=0),
                    'val_auc': roc_auc_score(y_val, val_pred_proba) if len(np.unique(y_val)) > 1 else 0.5
                })
            
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time("xgboost_training", duration_ms, True)
            
            return metrics
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time("xgboost_training", duration_ms, False)
            self.data_logger.log_data_collection("model_training", "xgboost", 0, False, str(e))
            raise
    
    def predict(self, X: np.ndarray, return_probabilities: bool = True) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Make predictions"""
        start_time = time.time()
        
        try:
            if not self.is_trained or self.model is None:
                raise ValueError("Model not trained. Call train() first.")
            
            if X.size == 0:
                return np.array([]), np.array([])
            
            # Get predictions
            predictions = self.model.predict(X)
            
            if return_probabilities:
                probabilities = self.model.predict_proba(X)[:, 1]  # Probability of positive class
                
                duration_ms = (time.time() - start_time) * 1000
                self.performance_logger.log_inference_time("xgboost_prediction", duration_ms, True)
                
                return predictions, probabilities
            else:
                duration_ms = (time.time() - start_time) * 1000
                self.performance_logger.log_inference_time("xgboost_prediction", duration_ms, True)
                
                return predictions
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time("xgboost_prediction", duration_ms, False)
            self.data_logger.log_data_collection("model_prediction", "xgboost", 0, False, str(e))
            raise
    
    def save_model(self, filepath: str):
        """Save trained model to disk"""
        try:
            if not self.is_trained:
                raise ValueError("No trained model to save")
            
            model_data = {
                'model': self.model,
                'feature_importance': self.feature_importance,
                'model_version': self.model_version,
                'model_params': self.model_params,
                'trained_timestamp': datetime.now().isoformat()
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            self.data_logger.log_data_collection("model_save", filepath, 1, True)
        
        except Exception as e:
            self.data_logger.log_data_collection("model_save", filepath, 0, False, str(e))
            raise
    
    def load_model(self, filepath: str):
        """Load trained model from disk"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.feature_importance = model_data.get('feature_importance', {})
            self.model_version = model_data.get('model_version', 'unknown')
            self.model_params = model_data.get('model_params', {})
            self.is_trained = True
            
            self.data_logger.log_data_collection("model_load", filepath, 1, True)
        
        except Exception as e:
            self.data_logger.log_data_collection("model_load", filepath, 0, False, str(e))
            raise

class BacktestingFramework:
    """Framework for backtesting model performance on historical data"""
    
    def __init__(self):
        self.data_logger = get_data_logger()
        self.performance_logger = get_performance_logger()
    
    def run_backtest(self, model: XGBoostRiskModel, 
                    historical_features: List[FeatureSet],
                    historical_outcomes: Dict[str, List[float]],
                    test_start_date: datetime,
                    test_end_date: datetime,
                    drop_threshold: float = -0.03) -> BacktestResult:
        """Run comprehensive backtest on historical data"""
        start_time = time.time()
        
        try:
            if not model.is_trained:
                raise ValueError("Model must be trained before backtesting")
            
            # Prepare test data
            preprocessor = FeaturePreprocessor()
            feature_df = preprocessor.prepare_features_from_feature_sets(historical_features)
            
            if feature_df.empty:
                raise ValueError("No features available for backtesting")
            
            # Filter data by test period
            test_mask = (feature_df['timestamp'] >= test_start_date) & (feature_df['timestamp'] <= test_end_date)
            test_features = feature_df[test_mask]
            
            if test_features.empty:
                raise ValueError("No test data in specified period")
            
            # Create labels
            test_labels = preprocessor.create_target_labels(test_features, historical_outcomes, drop_threshold)
            
            # Check if we have valid numeric features before fitting
            numeric_features = test_features.select_dtypes(include=[np.number]).columns.tolist()
            exclude_cols = ['timestamp', 'asset_symbol']
            numeric_features = [col for col in numeric_features if col not in exclude_cols]
            
            if not numeric_features or len(test_features) == 0:
                raise ValueError("No valid numeric features for testing")
            
            # Fit the preprocessor on the test features (for backtesting, we fit on available data)
            X_test = preprocessor.fit_transform(test_features)
            y_test = test_labels.values
            
            if X_test.size == 0:
                raise ValueError("No valid features for testing after preprocessing")
            
            # Make predictions
            prediction_start = time.time()
            predictions, probabilities = model.predict(X_test, return_probabilities=True)
            prediction_time = (time.time() - prediction_start) * 1000
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, predictions)
            precision = precision_score(y_test, predictions, zero_division=0)
            recall = recall_score(y_test, predictions, zero_division=0)
            f1 = f1_score(y_test, predictions, zero_division=0)
            
            # AUC score if we have both classes
            auc = None
            if len(np.unique(y_test)) > 1:
                auc = roc_auc_score(y_test, probabilities)
            
            # Calculate confusion matrix components
            true_positives = np.sum((predictions == 1) & (y_test == 1))
            false_positives = np.sum((predictions == 1) & (y_test == 0))
            false_negatives = np.sum((predictions == 0) & (y_test == 1))
            true_negatives = np.sum((predictions == 0) & (y_test == 0))
            
            # Risk-specific metrics
            risk_events_detected = true_positives
            risk_events_missed = false_negatives
            false_alarms = false_positives
            
            # Create backtest result
            result = BacktestResult(
                model_name=model.model_version,
                test_period_start=test_start_date,
                test_period_end=test_end_date,
                accuracy=float(accuracy),
                precision=float(precision),
                recall=float(recall),
                f1_score=float(f1),
                auc_score=float(auc) if auc is not None else None,
                total_predictions=len(predictions),
                correct_predictions=int(true_positives + true_negatives),
                false_positives=int(false_positives),
                false_negatives=int(false_negatives),
                risk_events_detected=int(risk_events_detected),
                risk_events_missed=int(risk_events_missed),
                false_alarms=int(false_alarms),
                average_prediction_latency_ms=float(prediction_time / len(predictions)),
                test_data_points=len(test_features),
                model_version=model.model_version
            )
            
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time("backtesting", duration_ms, True)
            
            return result
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time("backtesting", duration_ms, False)
            self.data_logger.log_data_collection("backtesting", "historical", 0, False, str(e))
            raise
    
    def time_series_cross_validation(self, model_class, 
                                   features: List[FeatureSet],
                                   outcomes: Dict[str, List[float]],
                                   n_splits: int = 5,
                                   test_size_ratio: float = 0.2) -> List[BacktestResult]:
        """Perform time series cross-validation"""
        try:
            # Prepare data
            preprocessor = FeaturePreprocessor()
            feature_df = preprocessor.prepare_features_from_feature_sets(features)
            labels = preprocessor.create_target_labels(feature_df, outcomes)
            
            if feature_df.empty:
                return []
            
            # Sort by timestamp for time series split
            feature_df = feature_df.sort_values('timestamp')
            labels = labels.reindex(feature_df.index)
            
            # Time series split
            tscv = TimeSeriesSplit(n_splits=n_splits, test_size=int(len(feature_df) * test_size_ratio))
            
            results = []
            
            for fold, (train_idx, test_idx) in enumerate(tscv.split(feature_df)):
                try:
                    # Split data
                    train_features = feature_df.iloc[train_idx]
                    test_features = feature_df.iloc[test_idx]
                    train_labels = labels.iloc[train_idx]
                    test_labels = labels.iloc[test_idx]
                    
                    # Prepare features
                    X_train = preprocessor.fit_transform(train_features)
                    X_test = preprocessor.transform(test_features)
                    
                    if X_train.size == 0 or X_test.size == 0:
                        continue
                    
                    # Train model
                    model = model_class()
                    model.train(X_train, train_labels.values)
                    
                    # Create backtest result for this fold
                    test_start = test_features['timestamp'].min()
                    test_end = test_features['timestamp'].max()
                    
                    # Run backtest on this fold
                    fold_result = self.run_backtest(
                        model, 
                        [fs for fs in features if test_start <= fs.timestamp <= test_end],
                        outcomes,
                        test_start,
                        test_end
                    )
                    
                    results.append(fold_result)
                
                except Exception as e:
                    self.data_logger.log_data_collection("cv_fold", f"fold_{fold}", 0, False, str(e))
                    continue
            
            return results
        
        except Exception as e:
            self.data_logger.log_data_collection("time_series_cv", "all_folds", 0, False, str(e))
            return []

class MLModelTrainer:
    """Main ML model trainer combining all components"""
    
    def __init__(self, model_save_dir: str = "models/trained"):
        self.preprocessor = FeaturePreprocessor()
        self.backtesting = BacktestingFramework()
        self.config = get_config()
        self.data_logger = get_data_logger()
        self.performance_logger = get_performance_logger()
        
        # Model storage
        self.model_save_dir = Path(model_save_dir)
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        
        # Current model
        self.current_model = None
    
    def train_risk_prediction_model(self, 
                                  training_features: List[FeatureSet],
                                  training_outcomes: Dict[str, List[float]],
                                  validation_split: float = 0.2,
                                  model_params: Optional[Dict] = None) -> Tuple[XGBoostRiskModel, Dict[str, float]]:
        """Train risk prediction model with validation"""
        start_time = time.time()
        
        try:
            if not training_features:
                raise ValueError("No training features provided")
            
            # Prepare features
            feature_df = self.preprocessor.prepare_features_from_feature_sets(training_features)
            labels = self.preprocessor.create_target_labels(feature_df, training_outcomes)
            
            if feature_df.empty:
                raise ValueError("No valid features for training")
            
            # Split data
            X = self.preprocessor.fit_transform(feature_df)
            y = labels.values
            
            if X.size == 0:
                raise ValueError("No valid feature matrix")
            
            # Train/validation split
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
            )
            
            # Create and train model
            model = XGBoostRiskModel(model_params)
            metrics = model.train(X_train, y_train, X_val, y_val, self.preprocessor.feature_columns)
            
            # Save model
            model_filename = f"risk_model_{model.model_version}.pkl"
            model_path = self.model_save_dir / model_filename
            model.save_model(str(model_path))
            
            self.current_model = model
            
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time("model_training_complete", duration_ms, True)
            
            return model, metrics
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time("model_training_complete", duration_ms, False)
            self.data_logger.log_data_collection("model_training", "risk_prediction", 0, False, str(e))
            raise
    
    def validate_model_performance(self, 
                                 model: XGBoostRiskModel,
                                 validation_features: List[FeatureSet],
                                 validation_outcomes: Dict[str, List[float]],
                                 validation_period_start: datetime,
                                 validation_period_end: datetime) -> BacktestResult:
        """Validate model performance on historical data"""
        return self.backtesting.run_backtest(
            model, validation_features, validation_outcomes,
            validation_period_start, validation_period_end
        )
    
    def create_prediction(self, 
                         model: XGBoostRiskModel,
                         feature_set: FeatureSet,
                         correlation_patterns: Optional[List[CorrelationPattern]] = None) -> ModelPrediction:
        """Create a model prediction from feature set"""
        try:
            # Prepare single feature set
            feature_df = self.preprocessor.prepare_features_from_feature_sets([feature_set])
            
            if feature_df.empty:
                raise ValueError("No valid features for prediction")
            
            # Transform features
            X = self.preprocessor.transform(feature_df)
            
            if X.size == 0:
                raise ValueError("No valid feature matrix for prediction")
            
            # Make prediction
            predictions, probabilities = model.predict(X, return_probabilities=True)
            
            # Extract results
            risk_score = float(probabilities[0])
            price_drop_probability = risk_score  # Same as risk score for binary classification
            
            # Estimate volatility from features
            volatility_prediction = feature_set.current_volatility or 0.0
            
            # Calculate confidence based on feature quality
            confidence = self._calculate_prediction_confidence(feature_set, risk_score)
            
            # Extract correlation signals
            correlation_signals = []
            if correlation_patterns:
                correlation_signals = [p.pattern_type for p in correlation_patterns if p.confidence_score > 0.7]
            
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
                model_version=model.model_version,
                feature_importance=model.feature_importance,
                input_features={
                    'current_price': feature_set.current_price,
                    'current_volatility': feature_set.current_volatility,
                    'price_change_24h': feature_set.price_change_24h,
                    'rsi_normalized': feature_set.rsi_normalized,
                    'btc_correlation': feature_set.btc_correlation
                },
                correlation_signals=correlation_signals,
                market_regime=market_regime
            )
            
            return prediction
        
        except Exception as e:
            self.data_logger.log_data_collection("model_prediction", feature_set.asset_symbol, 0, False, str(e))
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
            confidence = np.mean(confidence_factors)
            return float(np.clip(confidence, 0.0, 1.0))
        
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
    
    def load_model(self, model_path: str) -> XGBoostRiskModel:
        """Load a trained model"""
        model = XGBoostRiskModel()
        model.load_model(model_path)
        self.current_model = model
        return model
    
    def get_model_performance_summary(self, backtest_results: List[BacktestResult]) -> Dict[str, float]:
        """Get summary statistics from multiple backtest results"""
        if not backtest_results:
            return {}
        
        return {
            'avg_accuracy': np.mean([r.accuracy for r in backtest_results]),
            'avg_precision': np.mean([r.precision for r in backtest_results]),
            'avg_recall': np.mean([r.recall for r in backtest_results]),
            'avg_f1_score': np.mean([r.f1_score for r in backtest_results]),
            'avg_auc_score': np.mean([r.auc_score for r in backtest_results if r.auc_score is not None]),
            'total_predictions': sum(r.total_predictions for r in backtest_results),
            'total_correct': sum(r.correct_predictions for r in backtest_results),
            'avg_latency_ms': np.mean([r.average_prediction_latency_ms for r in backtest_results])
        }