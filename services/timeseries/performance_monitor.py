"""
Performance Monitoring System for AI Risk Oracle Model
Tracks model performance over time and detects degradation
"""

import asyncio
import pandas as pd
import numpy as np
import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import time
import sqlite3
from threading import Lock

from config import get_config
from logging_config import get_data_logger, get_performance_logger
from services.storage.data_manager import DataManager
from services.timeseries.ml_trainer import XGBoostRiskModel, BacktestResult, ModelPrediction

@dataclass
class PerformanceMetric:
    """Individual performance metric measurement"""
    timestamp: datetime
    metric_name: str
    metric_value: float
    model_version: str
    data_period_start: datetime
    data_period_end: datetime
    sample_size: int
    confidence_interval: Optional[Tuple[float, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DegradationAlert:
    """Alert for model performance degradation"""
    alert_id: str
    timestamp: datetime
    alert_type: str  # "degradation", "critical_degradation", "data_drift"
    severity: str    # "low", "medium", "high", "critical"
    model_version: str
    affected_metrics: List[str]
    current_performance: Dict[str, float]
    baseline_performance: Dict[str, float]
    degradation_percentage: float
    recommended_actions: List[str]
    alert_message: str
    acknowledged: bool = False
    resolved: bool = False

@dataclass
class PerformanceReport:
    """Comprehensive performance monitoring report"""
    report_id: str
    generation_timestamp: datetime
    model_version: str
    monitoring_period_start: datetime
    monitoring_period_end: datetime
    
    # Current performance metrics
    current_metrics: Dict[str, float]
    baseline_metrics: Dict[str, float]
    performance_trends: Dict[str, List[float]]
    
    # Degradation analysis
    degradation_detected: bool
    degradation_metrics: List[str]
    degradation_severity: str
    
    # Alerts and recommendations
    active_alerts: List[DegradationAlert]
    recommendations: List[str]
    
    # Statistical analysis
    statistical_significance: Dict[str, bool]
    confidence_intervals: Dict[str, Tuple[float, float]]
    
    # Metadata
    total_predictions_monitored: int
    monitoring_data_quality: float

class PerformanceDatabase:
    """Database for storing performance monitoring data"""
    
    def __init__(self, db_path: str = "data/performance_monitoring.db"):
        self.db_path = db_path
        self.lock = Lock()
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize performance monitoring database"""
        try:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Performance metrics table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        metric_name TEXT NOT NULL,
                        metric_value REAL NOT NULL,
                        model_version TEXT NOT NULL,
                        data_period_start TEXT NOT NULL,
                        data_period_end TEXT NOT NULL,
                        sample_size INTEGER NOT NULL,
                        confidence_lower REAL,
                        confidence_upper REAL,
                        metadata TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Degradation alerts table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS degradation_alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        alert_id TEXT UNIQUE NOT NULL,
                        timestamp TEXT NOT NULL,
                        alert_type TEXT NOT NULL,
                        severity TEXT NOT NULL,
                        model_version TEXT NOT NULL,
                        affected_metrics TEXT NOT NULL,
                        current_performance TEXT NOT NULL,
                        baseline_performance TEXT NOT NULL,
                        degradation_percentage REAL NOT NULL,
                        recommended_actions TEXT NOT NULL,
                        alert_message TEXT NOT NULL,
                        acknowledged INTEGER DEFAULT 0,
                        resolved INTEGER DEFAULT 0,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Model predictions log table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS prediction_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        asset_symbol TEXT NOT NULL,
                        model_version TEXT NOT NULL,
                        risk_score REAL NOT NULL,
                        confidence REAL NOT NULL,
                        actual_outcome INTEGER,  -- 1 for risk event, 0 for normal, NULL for unknown
                        prediction_correct INTEGER,  -- 1 for correct, 0 for incorrect, NULL for unknown
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create indexes for better performance
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON performance_metrics(timestamp)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_model ON performance_metrics(model_version)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON degradation_alerts(timestamp)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON prediction_log(timestamp)")
                
                conn.commit()
        
        except Exception as e:
            print(f"Error initializing performance database: {e}")
            raise
    
    def store_performance_metric(self, metric: PerformanceMetric):
        """Store a performance metric"""
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO performance_metrics 
                        (timestamp, metric_name, metric_value, model_version, 
                         data_period_start, data_period_end, sample_size,
                         confidence_lower, confidence_upper, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        metric.timestamp.isoformat(),
                        metric.metric_name,
                        metric.metric_value,
                        metric.model_version,
                        metric.data_period_start.isoformat(),
                        metric.data_period_end.isoformat(),
                        metric.sample_size,
                        metric.confidence_interval[0] if metric.confidence_interval else None,
                        metric.confidence_interval[1] if metric.confidence_interval else None,
                        json.dumps(metric.metadata)
                    ))
                    conn.commit()
            except Exception as e:
                print(f"Error storing performance metric: {e}")
                raise
    
    def store_degradation_alert(self, alert: DegradationAlert):
        """Store a degradation alert"""
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT OR REPLACE INTO degradation_alerts 
                        (alert_id, timestamp, alert_type, severity, model_version,
                         affected_metrics, current_performance, baseline_performance,
                         degradation_percentage, recommended_actions, alert_message,
                         acknowledged, resolved)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        alert.alert_id,
                        alert.timestamp.isoformat(),
                        alert.alert_type,
                        alert.severity,
                        alert.model_version,
                        json.dumps(alert.affected_metrics),
                        json.dumps(alert.current_performance),
                        json.dumps(alert.baseline_performance),
                        alert.degradation_percentage,
                        json.dumps(alert.recommended_actions),
                        alert.alert_message,
                        1 if alert.acknowledged else 0,
                        1 if alert.resolved else 0
                    ))
                    conn.commit()
            except Exception as e:
                print(f"Error storing degradation alert: {e}")
                raise
    
    def log_prediction(self, prediction: ModelPrediction, actual_outcome: Optional[int] = None):
        """Log a model prediction for monitoring"""
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # Determine if prediction was correct (if we have actual outcome)
                    prediction_correct = None
                    if actual_outcome is not None:
                        predicted_risk = 1 if prediction.risk_score > 0.5 else 0
                        prediction_correct = 1 if predicted_risk == actual_outcome else 0
                    
                    cursor.execute("""
                        INSERT INTO prediction_log 
                        (timestamp, asset_symbol, model_version, risk_score, confidence,
                         actual_outcome, prediction_correct)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        prediction.prediction_timestamp.isoformat(),
                        prediction.asset_symbol,
                        prediction.model_version,
                        prediction.risk_score,
                        prediction.confidence,
                        actual_outcome,
                        prediction_correct
                    ))
                    conn.commit()
            except Exception as e:
                print(f"Error logging prediction: {e}")
                raise
    
    def get_recent_metrics(self, model_version: str, metric_names: List[str], 
                          days_back: int = 30) -> List[PerformanceMetric]:
        """Get recent performance metrics"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT timestamp, metric_name, metric_value, model_version,
                           data_period_start, data_period_end, sample_size,
                           confidence_lower, confidence_upper, metadata
                    FROM performance_metrics
                    WHERE model_version = ? AND metric_name IN ({})
                    AND timestamp >= ?
                    ORDER BY timestamp DESC
                """.format(','.join('?' * len(metric_names))), 
                [model_version] + metric_names + [cutoff_date.isoformat()])
                
                metrics = []
                for row in cursor.fetchall():
                    confidence_interval = None
                    if row[7] is not None and row[8] is not None:
                        confidence_interval = (row[7], row[8])
                    
                    metadata = json.loads(row[9]) if row[9] else {}
                    
                    metrics.append(PerformanceMetric(
                        timestamp=datetime.fromisoformat(row[0]),
                        metric_name=row[1],
                        metric_value=row[2],
                        model_version=row[3],
                        data_period_start=datetime.fromisoformat(row[4]),
                        data_period_end=datetime.fromisoformat(row[5]),
                        sample_size=row[6],
                        confidence_interval=confidence_interval,
                        metadata=metadata
                    ))
                
                return metrics
        
        except Exception as e:
            print(f"Error getting recent metrics: {e}")
            return []
    
    def get_active_alerts(self, model_version: Optional[str] = None) -> List[DegradationAlert]:
        """Get active degradation alerts"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                query = """
                    SELECT alert_id, timestamp, alert_type, severity, model_version,
                           affected_metrics, current_performance, baseline_performance,
                           degradation_percentage, recommended_actions, alert_message,
                           acknowledged, resolved
                    FROM degradation_alerts
                    WHERE resolved = 0
                """
                params = []
                
                if model_version:
                    query += " AND model_version = ?"
                    params.append(model_version)
                
                query += " ORDER BY timestamp DESC"
                
                cursor.execute(query, params)
                
                alerts = []
                for row in cursor.fetchall():
                    alerts.append(DegradationAlert(
                        alert_id=row[0],
                        timestamp=datetime.fromisoformat(row[1]),
                        alert_type=row[2],
                        severity=row[3],
                        model_version=row[4],
                        affected_metrics=json.loads(row[5]),
                        current_performance=json.loads(row[6]),
                        baseline_performance=json.loads(row[7]),
                        degradation_percentage=row[8],
                        recommended_actions=json.loads(row[9]),
                        alert_message=row[10],
                        acknowledged=bool(row[11]),
                        resolved=bool(row[12])
                    ))
                
                return alerts
        
        except Exception as e:
            print(f"Error getting active alerts: {e}")
            return []

class ModelPerformanceMonitor:
    """Main performance monitoring system"""
    
    def __init__(self, db_path: str = "data/performance_monitoring.db"):
        self.config = get_config()
        self.data_logger = get_data_logger()
        self.performance_logger = get_performance_logger()
        
        # Initialize database
        self.db = PerformanceDatabase(db_path)
        
        # Monitoring configuration
        self.monitoring_config = {
            "baseline_window_days": 30,  # Days to use for baseline calculation
            "degradation_threshold": 0.05,  # 5% degradation threshold
            "critical_degradation_threshold": 0.15,  # 15% critical threshold
            "minimum_sample_size": 100,  # Minimum predictions for reliable metrics
            "confidence_level": 0.95,  # Statistical confidence level
            "monitoring_frequency_hours": 6,  # How often to run monitoring
        }
        
        # Key metrics to monitor
        self.key_metrics = [
            "accuracy", "precision", "recall", "f1_score", "auc_score",
            "false_positive_rate", "false_negative_rate", "prediction_latency"
        ]
    
    def calculate_current_performance(self, model: XGBoostRiskModel, 
                                    recent_predictions: List[ModelPrediction],
                                    actual_outcomes: List[int]) -> Dict[str, float]:
        """Calculate current model performance metrics"""
        try:
            if len(recent_predictions) != len(actual_outcomes):
                raise ValueError("Predictions and outcomes must have same length")
            
            if len(recent_predictions) < self.monitoring_config["minimum_sample_size"]:
                self.data_logger.log_data_collection(
                    "performance_monitoring", "insufficient_data", 0, False,
                    f"Only {len(recent_predictions)} predictions available, need {self.monitoring_config['minimum_sample_size']}"
                )
                return {}
            
            # Convert predictions to binary classifications
            predicted_labels = [1 if p.risk_score > 0.5 else 0 for p in recent_predictions]
            
            # Calculate confusion matrix components
            true_positives = sum(1 for pred, actual in zip(predicted_labels, actual_outcomes) 
                               if pred == 1 and actual == 1)
            false_positives = sum(1 for pred, actual in zip(predicted_labels, actual_outcomes) 
                                if pred == 1 and actual == 0)
            true_negatives = sum(1 for pred, actual in zip(predicted_labels, actual_outcomes) 
                               if pred == 0 and actual == 0)
            false_negatives = sum(1 for pred, actual in zip(predicted_labels, actual_outcomes) 
                                if pred == 0 and actual == 1)
            
            # Calculate metrics
            total = len(predicted_labels)
            accuracy = (true_positives + true_negatives) / total if total > 0 else 0
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            false_positive_rate = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0
            false_negative_rate = false_negatives / (false_negatives + true_positives) if (false_negatives + true_positives) > 0 else 0
            
            # Calculate AUC if we have both classes
            auc_score = 0.5  # Default neutral AUC
            if len(set(actual_outcomes)) > 1:
                try:
                    from sklearn.metrics import roc_auc_score
                    risk_scores = [p.risk_score for p in recent_predictions]
                    auc_score = roc_auc_score(actual_outcomes, risk_scores)
                except Exception:
                    pass  # Keep default AUC
            
            # Calculate average prediction latency (if available)
            prediction_latency = 0.0
            if hasattr(recent_predictions[0], 'prediction_latency_ms'):
                latencies = [p.prediction_latency_ms for p in recent_predictions 
                           if hasattr(p, 'prediction_latency_ms') and p.prediction_latency_ms is not None]
                prediction_latency = np.mean(latencies) if latencies else 0.0
            
            return {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "auc_score": auc_score,
                "false_positive_rate": false_positive_rate,
                "false_negative_rate": false_negative_rate,
                "prediction_latency": prediction_latency,
                "sample_size": total
            }
        
        except Exception as e:
            self.data_logger.log_data_collection(
                "performance_calculation", "current", 0, False, str(e)
            )
            return {}
    
    def get_baseline_performance(self, model_version: str) -> Dict[str, float]:
        """Get baseline performance metrics for comparison"""
        try:
            # Get metrics from the baseline window
            baseline_metrics = self.db.get_recent_metrics(
                model_version, 
                self.key_metrics,
                self.monitoring_config["baseline_window_days"]
            )
            
            if not baseline_metrics:
                self.data_logger.log_data_collection(
                    "baseline_performance", model_version, 0, False,
                    "No baseline metrics found"
                )
                return {}
            
            # Calculate average baseline performance for each metric
            baseline_performance = {}
            for metric_name in self.key_metrics:
                metric_values = [m.metric_value for m in baseline_metrics if m.metric_name == metric_name]
                if metric_values:
                    baseline_performance[metric_name] = np.mean(metric_values)
            
            return baseline_performance
        
        except Exception as e:
            self.data_logger.log_data_collection(
                "baseline_performance", model_version, 0, False, str(e)
            )
            return {}
    
    def detect_performance_degradation(self, current_performance: Dict[str, float],
                                     baseline_performance: Dict[str, float],
                                     model_version: str) -> List[DegradationAlert]:
        """Detect performance degradation and generate alerts"""
        alerts = []
        
        try:
            degraded_metrics = []
            critical_degraded_metrics = []
            
            for metric_name in self.key_metrics:
                if metric_name not in current_performance or metric_name not in baseline_performance:
                    continue
                
                current_value = current_performance[metric_name]
                baseline_value = baseline_performance[metric_name]
                
                if baseline_value == 0:
                    continue  # Skip if baseline is zero
                
                # Calculate degradation percentage
                degradation = (baseline_value - current_value) / baseline_value
                
                # Check for degradation (positive degradation means performance got worse)
                if degradation > self.monitoring_config["degradation_threshold"]:
                    degraded_metrics.append(metric_name)
                    
                    if degradation > self.monitoring_config["critical_degradation_threshold"]:
                        critical_degraded_metrics.append(metric_name)
            
            # Generate alerts
            if critical_degraded_metrics:
                alert = self._create_degradation_alert(
                    "critical_degradation", "critical", model_version,
                    critical_degraded_metrics, current_performance, baseline_performance
                )
                alerts.append(alert)
            
            elif degraded_metrics:
                alert = self._create_degradation_alert(
                    "degradation", "medium", model_version,
                    degraded_metrics, current_performance, baseline_performance
                )
                alerts.append(alert)
            
            return alerts
        
        except Exception as e:
            self.data_logger.log_data_collection(
                "degradation_detection", model_version, 0, False, str(e)
            )
            return []
    
    def _create_degradation_alert(self, alert_type: str, severity: str, model_version: str,
                                affected_metrics: List[str], current_performance: Dict[str, float],
                                baseline_performance: Dict[str, float]) -> DegradationAlert:
        """Create a degradation alert"""
        
        # Calculate overall degradation percentage
        degradations = []
        for metric in affected_metrics:
            if metric in current_performance and metric in baseline_performance:
                baseline_val = baseline_performance[metric]
                if baseline_val > 0:
                    degradation = (baseline_val - current_performance[metric]) / baseline_val
                    degradations.append(degradation)
        
        avg_degradation = np.mean(degradations) if degradations else 0.0
        
        # Generate recommendations
        recommendations = self._generate_recommendations(alert_type, affected_metrics, avg_degradation)
        
        # Create alert message
        alert_message = self._generate_alert_message(alert_type, affected_metrics, avg_degradation)
        
        alert_id = f"{alert_type}_{model_version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return DegradationAlert(
            alert_id=alert_id,
            timestamp=datetime.now(),
            alert_type=alert_type,
            severity=severity,
            model_version=model_version,
            affected_metrics=affected_metrics,
            current_performance=current_performance,
            baseline_performance=baseline_performance,
            degradation_percentage=avg_degradation * 100,
            recommended_actions=recommendations,
            alert_message=alert_message
        )
    
    def _generate_recommendations(self, alert_type: str, affected_metrics: List[str], 
                                degradation: float) -> List[str]:
        """Generate recommendations based on degradation type and severity"""
        recommendations = []
        
        if alert_type == "critical_degradation":
            recommendations.extend([
                "Immediately trigger model retraining with latest data",
                "Review recent market conditions for significant changes",
                "Consider switching to backup model if available",
                "Increase monitoring frequency to hourly",
                "Alert system administrators"
            ])
        
        elif alert_type == "degradation":
            recommendations.extend([
                "Schedule model retraining within 24 hours",
                "Collect additional recent training data",
                "Review feature importance for changes",
                "Increase monitoring frequency"
            ])
        
        # Metric-specific recommendations
        if "accuracy" in affected_metrics or "f1_score" in affected_metrics:
            recommendations.append("Review threshold optimization settings")
        
        if "false_positive_rate" in affected_metrics:
            recommendations.append("Consider increasing risk threshold to reduce false alarms")
        
        if "false_negative_rate" in affected_metrics:
            recommendations.append("Consider decreasing risk threshold to catch more events")
        
        if "prediction_latency" in affected_metrics:
            recommendations.append("Review system performance and resource allocation")
        
        return recommendations
    
    def _generate_alert_message(self, alert_type: str, affected_metrics: List[str], 
                              degradation: float) -> str:
        """Generate human-readable alert message"""
        severity_text = "CRITICAL" if alert_type == "critical_degradation" else "WARNING"
        metrics_text = ", ".join(affected_metrics)
        
        return (f"{severity_text}: Model performance degradation detected. "
                f"Affected metrics: {metrics_text}. "
                f"Average degradation: {degradation:.1%}. "
                f"Immediate attention required.")
    
    def store_performance_metrics(self, model_version: str, performance_metrics: Dict[str, float],
                                data_period_start: datetime, data_period_end: datetime,
                                sample_size: int):
        """Store performance metrics in database"""
        try:
            timestamp = datetime.now()
            
            for metric_name, metric_value in performance_metrics.items():
                if metric_name in self.key_metrics:
                    metric = PerformanceMetric(
                        timestamp=timestamp,
                        metric_name=metric_name,
                        metric_value=metric_value,
                        model_version=model_version,
                        data_period_start=data_period_start,
                        data_period_end=data_period_end,
                        sample_size=sample_size
                    )
                    self.db.store_performance_metric(metric)
            
            self.data_logger.log_data_collection(
                "performance_metrics_storage", model_version, len(performance_metrics), True
            )
        
        except Exception as e:
            self.data_logger.log_data_collection(
                "performance_metrics_storage", model_version, 0, False, str(e)
            )
            raise
    
    def run_performance_monitoring(self, model: XGBoostRiskModel,
                                 recent_predictions: List[ModelPrediction],
                                 actual_outcomes: List[int]) -> PerformanceReport:
        """Run complete performance monitoring cycle"""
        start_time = time.time()
        
        try:
            # Calculate current performance
            current_performance = self.calculate_current_performance(
                model, recent_predictions, actual_outcomes
            )
            
            if not current_performance:
                raise ValueError("Could not calculate current performance metrics")
            
            # Get baseline performance
            baseline_performance = self.get_baseline_performance(model.model_version)
            
            # Store current metrics
            data_period_start = min(p.prediction_timestamp for p in recent_predictions)
            data_period_end = max(p.prediction_timestamp for p in recent_predictions)
            
            self.store_performance_metrics(
                model.model_version, current_performance,
                data_period_start, data_period_end,
                len(recent_predictions)
            )
            
            # Detect degradation
            alerts = []
            if baseline_performance:
                alerts = self.detect_performance_degradation(
                    current_performance, baseline_performance, model.model_version
                )
                
                # Store alerts
                for alert in alerts:
                    self.db.store_degradation_alert(alert)
            
            # Generate performance report
            report = self._generate_performance_report(
                model.model_version, current_performance, baseline_performance,
                alerts, data_period_start, data_period_end, len(recent_predictions)
            )
            
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time(
                "performance_monitoring", duration_ms, True
            )
            
            return report
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time(
                "performance_monitoring", duration_ms, False
            )
            self.data_logger.log_data_collection(
                "performance_monitoring", "run_monitoring", 0, False, str(e)
            )
            raise
    
    def _generate_performance_report(self, model_version: str, 
                                   current_performance: Dict[str, float],
                                   baseline_performance: Dict[str, float],
                                   alerts: List[DegradationAlert],
                                   data_period_start: datetime,
                                   data_period_end: datetime,
                                   sample_size: int) -> PerformanceReport:
        """Generate comprehensive performance report"""
        
        # Determine degradation status
        degradation_detected = len(alerts) > 0
        degradation_metrics = []
        degradation_severity = "none"
        
        if alerts:
            for alert in alerts:
                degradation_metrics.extend(alert.affected_metrics)
            degradation_metrics = list(set(degradation_metrics))
            
            # Determine overall severity
            severities = [alert.severity for alert in alerts]
            if "critical" in severities:
                degradation_severity = "critical"
            elif "high" in severities:
                degradation_severity = "high"
            elif "medium" in severities:
                degradation_severity = "medium"
            else:
                degradation_severity = "low"
        
        # Generate recommendations
        recommendations = []
        if degradation_detected:
            for alert in alerts:
                recommendations.extend(alert.recommended_actions)
            recommendations = list(set(recommendations))  # Remove duplicates
        else:
            recommendations = ["Model performance is stable", "Continue regular monitoring"]
        
        # Calculate performance trends (simplified - would need historical data for full implementation)
        performance_trends = {}
        for metric in self.key_metrics:
            if metric in current_performance:
                performance_trends[metric] = [current_performance[metric]]  # Simplified
        
        # Statistical significance (simplified)
        statistical_significance = {}
        confidence_intervals = {}
        for metric in current_performance:
            statistical_significance[metric] = sample_size >= self.monitoring_config["minimum_sample_size"]
            # Simplified confidence interval calculation
            std_error = np.sqrt(current_performance[metric] * (1 - current_performance[metric]) / sample_size)
            margin = 1.96 * std_error  # 95% confidence
            confidence_intervals[metric] = (
                max(0, current_performance[metric] - margin),
                min(1, current_performance[metric] + margin)
            )
        
        report_id = f"perf_report_{model_version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return PerformanceReport(
            report_id=report_id,
            generation_timestamp=datetime.now(),
            model_version=model_version,
            monitoring_period_start=data_period_start,
            monitoring_period_end=data_period_end,
            current_metrics=current_performance,
            baseline_metrics=baseline_performance,
            performance_trends=performance_trends,
            degradation_detected=degradation_detected,
            degradation_metrics=degradation_metrics,
            degradation_severity=degradation_severity,
            active_alerts=alerts,
            recommendations=recommendations,
            statistical_significance=statistical_significance,
            confidence_intervals=confidence_intervals,
            total_predictions_monitored=sample_size,
            monitoring_data_quality=1.0  # Simplified
        )
    
    def get_monitoring_status(self, model_version: str) -> Dict[str, Any]:
        """Get current monitoring status for a model"""
        try:
            # Get recent metrics
            recent_metrics = self.db.get_recent_metrics(model_version, self.key_metrics, 7)
            
            # Get active alerts
            active_alerts = self.db.get_active_alerts(model_version)
            
            # Calculate status
            status = "healthy"
            if any(alert.severity == "critical" for alert in active_alerts):
                status = "critical"
            elif any(alert.severity in ["high", "medium"] for alert in active_alerts):
                status = "degraded"
            elif not recent_metrics:
                status = "no_data"
            
            return {
                "model_version": model_version,
                "status": status,
                "last_monitoring_time": recent_metrics[0].timestamp.isoformat() if recent_metrics else None,
                "active_alerts_count": len(active_alerts),
                "critical_alerts_count": sum(1 for alert in active_alerts if alert.severity == "critical"),
                "recent_metrics_count": len(recent_metrics),
                "monitoring_health": "active" if recent_metrics else "inactive"
            }
        
        except Exception as e:
            self.data_logger.log_data_collection(
                "monitoring_status", model_version, 0, False, str(e)
            )
            return {
                "model_version": model_version,
                "status": "error",
                "error": str(e)
            }