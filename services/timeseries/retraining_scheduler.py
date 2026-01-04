"""
Automated Retraining Scheduler for AI Risk Oracle Model
Handles scheduled retraining, data freshness checking, and retraining decision logic
"""

import asyncio
import pandas as pd
import numpy as np
import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import time
import sqlite3
from threading import Lock, Thread
import schedule
from enum import Enum

from config import get_config
from logging_config import get_data_logger, get_performance_logger
from services.storage.data_manager import DataManager
from services.timeseries.performance_monitor import ModelPerformanceMonitor, DegradationAlert
from services.timeseries.training_pipeline import TrainingPipeline
from services.timeseries.ml_trainer import XGBoostRiskModel

class RetrainingTrigger(Enum):
    """Types of retraining triggers"""
    SCHEDULED = "scheduled"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    DATA_FRESHNESS = "data_freshness"
    MANUAL = "manual"
    CRITICAL_ALERT = "critical_alert"

@dataclass
class RetrainingRequest:
    """Request for model retraining"""
    request_id: str
    timestamp: datetime
    trigger_type: RetrainingTrigger
    model_version: str
    priority: str  # "low", "medium", "high", "critical"
    reason: str
    requested_by: str  # "system", "user", "alert"
    
    # Retraining parameters
    training_period_months: int = 24
    validation_period_months: int = 6
    incremental: bool = False
    
    # Status tracking
    status: str = "pending"  # "pending", "in_progress", "completed", "failed"
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    
    # Results
    new_model_version: Optional[str] = None
    performance_improvement: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DataFreshnessReport:
    """Report on data freshness for retraining decisions"""
    report_timestamp: datetime
    last_data_update: datetime
    data_age_hours: float
    freshness_score: float  # 0.0 to 1.0
    missing_data_periods: List[Tuple[datetime, datetime]]
    data_quality_score: float
    recommendation: str  # "no_action", "collect_more_data", "trigger_retraining"
    
    # Asset-specific freshness
    asset_freshness: Dict[str, float]
    stale_assets: List[str]

@dataclass
class RetrainingSchedule:
    """Configuration for scheduled retraining"""
    schedule_id: str
    model_version: str
    frequency: str  # "daily", "weekly", "monthly"
    time_of_day: str  # "HH:MM" format
    enabled: bool = True
    
    # Conditions
    minimum_data_age_hours: int = 24
    minimum_performance_threshold: float = 0.8
    
    # Last execution
    last_execution: Optional[datetime] = None
    next_execution: Optional[datetime] = None

class RetrainingDatabase:
    """Database for storing retraining requests and schedules"""
    
    def __init__(self, db_path: str = "data/retraining.db"):
        self.db_path = db_path
        self.lock = Lock()
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize retraining database"""
        try:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Retraining requests table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS retraining_requests (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        request_id TEXT UNIQUE NOT NULL,
                        timestamp TEXT NOT NULL,
                        trigger_type TEXT NOT NULL,
                        model_version TEXT NOT NULL,
                        priority TEXT NOT NULL,
                        reason TEXT NOT NULL,
                        requested_by TEXT NOT NULL,
                        training_period_months INTEGER NOT NULL,
                        validation_period_months INTEGER NOT NULL,
                        incremental INTEGER NOT NULL,
                        status TEXT NOT NULL,
                        started_at TEXT,
                        completed_at TEXT,
                        error_message TEXT,
                        new_model_version TEXT,
                        performance_improvement TEXT,
                        metadata TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Retraining schedules table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS retraining_schedules (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        schedule_id TEXT UNIQUE NOT NULL,
                        model_version TEXT NOT NULL,
                        frequency TEXT NOT NULL,
                        time_of_day TEXT NOT NULL,
                        enabled INTEGER NOT NULL,
                        minimum_data_age_hours INTEGER NOT NULL,
                        minimum_performance_threshold REAL NOT NULL,
                        last_execution TEXT,
                        next_execution TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Data freshness log table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS data_freshness_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        last_data_update TEXT NOT NULL,
                        data_age_hours REAL NOT NULL,
                        freshness_score REAL NOT NULL,
                        data_quality_score REAL NOT NULL,
                        recommendation TEXT NOT NULL,
                        asset_freshness TEXT NOT NULL,
                        stale_assets TEXT NOT NULL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create indexes
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_requests_timestamp ON retraining_requests(timestamp)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_requests_status ON retraining_requests(status)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_schedules_model ON retraining_schedules(model_version)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_freshness_timestamp ON data_freshness_log(timestamp)")
                
                conn.commit()
        
        except Exception as e:
            print(f"Error initializing retraining database: {e}")
            raise
    
    def store_retraining_request(self, request: RetrainingRequest):
        """Store a retraining request"""
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT OR REPLACE INTO retraining_requests 
                        (request_id, timestamp, trigger_type, model_version, priority, reason,
                         requested_by, training_period_months, validation_period_months, incremental,
                         status, started_at, completed_at, error_message, new_model_version,
                         performance_improvement, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        request.request_id,
                        request.timestamp.isoformat(),
                        request.trigger_type.value,
                        request.model_version,
                        request.priority,
                        request.reason,
                        request.requested_by,
                        request.training_period_months,
                        request.validation_period_months,
                        1 if request.incremental else 0,
                        request.status,
                        request.started_at.isoformat() if request.started_at else None,
                        request.completed_at.isoformat() if request.completed_at else None,
                        request.error_message,
                        request.new_model_version,
                        json.dumps(request.performance_improvement) if request.performance_improvement else None,
                        json.dumps(request.metadata)
                    ))
                    conn.commit()
            except Exception as e:
                print(f"Error storing retraining request: {e}")
                raise
    
    def get_pending_requests(self) -> List[RetrainingRequest]:
        """Get pending retraining requests"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT request_id, timestamp, trigger_type, model_version, priority, reason,
                           requested_by, training_period_months, validation_period_months, incremental,
                           status, started_at, completed_at, error_message, new_model_version,
                           performance_improvement, metadata
                    FROM retraining_requests
                    WHERE status = 'pending'
                    ORDER BY priority DESC, timestamp ASC
                """)
                
                requests = []
                for row in cursor.fetchall():
                    performance_improvement = None
                    if row[15]:
                        performance_improvement = json.loads(row[15])
                    
                    metadata = json.loads(row[16]) if row[16] else {}
                    
                    requests.append(RetrainingRequest(
                        request_id=row[0],
                        timestamp=datetime.fromisoformat(row[1]),
                        trigger_type=RetrainingTrigger(row[2]),
                        model_version=row[3],
                        priority=row[4],
                        reason=row[5],
                        requested_by=row[6],
                        training_period_months=row[7],
                        validation_period_months=row[8],
                        incremental=bool(row[9]),
                        status=row[10],
                        started_at=datetime.fromisoformat(row[11]) if row[11] else None,
                        completed_at=datetime.fromisoformat(row[12]) if row[12] else None,
                        error_message=row[13],
                        new_model_version=row[14],
                        performance_improvement=performance_improvement,
                        metadata=metadata
                    ))
                
                return requests
        
        except Exception as e:
            print(f"Error getting pending requests: {e}")
            return []
    
    def store_retraining_schedule(self, schedule: RetrainingSchedule):
        """Store a retraining schedule"""
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT OR REPLACE INTO retraining_schedules 
                        (schedule_id, model_version, frequency, time_of_day, enabled,
                         minimum_data_age_hours, minimum_performance_threshold,
                         last_execution, next_execution)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        schedule.schedule_id,
                        schedule.model_version,
                        schedule.frequency,
                        schedule.time_of_day,
                        1 if schedule.enabled else 0,
                        schedule.minimum_data_age_hours,
                        schedule.minimum_performance_threshold,
                        schedule.last_execution.isoformat() if schedule.last_execution else None,
                        schedule.next_execution.isoformat() if schedule.next_execution else None
                    ))
                    conn.commit()
            except Exception as e:
                print(f"Error storing retraining schedule: {e}")
                raise
    
    def get_active_schedules(self) -> List[RetrainingSchedule]:
        """Get active retraining schedules"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT schedule_id, model_version, frequency, time_of_day, enabled,
                           minimum_data_age_hours, minimum_performance_threshold,
                           last_execution, next_execution
                    FROM retraining_schedules
                    WHERE enabled = 1
                """)
                
                schedules = []
                for row in cursor.fetchall():
                    schedules.append(RetrainingSchedule(
                        schedule_id=row[0],
                        model_version=row[1],
                        frequency=row[2],
                        time_of_day=row[3],
                        enabled=bool(row[4]),
                        minimum_data_age_hours=row[5],
                        minimum_performance_threshold=row[6],
                        last_execution=datetime.fromisoformat(row[7]) if row[7] else None,
                        next_execution=datetime.fromisoformat(row[8]) if row[8] else None
                    ))
                
                return schedules
        
        except Exception as e:
            print(f"Error getting active schedules: {e}")
            return []
    
    def store_data_freshness_report(self, report: DataFreshnessReport):
        """Store a data freshness report"""
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO data_freshness_log 
                        (timestamp, last_data_update, data_age_hours, freshness_score,
                         data_quality_score, recommendation, asset_freshness, stale_assets)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        report.report_timestamp.isoformat(),
                        report.last_data_update.isoformat(),
                        report.data_age_hours,
                        report.freshness_score,
                        report.data_quality_score,
                        report.recommendation,
                        json.dumps(report.asset_freshness),
                        json.dumps(report.stale_assets)
                    ))
                    conn.commit()
            except Exception as e:
                print(f"Error storing data freshness report: {e}")
                raise

class DataFreshnessChecker:
    """Check data freshness for retraining decisions"""
    
    def __init__(self):
        self.config = get_config()
        self.data_logger = get_data_logger()
        self.data_manager = DataManager()
        
        # Freshness thresholds
        self.freshness_config = {
            "max_data_age_hours": 48,  # Data older than 48 hours is considered stale
            "critical_data_age_hours": 72,  # Critical threshold
            "minimum_freshness_score": 0.7,  # Minimum acceptable freshness
            "stale_asset_threshold": 0.5,  # Asset freshness threshold
        }
    
    def check_data_freshness(self, assets: List[str] = None) -> DataFreshnessReport:
        """Check data freshness for specified assets"""
        try:
            if assets is None:
                assets = ["BTC", "ETH", "XRP", "SOL", "QIE", "GOLD", "BNB"]
            
            report_timestamp = datetime.now()
            
            # Get latest data timestamps for each asset
            asset_freshness = {}
            stale_assets = []
            last_data_updates = []
            
            for asset in assets:
                try:
                    # Get latest data timestamp for this asset
                    latest_timestamp = self._get_latest_data_timestamp(asset)
                    
                    if latest_timestamp:
                        data_age_hours = (report_timestamp - latest_timestamp).total_seconds() / 3600
                        
                        # Calculate freshness score (1.0 = fresh, 0.0 = very stale)
                        if data_age_hours <= 1:
                            freshness_score = 1.0
                        elif data_age_hours <= self.freshness_config["max_data_age_hours"]:
                            # Linear decay from 1.0 to 0.5
                            freshness_score = 1.0 - (data_age_hours / self.freshness_config["max_data_age_hours"]) * 0.5
                        else:
                            # Exponential decay after max age
                            excess_hours = data_age_hours - self.freshness_config["max_data_age_hours"]
                            freshness_score = 0.5 * np.exp(-excess_hours / 24)  # Decay with 24h half-life
                        
                        asset_freshness[asset] = freshness_score
                        last_data_updates.append(latest_timestamp)
                        
                        # Check if asset is stale
                        if freshness_score < self.freshness_config["stale_asset_threshold"]:
                            stale_assets.append(asset)
                    
                    else:
                        # No data found for asset
                        asset_freshness[asset] = 0.0
                        stale_assets.append(asset)
                
                except Exception as e:
                    self.data_logger.log_data_collection(
                        "data_freshness_check", asset, 0, False, str(e)
                    )
                    asset_freshness[asset] = 0.0
                    stale_assets.append(asset)
            
            # Calculate overall metrics
            if last_data_updates:
                last_data_update = max(last_data_updates)
                data_age_hours = (report_timestamp - last_data_update).total_seconds() / 3600
            else:
                last_data_update = report_timestamp - timedelta(days=30)  # Very old
                data_age_hours = 720  # 30 days
            
            # Overall freshness score (average of asset scores)
            freshness_score = np.mean(list(asset_freshness.values())) if asset_freshness else 0.0
            
            # Data quality score (simplified - based on number of fresh assets)
            fresh_assets = len([asset for asset, score in asset_freshness.items() 
                              if score >= self.freshness_config["stale_asset_threshold"]])
            data_quality_score = fresh_assets / len(assets) if assets else 0.0
            
            # Generate recommendation
            recommendation = self._generate_freshness_recommendation(
                freshness_score, data_age_hours, len(stale_assets), len(assets)
            )
            
            # Detect missing data periods (simplified)
            missing_data_periods = self._detect_missing_data_periods(assets)
            
            return DataFreshnessReport(
                report_timestamp=report_timestamp,
                last_data_update=last_data_update,
                data_age_hours=data_age_hours,
                freshness_score=freshness_score,
                missing_data_periods=missing_data_periods,
                data_quality_score=data_quality_score,
                recommendation=recommendation,
                asset_freshness=asset_freshness,
                stale_assets=stale_assets
            )
        
        except Exception as e:
            self.data_logger.log_data_collection(
                "data_freshness_check", "overall", 0, False, str(e)
            )
            # Return default report on error
            return DataFreshnessReport(
                report_timestamp=datetime.now(),
                last_data_update=datetime.now() - timedelta(days=1),
                data_age_hours=24.0,
                freshness_score=0.0,
                missing_data_periods=[],
                data_quality_score=0.0,
                recommendation="error_occurred",
                asset_freshness={},
                stale_assets=assets or []
            )
    
    def _get_latest_data_timestamp(self, asset: str) -> Optional[datetime]:
        """Get the latest data timestamp for an asset"""
        try:
            # Query the data manager for latest timestamp
            # This is a simplified implementation - in practice would query the actual database
            
            # Try to get recent data from the last 7 days
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            
            # Check if we have any recent data
            if asset == "GOLD":
                # Traditional asset
                recent_data = self.data_manager.get_traditional_asset_data(
                    asset, start_date, end_date, limit=1
                )
            else:
                # Crypto asset
                recent_data = self.data_manager.get_crypto_price_data(
                    asset, start_date, end_date, limit=1
                )
            
            if recent_data:
                # Get timestamp from the most recent data point
                latest_point = recent_data[0]
                return getattr(latest_point, 'timestamp', None)
            
            return None
        
        except Exception as e:
            self.data_logger.log_data_collection(
                "latest_timestamp_check", asset, 0, False, str(e)
            )
            return None
    
    def _generate_freshness_recommendation(self, freshness_score: float, data_age_hours: float,
                                         stale_assets_count: int, total_assets: int) -> str:
        """Generate recommendation based on data freshness"""
        
        if freshness_score >= self.freshness_config["minimum_freshness_score"]:
            return "no_action"
        
        elif data_age_hours >= self.freshness_config["critical_data_age_hours"]:
            return "trigger_retraining"
        
        elif stale_assets_count > total_assets * 0.5:  # More than half assets are stale
            return "collect_more_data"
        
        elif freshness_score < 0.3:  # Very low freshness
            return "trigger_retraining"
        
        else:
            return "collect_more_data"
    
    def _detect_missing_data_periods(self, assets: List[str]) -> List[Tuple[datetime, datetime]]:
        """Detect periods with missing data (simplified implementation)"""
        missing_periods = []
        
        try:
            # Check for gaps in the last 7 days
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            
            # This is a simplified implementation
            # In practice, would analyze actual data gaps
            
            # For now, return empty list
            return missing_periods
        
        except Exception:
            return missing_periods

class AutomatedRetrainingScheduler:
    """Main automated retraining scheduler"""
    
    def __init__(self, db_path: str = "data/retraining.db"):
        self.config = get_config()
        self.data_logger = get_data_logger()
        self.performance_logger = get_performance_logger()
        
        # Initialize components
        self.db = RetrainingDatabase(db_path)
        self.performance_monitor = ModelPerformanceMonitor()
        self.freshness_checker = DataFreshnessChecker()
        self.training_pipeline = TrainingPipeline()
        
        # Scheduler state
        self.scheduler_running = False
        self.scheduler_thread = None
        
        # Retraining configuration
        self.retraining_config = {
            "max_concurrent_retraining": 1,  # Only one retraining at a time
            "retraining_timeout_hours": 12,  # Maximum time for retraining
            "performance_check_interval_hours": 6,  # How often to check performance
            "freshness_check_interval_hours": 4,  # How often to check data freshness
            "auto_approve_scheduled": True,  # Auto-approve scheduled retraining
            "auto_approve_critical": True,  # Auto-approve critical alerts
        }
        
        # Currently running retraining
        self.active_retraining = None
    
    def start_scheduler(self):
        """Start the automated retraining scheduler"""
        if self.scheduler_running:
            self.data_logger.log_data_collection(
                "retraining_scheduler", "start", 0, False, "Scheduler already running"
            )
            return
        
        self.scheduler_running = True
        
        # Schedule periodic checks
        schedule.every(self.retraining_config["performance_check_interval_hours"]).hours.do(
            self._check_performance_triggers
        )
        
        schedule.every(self.retraining_config["freshness_check_interval_hours"]).hours.do(
            self._check_freshness_triggers
        )
        
        schedule.every(1).hours.do(self._process_pending_requests)
        
        # Start scheduler thread
        self.scheduler_thread = Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        self.data_logger.log_data_collection(
            "retraining_scheduler", "start", 1, True, "Scheduler started successfully"
        )
    
    def stop_scheduler(self):
        """Stop the automated retraining scheduler"""
        self.scheduler_running = False
        
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=5)
        
        schedule.clear()
        
        self.data_logger.log_data_collection(
            "retraining_scheduler", "stop", 1, True, "Scheduler stopped"
        )
    
    def _run_scheduler(self):
        """Main scheduler loop"""
        while self.scheduler_running:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except Exception as e:
                self.data_logger.log_data_collection(
                    "retraining_scheduler", "loop", 0, False, str(e)
                )
                time.sleep(300)  # Wait 5 minutes on error
    
    def _check_performance_triggers(self):
        """Check for performance-based retraining triggers"""
        try:
            # Get current model (simplified - would get from model registry)
            current_model_version = "current"  # Placeholder
            
            # Get active alerts from performance monitor
            active_alerts = self.performance_monitor.db.get_active_alerts(current_model_version)
            
            for alert in active_alerts:
                if alert.severity == "critical" and not alert.acknowledged:
                    # Create critical retraining request
                    request = RetrainingRequest(
                        request_id=f"perf_critical_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        timestamp=datetime.now(),
                        trigger_type=RetrainingTrigger.CRITICAL_ALERT,
                        model_version=alert.model_version,
                        priority="critical",
                        reason=f"Critical performance degradation: {alert.alert_message}",
                        requested_by="system",
                        training_period_months=12,  # Shorter for critical issues
                        validation_period_months=3,
                        incremental=False,  # Full retraining for critical issues
                        metadata={"alert_id": alert.alert_id, "degradation_percentage": alert.degradation_percentage}
                    )
                    
                    self.db.store_retraining_request(request)
                    
                    self.data_logger.log_data_collection(
                        "retraining_trigger", "critical_performance", 1, True,
                        f"Critical retraining triggered for {alert.model_version}"
                    )
                
                elif alert.severity in ["high", "medium"] and not alert.acknowledged:
                    # Create standard retraining request
                    request = RetrainingRequest(
                        request_id=f"perf_degradation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        timestamp=datetime.now(),
                        trigger_type=RetrainingTrigger.PERFORMANCE_DEGRADATION,
                        model_version=alert.model_version,
                        priority="high" if alert.severity == "high" else "medium",
                        reason=f"Performance degradation detected: {alert.alert_message}",
                        requested_by="system",
                        training_period_months=18,
                        validation_period_months=6,
                        incremental=True,  # Try incremental first
                        metadata={"alert_id": alert.alert_id, "degradation_percentage": alert.degradation_percentage}
                    )
                    
                    self.db.store_retraining_request(request)
                    
                    self.data_logger.log_data_collection(
                        "retraining_trigger", "performance_degradation", 1, True,
                        f"Retraining triggered for degradation in {alert.model_version}"
                    )
        
        except Exception as e:
            self.data_logger.log_data_collection(
                "performance_trigger_check", "error", 0, False, str(e)
            )
    
    def _check_freshness_triggers(self):
        """Check for data freshness-based retraining triggers"""
        try:
            # Check data freshness
            freshness_report = self.freshness_checker.check_data_freshness()
            
            # Store freshness report
            self.db.store_data_freshness_report(freshness_report)
            
            # Check if retraining is needed based on freshness
            if freshness_report.recommendation == "trigger_retraining":
                request = RetrainingRequest(
                    request_id=f"freshness_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    timestamp=datetime.now(),
                    trigger_type=RetrainingTrigger.DATA_FRESHNESS,
                    model_version="current",  # Placeholder
                    priority="medium",
                    reason=f"Data freshness below threshold: {freshness_report.freshness_score:.2f}",
                    requested_by="system",
                    training_period_months=24,
                    validation_period_months=6,
                    incremental=True,
                    metadata={
                        "freshness_score": freshness_report.freshness_score,
                        "data_age_hours": freshness_report.data_age_hours,
                        "stale_assets": freshness_report.stale_assets
                    }
                )
                
                self.db.store_retraining_request(request)
                
                self.data_logger.log_data_collection(
                    "retraining_trigger", "data_freshness", 1, True,
                    f"Retraining triggered due to stale data: {freshness_report.freshness_score:.2f}"
                )
        
        except Exception as e:
            self.data_logger.log_data_collection(
                "freshness_trigger_check", "error", 0, False, str(e)
            )
    
    def _process_pending_requests(self):
        """Process pending retraining requests"""
        try:
            # Skip if already retraining
            if self.active_retraining:
                return
            
            # Get pending requests
            pending_requests = self.db.get_pending_requests()
            
            if not pending_requests:
                return
            
            # Get highest priority request
            request = pending_requests[0]  # Already sorted by priority
            
            # Check if we should auto-approve
            should_approve = False
            
            if request.trigger_type == RetrainingTrigger.SCHEDULED and self.retraining_config["auto_approve_scheduled"]:
                should_approve = True
            elif request.trigger_type == RetrainingTrigger.CRITICAL_ALERT and self.retraining_config["auto_approve_critical"]:
                should_approve = True
            elif request.priority == "critical":
                should_approve = True
            
            if should_approve:
                # Start retraining
                self._start_retraining(request)
            else:
                self.data_logger.log_data_collection(
                    "retraining_approval", "pending", 1, True,
                    f"Retraining request {request.request_id} requires manual approval"
                )
        
        except Exception as e:
            self.data_logger.log_data_collection(
                "process_pending_requests", "error", 0, False, str(e)
            )
    
    def _start_retraining(self, request: RetrainingRequest):
        """Start model retraining"""
        try:
            # Update request status
            request.status = "in_progress"
            request.started_at = datetime.now()
            self.db.store_retraining_request(request)
            
            self.active_retraining = request
            
            self.data_logger.log_data_collection(
                "retraining_start", request.request_id, 1, True,
                f"Starting retraining for {request.model_version}"
            )
            
            # Start retraining in background thread
            retraining_thread = Thread(
                target=self._execute_retraining,
                args=(request,),
                daemon=True
            )
            retraining_thread.start()
        
        except Exception as e:
            # Update request with error
            request.status = "failed"
            request.error_message = str(e)
            request.completed_at = datetime.now()
            self.db.store_retraining_request(request)
            
            self.active_retraining = None
            
            self.data_logger.log_data_collection(
                "retraining_start", request.request_id, 0, False, str(e)
            )
    
    def _execute_retraining(self, request: RetrainingRequest):
        """Execute the actual retraining process"""
        try:
            start_time = time.time()
            
            # Run training pipeline
            training_result = asyncio.run(
                self.training_pipeline.run_complete_training_pipeline(
                    training_period_months=request.training_period_months,
                    validation_period_months=request.validation_period_months
                )
            )
            
            # Extract results
            new_model = training_result["model"]
            training_metrics = training_result["training_metrics"]
            backtest_result = training_result["backtest_result"]
            
            # Calculate performance improvement
            performance_improvement = {
                "accuracy_improvement": training_metrics.get("val_accuracy", 0) - 0.8,  # Placeholder baseline
                "f1_improvement": training_metrics.get("val_f1", 0) - 0.7,  # Placeholder baseline
                "training_time_minutes": (time.time() - start_time) / 60
            }
            
            # Update request with success
            request.status = "completed"
            request.completed_at = datetime.now()
            request.new_model_version = new_model.model_version
            request.performance_improvement = performance_improvement
            request.metadata.update({
                "training_metrics": training_metrics,
                "backtest_accuracy": backtest_result.accuracy,
                "backtest_f1": backtest_result.f1_score
            })
            
            self.db.store_retraining_request(request)
            
            self.data_logger.log_data_collection(
                "retraining_complete", request.request_id, 1, True,
                f"Retraining completed successfully: {new_model.model_version}"
            )
        
        except Exception as e:
            # Update request with error
            request.status = "failed"
            request.error_message = str(e)
            request.completed_at = datetime.now()
            self.db.store_retraining_request(request)
            
            self.data_logger.log_data_collection(
                "retraining_execute", request.request_id, 0, False, str(e)
            )
        
        finally:
            self.active_retraining = None
    
    def create_retraining_schedule(self, model_version: str, frequency: str, 
                                 time_of_day: str = "02:00") -> RetrainingSchedule:
        """Create a new retraining schedule"""
        schedule_id = f"schedule_{model_version}_{frequency}_{datetime.now().strftime('%Y%m%d')}"
        
        schedule_obj = RetrainingSchedule(
            schedule_id=schedule_id,
            model_version=model_version,
            frequency=frequency,
            time_of_day=time_of_day,
            enabled=True
        )
        
        # Calculate next execution time
        schedule_obj.next_execution = self._calculate_next_execution(frequency, time_of_day)
        
        self.db.store_retraining_schedule(schedule_obj)
        
        self.data_logger.log_data_collection(
            "retraining_schedule", "create", 1, True,
            f"Created {frequency} schedule for {model_version}"
        )
        
        return schedule_obj
    
    def _calculate_next_execution(self, frequency: str, time_of_day: str) -> datetime:
        """Calculate next execution time for a schedule"""
        try:
            hour, minute = map(int, time_of_day.split(':'))
            now = datetime.now()
            
            if frequency == "daily":
                next_exec = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                if next_exec <= now:
                    next_exec += timedelta(days=1)
            
            elif frequency == "weekly":
                # Schedule for next Monday at specified time
                days_ahead = 0 - now.weekday()  # Monday is 0
                if days_ahead <= 0:  # Target day already happened this week
                    days_ahead += 7
                next_exec = now + timedelta(days=days_ahead)
                next_exec = next_exec.replace(hour=hour, minute=minute, second=0, microsecond=0)
            
            elif frequency == "monthly":
                # Schedule for first day of next month
                if now.month == 12:
                    next_exec = now.replace(year=now.year + 1, month=1, day=1, 
                                          hour=hour, minute=minute, second=0, microsecond=0)
                else:
                    next_exec = now.replace(month=now.month + 1, day=1,
                                          hour=hour, minute=minute, second=0, microsecond=0)
            
            else:
                # Default to daily
                next_exec = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                if next_exec <= now:
                    next_exec += timedelta(days=1)
            
            return next_exec
        
        except Exception:
            # Default to tomorrow at 2 AM
            return datetime.now().replace(hour=2, minute=0, second=0, microsecond=0) + timedelta(days=1)
    
    def trigger_manual_retraining(self, model_version: str, reason: str, 
                                priority: str = "medium", incremental: bool = True) -> RetrainingRequest:
        """Manually trigger model retraining"""
        request = RetrainingRequest(
            request_id=f"manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(),
            trigger_type=RetrainingTrigger.MANUAL,
            model_version=model_version,
            priority=priority,
            reason=reason,
            requested_by="user",
            training_period_months=24,
            validation_period_months=6,
            incremental=incremental
        )
        
        self.db.store_retraining_request(request)
        
        self.data_logger.log_data_collection(
            "manual_retraining", model_version, 1, True,
            f"Manual retraining requested: {reason}"
        )
        
        return request
    
    def get_retraining_status(self) -> Dict[str, Any]:
        """Get current retraining system status"""
        try:
            pending_requests = self.db.get_pending_requests()
            active_schedules = self.db.get_active_schedules()
            
            return {
                "scheduler_running": self.scheduler_running,
                "active_retraining": self.active_retraining.request_id if self.active_retraining else None,
                "pending_requests_count": len(pending_requests),
                "active_schedules_count": len(active_schedules),
                "next_scheduled_retraining": min([s.next_execution for s in active_schedules if s.next_execution], default=None),
                "last_performance_check": datetime.now().isoformat(),  # Simplified
                "last_freshness_check": datetime.now().isoformat(),  # Simplified
                "system_health": "healthy" if self.scheduler_running else "stopped"
            }
        
        except Exception as e:
            return {
                "scheduler_running": self.scheduler_running,
                "error": str(e),
                "system_health": "error"
            }