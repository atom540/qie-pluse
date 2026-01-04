"""
Automated Retraining System Integration
Main interface for the complete automated retraining system
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path

from config import get_config
from logging_config import get_data_logger, get_performance_logger
from services.timeseries.performance_monitor import ModelPerformanceMonitor, PerformanceReport
from services.timeseries.retraining_scheduler import (
    AutomatedRetrainingScheduler, RetrainingRequest, RetrainingSchedule, 
    DataFreshnessChecker, RetrainingTrigger
)
from services.timeseries.incremental_trainer import (
    IncrementalModelTrainer, IncrementalTrainingConfig, ModelVersionManager
)

@dataclass
class AutomatedRetrainingConfig:
    """Configuration for automated retraining system"""
    
    # Performance monitoring settings
    performance_check_interval_hours: int = 6
    performance_degradation_threshold: float = 0.05  # 5% degradation triggers retraining
    minimum_performance_samples: int = 10
    
    # Data freshness settings
    data_freshness_check_interval_hours: int = 4
    maximum_data_age_hours: int = 48
    minimum_data_quality_score: float = 0.8
    
    # Retraining settings
    enable_scheduled_retraining: bool = True
    scheduled_retraining_frequency: str = "weekly"  # "daily", "weekly", "monthly"
    enable_performance_based_retraining: bool = True
    enable_data_freshness_retraining: bool = True
    
    # Model management
    max_model_versions: int = 10
    model_validation_threshold: float = 0.02  # New model must be 2% better
    enable_automatic_deployment: bool = False  # Require manual approval by default
    
    # Notification settings
    enable_notifications: bool = True
    notification_channels: List[str] = None
    
    def __post_init__(self):
        if self.notification_channels is None:
            self.notification_channels = ["log", "email"]


@dataclass
class RetrainingSystemStatus:
    """Current status of the automated retraining system"""
    
    system_enabled: bool
    last_performance_check: Optional[datetime]
    last_data_freshness_check: Optional[datetime]
    last_retraining_attempt: Optional[datetime]
    
    current_model_version: str
    model_performance_score: Optional[float]
    data_freshness_score: Optional[float]
    
    pending_retraining_requests: int
    active_retraining_jobs: int
    
    next_scheduled_retraining: Optional[datetime]
    system_health: str  # "healthy", "degraded", "critical"
    
    alerts: List[str]
    recommendations: List[str]


class AutomatedRetrainingSystem:
    """
    Main automated retraining system that coordinates all retraining activities
    """
    
    def __init__(self, config: Optional[AutomatedRetrainingConfig] = None):
        self.config = config or AutomatedRetrainingConfig()
        self.logger = get_data_logger()
        self.performance_logger = get_performance_logger()
        
        # Initialize components
        self.performance_monitor = ModelPerformanceMonitor()
        self.scheduler = AutomatedRetrainingScheduler()
        self.data_freshness_checker = DataFreshnessChecker()
        self.incremental_trainer = IncrementalModelTrainer()
        self.version_manager = ModelVersionManager()
        
        # System state
        self.is_running = False
        self.current_status = RetrainingSystemStatus(
            system_enabled=False,
            last_performance_check=None,
            last_data_freshness_check=None,
            last_retraining_attempt=None,
            current_model_version="unknown",
            model_performance_score=None,
            data_freshness_score=None,
            pending_retraining_requests=0,
            active_retraining_jobs=0,
            next_scheduled_retraining=None,
            system_health="unknown",
            alerts=[],
            recommendations=[]
        )
        
        # Background tasks
        self._background_tasks = []
        
        self.logger.info("Automated retraining system initialized")
    
    async def start(self):
        """Start the automated retraining system"""
        if self.is_running:
            self.logger.warning("Automated retraining system is already running")
            return
        
        self.logger.info("Starting automated retraining system")
        self.is_running = True
        self.current_status.system_enabled = True
        
        try:
            # Start background monitoring tasks
            if self.config.enable_performance_based_retraining:
                task = asyncio.create_task(self._performance_monitoring_loop())
                self._background_tasks.append(task)
            
            if self.config.enable_data_freshness_retraining:
                task = asyncio.create_task(self._data_freshness_monitoring_loop())
                self._background_tasks.append(task)
            
            if self.config.enable_scheduled_retraining:
                task = asyncio.create_task(self._scheduled_retraining_loop())
                self._background_tasks.append(task)
            
            # Start retraining request processor
            task = asyncio.create_task(self._process_retraining_requests_loop())
            self._background_tasks.append(task)
            
            # Update system status
            await self._update_system_status()
            
            self.logger.info("Automated retraining system started successfully")
            
        except Exception as e:
            self.logger.error(f"Error starting automated retraining system: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """Stop the automated retraining system"""
        if not self.is_running:
            return
        
        self.logger.info("Stopping automated retraining system")
        self.is_running = False
        self.current_status.system_enabled = False
        
        # Cancel all background tasks
        for task in self._background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        self._background_tasks.clear()
        self.logger.info("Automated retraining system stopped")
    
    async def _performance_monitoring_loop(self):
        """Background loop for performance monitoring"""
        while self.is_running:
            try:
                await self._check_model_performance()
                await asyncio.sleep(self.config.performance_check_interval_hours * 3600)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in performance monitoring loop: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retrying
    
    async def _data_freshness_monitoring_loop(self):
        """Background loop for data freshness monitoring"""
        while self.is_running:
            try:
                await self._check_data_freshness()
                await asyncio.sleep(self.config.data_freshness_check_interval_hours * 3600)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in data freshness monitoring loop: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retrying
    
    async def _scheduled_retraining_loop(self):
        """Background loop for scheduled retraining"""
        while self.is_running:
            try:
                await self._check_scheduled_retraining()
                await asyncio.sleep(3600)  # Check every hour
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in scheduled retraining loop: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retrying
    
    async def _process_retraining_requests_loop(self):
        """Background loop for processing retraining requests"""
        while self.is_running:
            try:
                await self._process_pending_retraining_requests()
                await asyncio.sleep(60)  # Check every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in retraining request processing loop: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retrying
    
    async def _check_model_performance(self):
        """Check current model performance and trigger retraining if needed"""
        try:
            self.logger.info("Checking model performance")
            
            # Get current model performance
            performance_report = await self.performance_monitor.get_current_performance()
            self.current_status.last_performance_check = datetime.now()
            self.current_status.model_performance_score = performance_report.overall_score
            
            # Check for performance degradation
            if performance_report.performance_degraded:
                degradation_amount = performance_report.degradation_amount
                
                if degradation_amount >= self.config.performance_degradation_threshold:
                    self.logger.warning(
                        f"Model performance degraded by {degradation_amount:.3f}, "
                        f"triggering retraining"
                    )
                    
                    # Create retraining request
                    request = RetrainingRequest(
                        request_id=f"perf_degradation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        timestamp=datetime.now(),
                        trigger_type=RetrainingTrigger.PERFORMANCE_DEGRADATION,
                        model_version=self.current_status.current_model_version,
                        priority="high",
                        reason=f"Performance degraded by {degradation_amount:.3f}",
                        requested_by="system",
                        metadata={
                            "degradation_amount": degradation_amount,
                            "performance_report": performance_report.to_dict()
                        }
                    )
                    
                    await self.scheduler.submit_retraining_request(request)
                    
                    # Add alert
                    alert = f"Performance degradation detected: {degradation_amount:.3f}"
                    self.current_status.alerts.append(alert)
            
        except Exception as e:
            self.logger.error(f"Error checking model performance: {e}")
    
    async def _check_data_freshness(self):
        """Check data freshness and trigger retraining if needed"""
        try:
            self.logger.info("Checking data freshness")
            
            # Get data freshness report
            freshness_report = await self.data_freshness_checker.check_data_freshness()
            self.current_status.last_data_freshness_check = datetime.now()
            self.current_status.data_freshness_score = freshness_report.freshness_score
            
            # Check if retraining is recommended
            if freshness_report.recommendation == "trigger_retraining":
                self.logger.warning(
                    f"Data freshness requires retraining: {freshness_report.reason}"
                )
                
                # Create retraining request
                request = RetrainingRequest(
                    request_id=f"data_freshness_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    timestamp=datetime.now(),
                    trigger_type=RetrainingTrigger.DATA_FRESHNESS,
                    model_version=self.current_status.current_model_version,
                    priority="medium",
                    reason=freshness_report.reason,
                    requested_by="system",
                    metadata={
                        "freshness_report": freshness_report.to_dict()
                    }
                )
                
                await self.scheduler.submit_retraining_request(request)
                
                # Add alert
                alert = f"Data freshness issue: {freshness_report.reason}"
                self.current_status.alerts.append(alert)
        
        except Exception as e:
            self.logger.error(f"Error checking data freshness: {e}")
    
    async def _check_scheduled_retraining(self):
        """Check for scheduled retraining that needs to be executed"""
        try:
            # Get due scheduled retraining jobs
            due_schedules = await self.scheduler.get_due_schedules()
            
            for schedule in due_schedules:
                self.logger.info(f"Executing scheduled retraining: {schedule.schedule_id}")
                
                # Create retraining request
                request = RetrainingRequest(
                    request_id=f"scheduled_{schedule.schedule_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    timestamp=datetime.now(),
                    trigger_type=RetrainingTrigger.SCHEDULED,
                    model_version=schedule.model_version,
                    priority="low",
                    reason=f"Scheduled retraining: {schedule.frequency}",
                    requested_by="system",
                    training_period_months=schedule.training_period_months or 24,
                    validation_period_months=schedule.validation_period_months or 6,
                    metadata={
                        "schedule_id": schedule.schedule_id,
                        "frequency": schedule.frequency
                    }
                )
                
                await self.scheduler.submit_retraining_request(request)
        
        except Exception as e:
            self.logger.error(f"Error checking scheduled retraining: {e}")
    
    async def _process_pending_retraining_requests(self):
        """Process pending retraining requests"""
        try:
            # Get pending requests
            pending_requests = await self.scheduler.get_pending_requests()
            self.current_status.pending_retraining_requests = len(pending_requests)
            
            # Process high priority requests first
            for request in pending_requests:
                if request.priority in ["critical", "high"]:
                    await self._execute_retraining_request(request)
                elif request.priority == "medium" and self.current_status.active_retraining_jobs == 0:
                    await self._execute_retraining_request(request)
                elif request.priority == "low" and self.current_status.active_retraining_jobs == 0:
                    # Only process low priority if no other jobs are running
                    await self._execute_retraining_request(request)
        
        except Exception as e:
            self.logger.error(f"Error processing retraining requests: {e}")
    
    async def _execute_retraining_request(self, request: RetrainingRequest):
        """Execute a retraining request"""
        try:
            self.logger.info(f"Executing retraining request: {request.request_id}")
            self.current_status.active_retraining_jobs += 1
            self.current_status.last_retraining_attempt = datetime.now()
            
            # Mark request as in progress
            await self.scheduler.update_request_status(request.request_id, "in_progress")
            
            # Configure incremental training
            training_config = IncrementalTrainingConfig(
                training_period_months=request.training_period_months,
                validation_period_months=request.validation_period_months,
                incremental=request.incremental,
                force_full_retrain=request.force_full_retrain
            )
            
            # Execute training
            training_result = await self.incremental_trainer.train_incremental_model(
                base_model_version=request.model_version,
                config=training_config
            )
            
            # Validate new model
            if training_result.success:
                validation_result = await self._validate_new_model(
                    training_result.new_model_version,
                    request.model_version
                )
                
                if validation_result.is_better:
                    # Deploy new model if automatic deployment is enabled
                    if self.config.enable_automatic_deployment:
                        await self._deploy_new_model(training_result.new_model_version)
                        status = "completed"
                        self.logger.info(f"New model {training_result.new_model_version} deployed automatically")
                    else:
                        status = "awaiting_approval"
                        self.logger.info(f"New model {training_result.new_model_version} awaiting manual approval")
                    
                    # Update request status
                    await self.scheduler.update_request_status(
                        request.request_id, 
                        status,
                        new_model_version=training_result.new_model_version,
                        performance_improvement=validation_result.improvement_amount
                    )
                else:
                    # New model is not better
                    await self.scheduler.update_request_status(
                        request.request_id,
                        "completed",
                        error_message="New model did not meet improvement threshold"
                    )
                    self.logger.warning(f"New model did not improve performance sufficiently")
            else:
                # Training failed
                await self.scheduler.update_request_status(
                    request.request_id,
                    "failed",
                    error_message=training_result.error_message
                )
                self.logger.error(f"Retraining failed: {training_result.error_message}")
        
        except Exception as e:
            self.logger.error(f"Error executing retraining request {request.request_id}: {e}")
            await self.scheduler.update_request_status(
                request.request_id,
                "failed",
                error_message=str(e)
            )
        finally:
            self.current_status.active_retraining_jobs -= 1
    
    async def _validate_new_model(self, new_model_version: str, current_model_version: str):
        """Validate new model against current model"""
        # This would implement model validation logic
        # For now, return a simple validation result
        from dataclasses import dataclass
        
        @dataclass
        class ValidationResult:
            is_better: bool
            improvement_amount: float
            metrics: Dict[str, float]
        
        # Placeholder validation - in real implementation, this would
        # compare model performance on validation dataset
        return ValidationResult(
            is_better=True,
            improvement_amount=0.03,  # 3% improvement
            metrics={"f1_score": 0.85, "precision": 0.82, "recall": 0.88}
        )
    
    async def _deploy_new_model(self, model_version: str):
        """Deploy new model version"""
        try:
            await self.version_manager.deploy_model_version(model_version)
            self.current_status.current_model_version = model_version
            self.logger.info(f"Successfully deployed model version: {model_version}")
        except Exception as e:
            self.logger.error(f"Error deploying model version {model_version}: {e}")
            raise
    
    async def _update_system_status(self):
        """Update system status"""
        try:
            # Update current model version
            current_version = await self.version_manager.get_current_model_version()
            self.current_status.current_model_version = current_version
            
            # Update next scheduled retraining
            next_schedule = await self.scheduler.get_next_scheduled_retraining()
            self.current_status.next_scheduled_retraining = next_schedule
            
            # Determine system health
            alerts_count = len(self.current_status.alerts)
            if alerts_count == 0:
                self.current_status.system_health = "healthy"
            elif alerts_count <= 2:
                self.current_status.system_health = "degraded"
            else:
                self.current_status.system_health = "critical"
            
            # Generate recommendations
            self.current_status.recommendations = await self._generate_recommendations()
        
        except Exception as e:
            self.logger.error(f"Error updating system status: {e}")
    
    async def _generate_recommendations(self) -> List[str]:
        """Generate system recommendations"""
        recommendations = []
        
        # Performance-based recommendations
        if (self.current_status.model_performance_score and 
            self.current_status.model_performance_score < 0.7):
            recommendations.append("Model performance is below threshold - consider retraining")
        
        # Data freshness recommendations
        if (self.current_status.data_freshness_score and 
            self.current_status.data_freshness_score < 0.5):
            recommendations.append("Data freshness is low - consider updating training data")
        
        # System health recommendations
        if self.current_status.system_health == "critical":
            recommendations.append("System health is critical - immediate attention required")
        elif self.current_status.system_health == "degraded":
            recommendations.append("System health is degraded - monitor closely")
        
        return recommendations
    
    async def get_system_status(self) -> RetrainingSystemStatus:
        """Get current system status"""
        await self._update_system_status()
        return self.current_status
    
    async def trigger_manual_retraining(self, reason: str, priority: str = "medium", 
                                      **kwargs) -> str:
        """Manually trigger retraining"""
        request = RetrainingRequest(
            request_id=f"manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(),
            trigger_type=RetrainingTrigger.MANUAL,
            model_version=self.current_status.current_model_version,
            priority=priority,
            reason=reason,
            requested_by="user",
            **kwargs
        )
        
        await self.scheduler.submit_retraining_request(request)
        self.logger.info(f"Manual retraining request submitted: {request.request_id}")
        return request.request_id
    
    async def get_retraining_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get retraining history"""
        return await self.scheduler.get_retraining_history(limit)
    
    async def cancel_retraining_request(self, request_id: str) -> bool:
        """Cancel a pending retraining request"""
        return await self.scheduler.cancel_retraining_request(request_id)


# Global instance for easy access
_automated_retraining_system = None

def get_automated_retraining_system(config: Optional[AutomatedRetrainingConfig] = None) -> AutomatedRetrainingSystem:
    """Get the global automated retraining system instance"""
    global _automated_retraining_system
    if _automated_retraining_system is None:
        _automated_retraining_system = AutomatedRetrainingSystem(config)
    return _automated_retraining_system


async def main():
    """Main function for testing the automated retraining system"""
    # Initialize system
    config = AutomatedRetrainingConfig(
        performance_check_interval_hours=1,  # Check every hour for testing
        data_freshness_check_interval_hours=1,
        enable_automatic_deployment=False  # Require manual approval
    )
    
    system = AutomatedRetrainingSystem(config)
    
    try:
        # Start the system
        await system.start()
        
        # Get initial status
        status = await system.get_system_status()
        print(f"System Status: {status.system_health}")
        print(f"Current Model: {status.current_model_version}")
        print(f"Pending Requests: {status.pending_retraining_requests}")
        
        # Trigger a manual retraining for testing
        request_id = await system.trigger_manual_retraining(
            reason="Testing automated retraining system",
            priority="low"
        )
        print(f"Manual retraining request submitted: {request_id}")
        
        # Run for a short time
        await asyncio.sleep(10)
        
        # Get updated status
        status = await system.get_system_status()
        print(f"Updated Status: {status.system_health}")
        print(f"Active Jobs: {status.active_retraining_jobs}")
        
    finally:
        # Stop the system
        await system.stop()


if __name__ == "__main__":
    asyncio.run(main())