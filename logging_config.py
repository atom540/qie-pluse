"""
AI Risk Oracle Logging Configuration
Structured logging setup with JSON formatting and performance tracking
"""

import logging
import logging.config
import sys
from pathlib import Path
from typing import Dict, Any
import structlog
from config import get_config

def setup_logging() -> None:
    """Configure structured logging for the AI Risk Oracle system"""
    config = get_config()
    
    # Create logs directory if it doesn't exist
    config.logging.logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure standard library logging
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "()": structlog.stdlib.ProcessorFormatter,
                "processor": structlog.dev.ConsoleRenderer(colors=False)
                if config.logging.log_format == "console"
                else structlog.processors.JSONRenderer(),
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "json",
                "stream": sys.stdout,
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "json",
                "filename": config.logging.logs_dir / "ai_risk_oracle.log",
                "maxBytes": 10 * 1024 * 1024,  # 10MB
                "backupCount": 5,
            },
            "error_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "json",
                "filename": config.logging.logs_dir / "errors.log",
                "maxBytes": 10 * 1024 * 1024,  # 10MB
                "backupCount": 5,
                "level": "ERROR",
            },
        },
        "loggers": {
            "": {  # Root logger
                "handlers": ["console", "file", "error_file"],
                "level": config.logging.log_level,
                "propagate": False,
            },
            "ai_risk_oracle": {
                "handlers": ["console", "file", "error_file"],
                "level": config.logging.log_level,
                "propagate": False,
            },
        },
    }
    
    logging.config.dictConfig(logging_config)
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

def get_logger(name: str) -> structlog.BoundLogger:
    """Get a structured logger instance"""
    return structlog.get_logger(name)

class PerformanceLogger:
    """Logger for performance metrics and timing"""
    
    def __init__(self):
        self.logger = get_logger("performance")
    
    def log_inference_time(self, component: str, duration_ms: float, success: bool = True) -> None:
        """Log inference timing metrics"""
        self.logger.info(
            "inference_timing",
            component=component,
            duration_ms=duration_ms,
            success=success,
            metric_type="latency"
        )
    
    def log_throughput(self, component: str, items_processed: int, duration_ms: float) -> None:
        """Log throughput metrics"""
        throughput = items_processed / (duration_ms / 1000.0) if duration_ms > 0 else 0
        self.logger.info(
            "throughput_metrics",
            component=component,
            items_processed=items_processed,
            duration_ms=duration_ms,
            throughput_per_second=throughput,
            metric_type="throughput"
        )
    
    def log_error_rate(self, component: str, total_requests: int, error_count: int) -> None:
        """Log error rate metrics"""
        error_rate = error_count / total_requests if total_requests > 0 else 0
        self.logger.info(
            "error_rate_metrics",
            component=component,
            total_requests=total_requests,
            error_count=error_count,
            error_rate=error_rate,
            metric_type="error_rate"
        )

class RiskLogger:
    """Logger for risk calculation events"""
    
    def __init__(self):
        self.logger = get_logger("risk_calculation")
    
    def log_risk_score(self, risk_factor: float, sentiment_score: float, 
                      market_score: float, reasoning: str, triggered: bool = False) -> None:
        """Log risk score calculation"""
        self.logger.info(
            "risk_score_calculated",
            risk_factor=risk_factor,
            sentiment_score=sentiment_score,
            market_score=market_score,
            reasoning=reasoning,
            triggered=triggered,
            event_type="risk_calculation"
        )
    
    def log_sentiment_anomaly(self, score: float, threshold: float, 
                             confidence: float, message_count: int) -> None:
        """Log sentiment anomaly detection"""
        self.logger.warning(
            "sentiment_anomaly_detected",
            sentiment_score=score,
            anomaly_threshold=threshold,
            confidence=confidence,
            message_count=message_count,
            event_type="anomaly_detection"
        )
    
    def log_blockchain_transaction(self, tx_hash: str, risk_score: float, 
                                  gas_used: int, success: bool) -> None:
        """Log blockchain transaction events"""
        self.logger.info(
            "blockchain_transaction",
            tx_hash=tx_hash,
            risk_score=risk_score,
            gas_used=gas_used,
            success=success,
            event_type="blockchain_transaction"
        )

class DataLogger:
    """Logger for data collection and processing events"""
    
    def __init__(self):
        self.logger = get_logger("data_processing")
    
    def log_data_collection(self, source: str, asset: str, records_collected: int, 
                           success: bool, error_msg: str = None) -> None:
        """Log data collection events"""
        log_data = {
            "data_collection_event": True,
            "source": source,
            "asset": asset,
            "records_collected": records_collected,
            "success": success,
            "event_type": "data_collection"
        }
        
        if error_msg:
            log_data["error_message"] = error_msg
        
        if success:
            self.logger.info("data_collection_success", **log_data)
        else:
            self.logger.error("data_collection_failed", **log_data)
    
    def log_model_training(self, model_type: str, training_samples: int, 
                          validation_accuracy: float, training_time_ms: float) -> None:
        """Log model training events"""
        self.logger.info(
            "model_training_completed",
            model_type=model_type,
            training_samples=training_samples,
            validation_accuracy=validation_accuracy,
            training_time_ms=training_time_ms,
            event_type="model_training"
        )

# Global logger instances
performance_logger = PerformanceLogger()
risk_logger = RiskLogger()
data_logger = DataLogger()

def get_performance_logger() -> PerformanceLogger:
    """Get the global performance logger instance"""
    return performance_logger

def get_risk_logger() -> RiskLogger:
    """Get the global risk logger instance"""
    return risk_logger

def get_data_logger() -> DataLogger:
    """Get the global data logger instance"""
    return data_logger