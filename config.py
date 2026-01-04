"""
AI Risk Oracle Configuration Management
Centralized configuration loading and validation
"""

import os
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@dataclass
class APIConfig:
    """API client configuration"""
    coingecko_api_key: str
    alpha_vantage_api_key: str
    binance_api_key: str
    binance_secret_key: str
    twitter_bearer_token: str
    twitter_api_key: str
    twitter_api_secret: str
    twitter_access_token: str
    twitter_access_token_secret: str
    telegram_api_id: int
    telegram_api_hash: str
    telegram_phone_number: str
    reddit_client_id: str
    reddit_client_secret: str
    reddit_user_agent: str

@dataclass
class BlockchainConfig:
    """Blockchain connection configuration"""
    qie_rpc_url: str
    qie_rpc_backup_url: str
    ai_risk_oracle_contract_address: str
    private_key: str
    default_gas_limit: int
    default_gas_price: int

@dataclass
class ModelConfig:
    """ML model configuration"""
    sentiment_model_name: str
    model_cache_dir: Path
    sentiment_weight: float
    market_weight: float
    risk_threshold: float
    ewma_alpha: float
    anomaly_threshold: float

@dataclass
class PerformanceConfig:
    """Performance and timeout configuration"""
    inference_timeout_ms: int
    blockchain_timeout_ms: int
    max_retries: int
    retry_backoff_factor: float
    retry_max_delay: float

@dataclass
class LoggingConfig:
    """Logging and monitoring configuration"""
    log_level: str
    log_format: str
    data_dir: Path
    models_dir: Path
    logs_dir: Path
    prometheus_port: int
    health_check_port: int
    alert_webhook_url: Optional[str]
    alert_timeout_seconds: int

@dataclass
class DevelopmentConfig:
    """Development and testing configuration"""
    environment: str
    debug: bool
    hypothesis_max_examples: int
    pytest_timeout: int

class Config:
    """Main configuration class"""
    
    def __init__(self):
        self.api = self._load_api_config()
        self.blockchain = self._load_blockchain_config()
        self.model = self._load_model_config()
        self.performance = self._load_performance_config()
        self.logging = self._load_logging_config()
        self.development = self._load_development_config()
        
        # Validate configuration
        self._validate_config()
    
    def _load_api_config(self) -> APIConfig:
        """Load API configuration from environment variables"""
        return APIConfig(
            coingecko_api_key=os.getenv("COINGECKO_API_KEY", ""),
            alpha_vantage_api_key=os.getenv("ALPHA_VANTAGE_API_KEY", ""),
            binance_api_key=os.getenv("BINANCE_API_KEY", ""),
            binance_secret_key=os.getenv("BINANCE_SECRET_KEY", ""),
            twitter_bearer_token=os.getenv("TWITTER_BEARER_TOKEN", ""),
            twitter_api_key=os.getenv("TWITTER_API_KEY", ""),
            twitter_api_secret=os.getenv("TWITTER_API_SECRET", ""),
            twitter_access_token=os.getenv("TWITTER_ACCESS_TOKEN", ""),
            twitter_access_token_secret=os.getenv("TWITTER_ACCESS_TOKEN_SECRET", ""),
            telegram_api_id=int(os.getenv("TELEGRAM_API_ID", "0")),
            telegram_api_hash=os.getenv("TELEGRAM_API_HASH", ""),
            telegram_phone_number=os.getenv("TELEGRAM_PHONE_NUMBER", ""),
            reddit_client_id=os.getenv("REDDIT_CLIENT_ID", ""),
            reddit_client_secret=os.getenv("REDDIT_CLIENT_SECRET", ""),
            reddit_user_agent=os.getenv("REDDIT_USER_AGENT", "ai-risk-oracle/1.0")
        )
    
    def _load_blockchain_config(self) -> BlockchainConfig:
        """Load blockchain configuration from environment variables"""
        return BlockchainConfig(
            qie_rpc_url=os.getenv("QIE_RPC_URL", "https://rpc.qie-pulse.com"),
            qie_rpc_backup_url=os.getenv("QIE_RPC_BACKUP_URL", "https://backup-rpc.qie-pulse.com"),
            ai_risk_oracle_contract_address=os.getenv("AI_RISK_ORACLE_CONTRACT_ADDRESS", ""),
            private_key=os.getenv("PRIVATE_KEY", ""),
            default_gas_limit=int(os.getenv("DEFAULT_GAS_LIMIT", "100000")),
            default_gas_price=int(os.getenv("DEFAULT_GAS_PRICE", "20000000000"))
        )
    
    def _load_model_config(self) -> ModelConfig:
        """Load model configuration from environment variables"""
        return ModelConfig(
            sentiment_model_name=os.getenv("SENTIMENT_MODEL_NAME", "cardiffnlp/twitter-roberta-base-sentiment"),
            model_cache_dir=Path(os.getenv("MODEL_CACHE_DIR", "./models/cache")),
            sentiment_weight=float(os.getenv("SENTIMENT_WEIGHT", "0.4")),
            market_weight=float(os.getenv("MARKET_WEIGHT", "0.6")),
            risk_threshold=float(os.getenv("RISK_THRESHOLD", "0.85")),
            ewma_alpha=float(os.getenv("EWMA_ALPHA", "0.3")),
            anomaly_threshold=float(os.getenv("ANOMALY_THRESHOLD", "2.0"))
        )
    
    def _load_performance_config(self) -> PerformanceConfig:
        """Load performance configuration from environment variables"""
        return PerformanceConfig(
            inference_timeout_ms=int(os.getenv("INFERENCE_TIMEOUT_MS", "500")),
            blockchain_timeout_ms=int(os.getenv("BLOCKCHAIN_TIMEOUT_MS", "3000")),
            max_retries=int(os.getenv("MAX_RETRIES", "3")),
            retry_backoff_factor=float(os.getenv("RETRY_BACKOFF_FACTOR", "2.0")),
            retry_max_delay=float(os.getenv("RETRY_MAX_DELAY", "60.0"))
        )
    
    def _load_logging_config(self) -> LoggingConfig:
        """Load logging configuration from environment variables"""
        return LoggingConfig(
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_format=os.getenv("LOG_FORMAT", "json"),
            data_dir=Path(os.getenv("DATA_DIR", "./data")),
            models_dir=Path(os.getenv("MODELS_DIR", "./models")),
            logs_dir=Path(os.getenv("LOGS_DIR", "./logs")),
            prometheus_port=int(os.getenv("PROMETHEUS_PORT", "8000")),
            health_check_port=int(os.getenv("HEALTH_CHECK_PORT", "8080")),
            alert_webhook_url=os.getenv("ALERT_WEBHOOK_URL"),
            alert_timeout_seconds=int(os.getenv("ALERT_TIMEOUT_SECONDS", "60"))
        )
    
    def _load_development_config(self) -> DevelopmentConfig:
        """Load development configuration from environment variables"""
        return DevelopmentConfig(
            environment=os.getenv("ENVIRONMENT", "development"),
            debug=os.getenv("DEBUG", "false").lower() == "true",
            hypothesis_max_examples=int(os.getenv("HYPOTHESIS_MAX_EXAMPLES", "100")),
            pytest_timeout=int(os.getenv("PYTEST_TIMEOUT", "30"))
        )
    
    def _validate_config(self) -> None:
        """Validate configuration values"""
        errors = []
        
        # Validate model weights sum to 1.0
        total_weight = self.model.sentiment_weight + self.model.market_weight
        if abs(total_weight - 1.0) > 0.001:
            errors.append(f"Model weights must sum to 1.0, got {total_weight}")
        
        # Validate risk threshold is between 0 and 1
        if not 0.0 <= self.model.risk_threshold <= 1.0:
            errors.append(f"Risk threshold must be between 0.0 and 1.0, got {self.model.risk_threshold}")
        
        # Validate EWMA alpha is between 0 and 1
        if not 0.0 <= self.model.ewma_alpha <= 1.0:
            errors.append(f"EWMA alpha must be between 0.0 and 1.0, got {self.model.ewma_alpha}")
        
        # Validate timeout values are positive
        if self.performance.inference_timeout_ms <= 0:
            errors.append(f"Inference timeout must be positive, got {self.performance.inference_timeout_ms}")
        
        if self.performance.blockchain_timeout_ms <= 0:
            errors.append(f"Blockchain timeout must be positive, got {self.performance.blockchain_timeout_ms}")
        
        # Validate retry configuration
        if self.performance.max_retries < 0:
            errors.append(f"Max retries must be non-negative, got {self.performance.max_retries}")
        
        if self.performance.retry_backoff_factor <= 1.0:
            errors.append(f"Retry backoff factor must be > 1.0, got {self.performance.retry_backoff_factor}")
        
        if errors:
            raise ValueError(f"Configuration validation failed:\n" + "\n".join(f"- {error}" for error in errors))
    
    def create_directories(self) -> None:
        """Create necessary directories if they don't exist"""
        directories = [
            self.logging.data_dir,
            self.logging.models_dir,
            self.logging.logs_dir,
            self.model.model_cache_dir,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_qie_oracle_endpoints(self) -> List[str]:
        """Get list of QIE oracle endpoints for the seven assets"""
        base_url = self.blockchain.qie_rpc_url.rstrip('/')
        assets = ["BTC", "ETH", "XRP", "SOL", "QIE", "GOLD", "BNB"]
        return [f"{base_url}/oracle/{asset.lower()}" for asset in assets]
    
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.development.environment.lower() == "production"
    
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.development.environment.lower() == "development"

# Global configuration instance
config = Config()

# Convenience function to get configuration
def get_config() -> Config:
    """Get the global configuration instance"""
    return config