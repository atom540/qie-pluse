"""
Database Schema for AI Risk Oracle Data Storage
Defines database tables and models for historical data, sentiment data, and metadata
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import json

from config import get_config
from logging_config import get_data_logger

class DataSource(Enum):
    """Enumeration of data sources"""
    COINGECKO = "coingecko"
    BINANCE = "binance"
    ALPHA_VANTAGE = "alphavantage"
    YAHOO_FINANCE = "yahoo_finance"
    TWITTER = "twitter"
    REDDIT = "reddit"
    TELEGRAM = "telegram"
    QIE_ORACLE = "qie_oracle"

class AssetType(Enum):
    """Enumeration of asset types"""
    CRYPTO = "crypto"
    TRADITIONAL = "traditional"
    COMMODITY = "commodity"

@dataclass
class HistoricalPriceData:
    """Database model for historical price data"""
    id: Optional[int] = None
    symbol: str = ""
    asset_type: str = ""
    price: float = 0.0
    volume_24h: float = 0.0
    price_change_24h: float = 0.0
    market_cap: Optional[float] = None
    timestamp: datetime = None
    source: str = ""
    data_lineage_id: Optional[int] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class SentimentData:
    """Database model for social sentiment data"""
    id: Optional[int] = None
    platform: str = ""
    post_id: str = ""
    text: str = ""
    author: str = ""
    engagement_score: float = 0.0
    sentiment_score: Optional[float] = None
    keywords: str = ""  # JSON string of keyword list
    timestamp: datetime = None
    source: str = ""
    data_lineage_id: Optional[int] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class ModelTrainingData:
    """Database model for model training datasets"""
    id: Optional[int] = None
    model_type: str = ""
    dataset_name: str = ""
    feature_columns: str = ""  # JSON string of feature column names
    target_column: str = ""
    training_samples: int = 0
    validation_samples: int = 0
    test_samples: int = 0
    data_sources: str = ""  # JSON string of data source list
    created_at: datetime = None
    data_lineage_id: Optional[int] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class DataLineage:
    """Database model for data lineage tracking"""
    id: Optional[int] = None
    collection_id: str = ""  # Unique identifier for data collection run
    source: str = ""
    collection_method: str = ""
    parameters: str = ""  # JSON string of collection parameters
    start_time: datetime = None
    end_time: datetime = None
    records_collected: int = 0
    success: bool = True
    error_message: Optional[str] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

class DatabaseSchema:
    """Database schema management for AI Risk Oracle"""
    
    def __init__(self, db_path: Optional[Path] = None):
        self.config = get_config()
        self.data_logger = get_data_logger()
        
        if db_path is None:
            self.db_path = self.config.logging.data_dir / "ai_risk_oracle.db"
        else:
            self.db_path = db_path
        
        # Ensure data directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
    
    def get_connection(self) -> sqlite3.Connection:
        """Get database connection with proper configuration"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row  # Enable column access by name
        
        # Enable foreign key constraints
        conn.execute("PRAGMA foreign_keys = ON")
        
        # Set journal mode for better concurrency
        conn.execute("PRAGMA journal_mode = WAL")
        
        return conn
    
    def create_tables(self) -> None:
        """Create all database tables with proper indexes"""
        with self.get_connection() as conn:
            # Data lineage table (must be created first due to foreign keys)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS data_lineage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    collection_id TEXT NOT NULL UNIQUE,
                    source TEXT NOT NULL,
                    collection_method TEXT NOT NULL,
                    parameters TEXT,
                    start_time TIMESTAMP NOT NULL,
                    end_time TIMESTAMP,
                    records_collected INTEGER DEFAULT 0,
                    success BOOLEAN DEFAULT TRUE,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Historical price data table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS historical_price_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    asset_type TEXT NOT NULL,
                    price REAL NOT NULL,
                    volume_24h REAL NOT NULL,
                    price_change_24h REAL NOT NULL,
                    market_cap REAL,
                    timestamp TIMESTAMP NOT NULL,
                    source TEXT NOT NULL,
                    data_lineage_id INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (data_lineage_id) REFERENCES data_lineage (id)
                )
            """)
            
            # Sentiment data table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sentiment_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    platform TEXT NOT NULL,
                    post_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    author TEXT NOT NULL,
                    engagement_score REAL NOT NULL,
                    sentiment_score REAL,
                    keywords TEXT,
                    timestamp TIMESTAMP NOT NULL,
                    source TEXT NOT NULL,
                    data_lineage_id INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (data_lineage_id) REFERENCES data_lineage (id),
                    UNIQUE(platform, post_id)
                )
            """)
            
            # Model training data table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_training_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_type TEXT NOT NULL,
                    dataset_name TEXT NOT NULL,
                    feature_columns TEXT NOT NULL,
                    target_column TEXT NOT NULL,
                    training_samples INTEGER NOT NULL,
                    validation_samples INTEGER NOT NULL,
                    test_samples INTEGER NOT NULL,
                    data_sources TEXT NOT NULL,
                    data_lineage_id INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (data_lineage_id) REFERENCES data_lineage (id)
                )
            """)
            
            conn.commit()
    
    def create_indexes(self) -> None:
        """Create database indexes for optimal query performance"""
        with self.get_connection() as conn:
            # Historical price data indexes
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_historical_price_symbol_timestamp 
                ON historical_price_data (symbol, timestamp)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_historical_price_source_timestamp 
                ON historical_price_data (source, timestamp)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_historical_price_asset_type 
                ON historical_price_data (asset_type)
            """)
            
            # Sentiment data indexes
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sentiment_platform_timestamp 
                ON sentiment_data (platform, timestamp)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sentiment_source_timestamp 
                ON sentiment_data (source, timestamp)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sentiment_keywords 
                ON sentiment_data (keywords)
            """)
            
            # Data lineage indexes
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_data_lineage_collection_id 
                ON data_lineage (collection_id)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_data_lineage_source_start_time 
                ON data_lineage (source, start_time)
            """)
            
            # Model training data indexes
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_model_training_type_created 
                ON model_training_data (model_type, created_at)
            """)
            
            conn.commit()
    
    def drop_tables(self) -> None:
        """Drop all tables (for testing/reset purposes)"""
        with self.get_connection() as conn:
            conn.execute("DROP TABLE IF EXISTS model_training_data")
            conn.execute("DROP TABLE IF EXISTS sentiment_data")
            conn.execute("DROP TABLE IF EXISTS historical_price_data")
            conn.execute("DROP TABLE IF EXISTS data_lineage")
            conn.commit()
    
    def get_table_info(self, table_name: str) -> List[Dict[str, Any]]:
        """Get table schema information"""
        with self.get_connection() as conn:
            cursor = conn.execute(f"PRAGMA table_info({table_name})")
            return [dict(row) for row in cursor.fetchall()]
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        with self.get_connection() as conn:
            stats = {}
            
            # Table row counts
            for table in ["historical_price_data", "sentiment_data", "model_training_data", "data_lineage"]:
                cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                stats[f"{table}_count"] = cursor.fetchone()[0]
            
            # Database file size
            stats["database_size_bytes"] = self.db_path.stat().st_size if self.db_path.exists() else 0
            
            # Latest data timestamps
            cursor = conn.execute("""
                SELECT MAX(timestamp) as latest_price_data 
                FROM historical_price_data
            """)
            result = cursor.fetchone()
            stats["latest_price_data"] = result[0] if result[0] else None
            
            cursor = conn.execute("""
                SELECT MAX(timestamp) as latest_sentiment_data 
                FROM sentiment_data
            """)
            result = cursor.fetchone()
            stats["latest_sentiment_data"] = result[0] if result[0] else None
            
            return stats

def create_database_schema(db_path: Optional[Path] = None) -> DatabaseSchema:
    """Create and initialize database schema"""
    schema = DatabaseSchema(db_path)
    schema.create_tables()
    schema.create_indexes()
    return schema