"""
Data Manager for AI Risk Oracle
Implements data persistence layer with proper indexing, timestamps, and data lineage tracking
"""

import sqlite3
import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import asdict
import pandas as pd

from config import get_config
from logging_config import get_data_logger, get_performance_logger
from services.collectors import CryptoPriceData, TraditionalAssetData, SocialSentimentData
from .schema import (
    DatabaseSchema, 
    HistoricalPriceData, 
    SentimentData, 
    ModelTrainingData, 
    DataLineage,
    DataSource,
    AssetType
)

class DataManager:
    """Main data management class for AI Risk Oracle"""
    
    def __init__(self, db_path: Optional[Path] = None):
        self.config = get_config()
        self.data_logger = get_data_logger()
        self.performance_logger = get_performance_logger()
        
        # Initialize database schema
        self.schema = DatabaseSchema(db_path)
        self.schema.create_tables()
        self.schema.create_indexes()
    
    def start_data_collection(self, source: str, method: str, parameters: Dict[str, Any] = None) -> str:
        """Start a new data collection session and return collection ID"""
        collection_id = str(uuid.uuid4())
        
        lineage = DataLineage(
            collection_id=collection_id,
            source=source,
            collection_method=method,
            parameters=json.dumps(parameters or {}),
            start_time=datetime.now(),
            records_collected=0,
            success=True
        )
        
        lineage_id = self._insert_data_lineage(lineage)
        return collection_id
    
    def end_data_collection(self, collection_id: str, records_collected: int, 
                           success: bool = True, error_message: str = None) -> None:
        """End a data collection session"""
        with self.schema.get_connection() as conn:
            conn.execute("""
                UPDATE data_lineage 
                SET end_time = ?, records_collected = ?, success = ?, error_message = ?
                WHERE collection_id = ?
            """, (datetime.now(), records_collected, success, error_message, collection_id))
            conn.commit()
    
    def _insert_data_lineage(self, lineage: DataLineage) -> int:
        """Insert data lineage record and return ID"""
        with self.schema.get_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO data_lineage 
                (collection_id, source, collection_method, parameters, start_time, 
                 records_collected, success, error_message, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                lineage.collection_id, lineage.source, lineage.collection_method,
                lineage.parameters, lineage.start_time, lineage.records_collected,
                lineage.success, lineage.error_message, lineage.created_at
            ))
            conn.commit()
            return cursor.lastrowid
    
    def _get_lineage_id(self, collection_id: str) -> Optional[int]:
        """Get lineage ID from collection ID"""
        with self.schema.get_connection() as conn:
            cursor = conn.execute("""
                SELECT id FROM data_lineage WHERE collection_id = ?
            """, (collection_id,))
            result = cursor.fetchone()
            return result[0] if result else None
    
    def store_crypto_price_data(self, data: List[CryptoPriceData], 
                               collection_id: str) -> int:
        """Store crypto price data with data lineage tracking"""
        lineage_id = self._get_lineage_id(collection_id)
        stored_count = 0
        
        with self.schema.get_connection() as conn:
            for item in data:
                try:
                    # Convert to database model
                    db_record = HistoricalPriceData(
                        symbol=item.symbol,
                        asset_type=AssetType.CRYPTO.value,
                        price=item.price,
                        volume_24h=item.volume_24h,
                        price_change_24h=item.price_change_24h,
                        market_cap=item.market_cap,
                        timestamp=item.timestamp,
                        source=item.source,
                        data_lineage_id=lineage_id
                    )
                    
                    # Insert record
                    conn.execute("""
                        INSERT INTO historical_price_data 
                        (symbol, asset_type, price, volume_24h, price_change_24h, 
                         market_cap, timestamp, source, data_lineage_id, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        db_record.symbol, db_record.asset_type, db_record.price,
                        db_record.volume_24h, db_record.price_change_24h,
                        db_record.market_cap, db_record.timestamp, db_record.source,
                        db_record.data_lineage_id, db_record.created_at
                    ))
                    stored_count += 1
                    
                except Exception as e:
                    self.data_logger.log_data_collection(
                        item.source, item.symbol, 0, False, str(e)
                    )
                    continue
            
            conn.commit()
        
        self.data_logger.log_data_collection(
            data[0].source if data else "unknown", 
            "crypto_batch", 
            stored_count, 
            True
        )
        
        return stored_count
    
    def store_traditional_asset_data(self, data: List[TraditionalAssetData], 
                                   collection_id: str) -> int:
        """Store traditional asset data with data lineage tracking"""
        lineage_id = self._get_lineage_id(collection_id)
        stored_count = 0
        
        with self.schema.get_connection() as conn:
            for item in data:
                try:
                    # Convert to database model
                    db_record = HistoricalPriceData(
                        symbol=item.symbol,
                        asset_type=AssetType.TRADITIONAL.value,
                        price=item.price,
                        volume_24h=item.volume_24h,
                        price_change_24h=item.price_change_24h,
                        market_cap=item.market_cap,
                        timestamp=item.timestamp,
                        source=item.source,
                        data_lineage_id=lineage_id
                    )
                    
                    # Insert record
                    conn.execute("""
                        INSERT INTO historical_price_data 
                        (symbol, asset_type, price, volume_24h, price_change_24h, 
                         market_cap, timestamp, source, data_lineage_id, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        db_record.symbol, db_record.asset_type, db_record.price,
                        db_record.volume_24h, db_record.price_change_24h,
                        db_record.market_cap, db_record.timestamp, db_record.source,
                        db_record.data_lineage_id, db_record.created_at
                    ))
                    stored_count += 1
                    
                except Exception as e:
                    self.data_logger.log_data_collection(
                        item.source, item.symbol, 0, False, str(e)
                    )
                    continue
            
            conn.commit()
        
        self.data_logger.log_data_collection(
            data[0].source if data else "unknown", 
            "traditional_batch", 
            stored_count, 
            True
        )
        
        return stored_count
    
    def store_sentiment_data(self, data: List[SocialSentimentData], 
                           collection_id: str) -> int:
        """Store social sentiment data with data lineage tracking"""
        lineage_id = self._get_lineage_id(collection_id)
        stored_count = 0
        
        with self.schema.get_connection() as conn:
            for item in data:
                try:
                    # Convert to database model
                    db_record = SentimentData(
                        platform=item.platform,
                        post_id=item.post_id,
                        text=item.text,
                        author=item.author,
                        engagement_score=item.engagement_score,
                        sentiment_score=item.sentiment_score,
                        keywords=json.dumps(item.contains_keywords),
                        timestamp=item.timestamp,
                        source=item.source,
                        data_lineage_id=lineage_id
                    )
                    
                    # Insert record (with conflict resolution for duplicates)
                    conn.execute("""
                        INSERT OR REPLACE INTO sentiment_data 
                        (platform, post_id, text, author, engagement_score, 
                         sentiment_score, keywords, timestamp, source, data_lineage_id, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        db_record.platform, db_record.post_id, db_record.text,
                        db_record.author, db_record.engagement_score,
                        db_record.sentiment_score, db_record.keywords,
                        db_record.timestamp, db_record.source,
                        db_record.data_lineage_id, db_record.created_at
                    ))
                    stored_count += 1
                    
                except Exception as e:
                    self.data_logger.log_data_collection(
                        item.source, item.platform, 0, False, str(e)
                    )
                    continue
            
            conn.commit()
        
        self.data_logger.log_data_collection(
            data[0].source if data else "unknown", 
            "sentiment_batch", 
            stored_count, 
            True
        )
        
        return stored_count
    
    def get_historical_price_data(self, symbol: str = None, 
                                 asset_type: str = None,
                                 source: str = None,
                                 start_date: datetime = None,
                                 end_date: datetime = None,
                                 limit: int = None) -> List[HistoricalPriceData]:
        """Retrieve historical price data with filtering options"""
        query = "SELECT * FROM historical_price_data WHERE 1=1"
        params = []
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        
        if asset_type:
            query += " AND asset_type = ?"
            params.append(asset_type)
        
        if source:
            query += " AND source = ?"
            params.append(source)
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)
        
        query += " ORDER BY timestamp DESC"
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        with self.schema.get_connection() as conn:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            
            return [
                HistoricalPriceData(
                    id=row["id"],
                    symbol=row["symbol"],
                    asset_type=row["asset_type"],
                    price=row["price"],
                    volume_24h=row["volume_24h"],
                    price_change_24h=row["price_change_24h"],
                    market_cap=row["market_cap"],
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    source=row["source"],
                    data_lineage_id=row["data_lineage_id"],
                    created_at=datetime.fromisoformat(row["created_at"])
                )
                for row in rows
            ]
    
    def get_sentiment_data(self, platform: str = None,
                          source: str = None,
                          keywords: List[str] = None,
                          start_date: datetime = None,
                          end_date: datetime = None,
                          limit: int = None) -> List[SentimentData]:
        """Retrieve sentiment data with filtering options"""
        query = "SELECT * FROM sentiment_data WHERE 1=1"
        params = []
        
        if platform:
            query += " AND platform = ?"
            params.append(platform)
        
        if source:
            query += " AND source = ?"
            params.append(source)
        
        if keywords:
            # Search for any of the keywords in the keywords JSON field
            keyword_conditions = []
            for keyword in keywords:
                keyword_conditions.append("keywords LIKE ?")
                params.append(f'%"{keyword}"%')
            query += f" AND ({' OR '.join(keyword_conditions)})"
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)
        
        query += " ORDER BY timestamp DESC"
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        with self.schema.get_connection() as conn:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            
            return [
                SentimentData(
                    id=row["id"],
                    platform=row["platform"],
                    post_id=row["post_id"],
                    text=row["text"],
                    author=row["author"],
                    engagement_score=row["engagement_score"],
                    sentiment_score=row["sentiment_score"],
                    keywords=row["keywords"],
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    source=row["source"],
                    data_lineage_id=row["data_lineage_id"],
                    created_at=datetime.fromisoformat(row["created_at"])
                )
                for row in rows
            ]
    
    def create_training_dataset(self, model_type: str, dataset_name: str,
                              feature_columns: List[str], target_column: str,
                              data_sources: List[str],
                              collection_id: str = None) -> int:
        """Create a model training dataset record"""
        lineage_id = self._get_lineage_id(collection_id) if collection_id else None
        
        # Get sample counts (simplified - in practice would analyze actual data)
        training_samples = 1000  # Placeholder
        validation_samples = 200  # Placeholder
        test_samples = 100  # Placeholder
        
        training_data = ModelTrainingData(
            model_type=model_type,
            dataset_name=dataset_name,
            feature_columns=json.dumps(feature_columns),
            target_column=target_column,
            training_samples=training_samples,
            validation_samples=validation_samples,
            test_samples=test_samples,
            data_sources=json.dumps(data_sources),
            data_lineage_id=lineage_id
        )
        
        with self.schema.get_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO model_training_data 
                (model_type, dataset_name, feature_columns, target_column,
                 training_samples, validation_samples, test_samples, 
                 data_sources, data_lineage_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                training_data.model_type, training_data.dataset_name,
                training_data.feature_columns, training_data.target_column,
                training_data.training_samples, training_data.validation_samples,
                training_data.test_samples, training_data.data_sources,
                training_data.data_lineage_id, training_data.created_at
            ))
            conn.commit()
            return cursor.lastrowid
    
    def get_data_lineage(self, collection_id: str = None,
                        source: str = None,
                        start_date: datetime = None,
                        end_date: datetime = None,
                        limit: int = None) -> List[DataLineage]:
        """Retrieve data lineage records"""
        query = "SELECT * FROM data_lineage WHERE 1=1"
        params = []
        
        if collection_id:
            query += " AND collection_id = ?"
            params.append(collection_id)
        
        if source:
            query += " AND source = ?"
            params.append(source)
        
        if start_date:
            query += " AND start_time >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND start_time <= ?"
            params.append(end_date)
        
        query += " ORDER BY start_time DESC"
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        with self.schema.get_connection() as conn:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            
            return [
                DataLineage(
                    id=row["id"],
                    collection_id=row["collection_id"],
                    source=row["source"],
                    collection_method=row["collection_method"],
                    parameters=row["parameters"],
                    start_time=datetime.fromisoformat(row["start_time"]),
                    end_time=datetime.fromisoformat(row["end_time"]) if row["end_time"] else None,
                    records_collected=row["records_collected"],
                    success=bool(row["success"]),
                    error_message=row["error_message"],
                    created_at=datetime.fromisoformat(row["created_at"])
                )
                for row in rows
            ]
    
    def export_to_pandas(self, table_name: str, **filters) -> pd.DataFrame:
        """Export data to pandas DataFrame for analysis"""
        if table_name == "historical_price_data":
            data = self.get_historical_price_data(**filters)
            return pd.DataFrame([asdict(item) for item in data])
        elif table_name == "sentiment_data":
            data = self.get_sentiment_data(**filters)
            return pd.DataFrame([asdict(item) for item in data])
        else:
            raise ValueError(f"Unsupported table: {table_name}")
    
    def cleanup_old_data(self, days_to_keep: int = 90) -> Dict[str, int]:
        """Clean up old data beyond retention period"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        deleted_counts = {}
        
        with self.schema.get_connection() as conn:
            # Delete old historical price data
            cursor = conn.execute("""
                DELETE FROM historical_price_data 
                WHERE created_at < ?
            """, (cutoff_date,))
            deleted_counts["historical_price_data"] = cursor.rowcount
            
            # Delete old sentiment data
            cursor = conn.execute("""
                DELETE FROM sentiment_data 
                WHERE created_at < ?
            """, (cutoff_date,))
            deleted_counts["sentiment_data"] = cursor.rowcount
            
            # Delete old data lineage records
            cursor = conn.execute("""
                DELETE FROM data_lineage 
                WHERE created_at < ?
            """, (cutoff_date,))
            deleted_counts["data_lineage"] = cursor.rowcount
            
            conn.commit()
        
        return deleted_counts
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics"""
        return self.schema.get_database_stats()
    
    def vacuum_database(self) -> None:
        """Optimize database by running VACUUM"""
        with self.schema.get_connection() as conn:
            conn.execute("VACUUM")
            conn.commit()