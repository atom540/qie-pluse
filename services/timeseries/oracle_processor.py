"""
QIE Oracle Data Processor
Implements QIE oracle data fetcher for all seven asset feeds with technical indicators
and feature engineering pipeline
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import time
import json

from config import get_config
from logging_config import get_data_logger, get_performance_logger
from services.collectors.crypto_collector import CryptoDataCollector, CryptoPriceData
from services.collectors.traditional_asset_collector import TraditionalAssetCollector, TraditionalAssetData
from services.timeseries.advanced_feature_engine import AdvancedFeatureEngine

@dataclass
class OracleData:
    """Data structure for QIE oracle feed information"""
    asset_symbol: str
    current_price: float
    price_change_24h: float
    volume_24h: float
    rsi: Optional[float] = None
    moving_avg_20: Optional[float] = None
    moving_avg_50: Optional[float] = None
    volatility_index: Optional[float] = None
    last_updated: datetime = field(default_factory=datetime.now)
    source: str = "qie_oracle"

@dataclass
class TechnicalIndicators:
    """Comprehensive technical indicators for time-series analysis"""
    asset_symbol: str
    timestamp: datetime
    
    # Price-based indicators
    rsi: Optional[float] = None
    rsi_14: Optional[float] = None
    rsi_21: Optional[float] = None
    
    # Moving averages
    sma_5: Optional[float] = None
    sma_10: Optional[float] = None
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None
    
    # Volatility measures
    volatility_10d: Optional[float] = None
    volatility_30d: Optional[float] = None
    atr: Optional[float] = None  # Average True Range
    
    # Momentum indicators
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    
    # Volume indicators
    volume_sma_20: Optional[float] = None
    volume_ratio: Optional[float] = None
    
    # Bollinger Bands
    bb_upper: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower: Optional[float] = None
    bb_width: Optional[float] = None

@dataclass
class FeatureSet:
    """Feature engineering output with lag features and correlations"""
    asset_symbol: str
    timestamp: datetime
    
    # Current features
    current_price: float
    current_volume: float
    current_volatility: float
    
    # Lag features (previous periods)
    price_lag_1: Optional[float] = None
    price_lag_3: Optional[float] = None
    price_lag_7: Optional[float] = None
    volume_lag_1: Optional[float] = None
    volume_lag_3: Optional[float] = None
    
    # Price change features
    price_change_1h: Optional[float] = None
    price_change_4h: Optional[float] = None
    price_change_24h: Optional[float] = None
    price_change_7d: Optional[float] = None
    
    # Technical indicator features
    rsi_normalized: Optional[float] = None
    macd_normalized: Optional[float] = None
    bb_position: Optional[float] = None  # Position within Bollinger Bands
    
    # Cross-asset correlation features
    btc_correlation: Optional[float] = None
    gold_correlation: Optional[float] = None
    market_correlation: Optional[float] = None
    
    # Risk features
    volatility_percentile: Optional[float] = None
    volume_anomaly_score: Optional[float] = None
    price_momentum_score: Optional[float] = None
    
    # Advanced features - Volatility regime
    volatility_regime_score: Optional[float] = None  # 0=low, 0.5=medium, 1=high
    volatility_regime_percentile: Optional[float] = None
    current_volatility_normalized: Optional[float] = None
    
    # Advanced features - Multi-timeframe momentum
    rsi_1h: Optional[float] = None
    rsi_4h: Optional[float] = None
    rsi_24h: Optional[float] = None
    macd_1h: Optional[float] = None
    macd_4h: Optional[float] = None
    macd_24h: Optional[float] = None
    bb_position_1h: Optional[float] = None
    bb_position_4h: Optional[float] = None
    bb_position_24h: Optional[float] = None
    momentum_score_1h: Optional[float] = None
    momentum_score_4h: Optional[float] = None
    momentum_score_24h: Optional[float] = None
    
    # Advanced features - Volume anomaly
    volume_zscore: Optional[float] = None
    volume_anomaly_strength: Optional[float] = None
    is_volume_anomaly: Optional[float] = None  # 1.0 if anomaly, 0.0 if not
    
    # Advanced features - Enhanced correlations (multiple windows)
    btc_correlation_7d: Optional[float] = None
    btc_correlation_30d: Optional[float] = None
    btc_correlation_90d: Optional[float] = None
    gold_correlation_7d: Optional[float] = None
    gold_correlation_30d: Optional[float] = None
    gold_correlation_90d: Optional[float] = None

class QIEOracleClient:
    """Client for QIE blockchain oracle data feeds"""
    
    def __init__(self):
        self.config = get_config()
        self.data_logger = get_data_logger()
        self.performance_logger = get_performance_logger()
        
        # QIE Oracle endpoints (these would be actual QIE blockchain RPC endpoints)
        self.oracle_endpoints = {
            "BTC": self.config.qie_oracles.btc_feed_url if hasattr(self.config, 'qie_oracles') else "http://localhost:8545/btc",
            "ETH": self.config.qie_oracles.eth_feed_url if hasattr(self.config, 'qie_oracles') else "http://localhost:8545/eth", 
            "XRP": self.config.qie_oracles.xrp_feed_url if hasattr(self.config, 'qie_oracles') else "http://localhost:8545/xrp",
            "SOL": self.config.qie_oracles.sol_feed_url if hasattr(self.config, 'qie_oracles') else "http://localhost:8545/sol",
            "QIE": self.config.qie_oracles.qie_feed_url if hasattr(self.config, 'qie_oracles') else "http://localhost:8545/qie",
            "GOLD": self.config.qie_oracles.gold_feed_url if hasattr(self.config, 'qie_oracles') else "http://localhost:8545/gold",
            "BNB": self.config.qie_oracles.bnb_feed_url if hasattr(self.config, 'qie_oracles') else "http://localhost:8545/bnb"
        }
        
        # All seven asset feeds as required
        self.target_assets = ["BTC", "ETH", "XRP", "SOL", "QIE", "GOLD", "BNB"]
    
    async def fetch_oracle_data(self, asset: str) -> Optional[OracleData]:
        """Fetch data from QIE oracle for a specific asset"""
        start_time = time.time()
        
        try:
            endpoint = self.oracle_endpoints.get(asset)
            if not endpoint:
                raise ValueError(f"No oracle endpoint configured for asset: {asset}")
            
            # For now, we'll simulate oracle calls since actual QIE oracles may not be available
            # In production, this would make actual RPC calls to QIE blockchain oracles
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.get(endpoint, timeout=aiohttp.ClientTimeout(total=5)) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            oracle_data = OracleData(
                                asset_symbol=asset,
                                current_price=float(data.get("price", 0)),
                                price_change_24h=float(data.get("change_24h", 0)),
                                volume_24h=float(data.get("volume_24h", 0)),
                                rsi=data.get("rsi"),
                                moving_avg_20=data.get("sma_20"),
                                moving_avg_50=data.get("sma_50"),
                                volatility_index=data.get("volatility"),
                                last_updated=datetime.now(),
                                source="qie_oracle"
                            )
                            
                            duration_ms = (time.time() - start_time) * 1000
                            self.performance_logger.log_inference_time(f"qie_oracle_{asset.lower()}", duration_ms, True)
                            self.data_logger.log_data_collection("qie_oracle", asset, 1, True)
                            
                            return oracle_data
                        
                        else:
                            raise Exception(f"QIE Oracle API error {response.status} for {asset}")
                
                except asyncio.TimeoutError:
                    # Fallback to external data sources if QIE oracle is unavailable
                    self.data_logger.log_data_collection("qie_oracle", asset, 0, False, "Oracle timeout, using fallback")
                    return await self._fallback_data_fetch(asset)
                
                except aiohttp.ClientError:
                    # Fallback to external data sources
                    self.data_logger.log_data_collection("qie_oracle", asset, 0, False, "Oracle connection error, using fallback")
                    return await self._fallback_data_fetch(asset)
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time(f"qie_oracle_{asset.lower()}", duration_ms, False)
            self.data_logger.log_data_collection("qie_oracle", asset, 0, False, str(e))
            
            # Try fallback data source
            return await self._fallback_data_fetch(asset)
    
    async def _fallback_data_fetch(self, asset: str) -> Optional[OracleData]:
        """Fallback to external data sources when QIE oracle is unavailable"""
        try:
            if asset == "GOLD":
                # Use traditional asset collector for Gold
                traditional_collector = TraditionalAssetCollector()
                gold_data = await traditional_collector.collect_current_prices("alphavantage")
                
                if gold_data:
                    gold_item = gold_data[0]
                    return OracleData(
                        asset_symbol=asset,
                        current_price=gold_item.price,
                        price_change_24h=gold_item.price_change_24h,
                        volume_24h=gold_item.volume_24h,
                        last_updated=datetime.now(),
                        source="fallback_alphavantage"
                    )
            
            else:
                # Use crypto collector for crypto assets
                crypto_collector = CryptoDataCollector()
                crypto_data = await crypto_collector.collect_current_prices("coingecko")
                
                for crypto_item in crypto_data:
                    if crypto_item.symbol == asset:
                        return OracleData(
                            asset_symbol=asset,
                            current_price=crypto_item.price,
                            price_change_24h=crypto_item.price_change_24h,
                            volume_24h=crypto_item.volume_24h,
                            last_updated=datetime.now(),
                            source="fallback_coingecko"
                        )
            
            return None
        
        except Exception as e:
            self.data_logger.log_data_collection("fallback", asset, 0, False, str(e))
            return None
    
    async def fetch_all_oracle_data(self) -> Dict[str, OracleData]:
        """Fetch data from all seven QIE oracle feeds"""
        start_time = time.time()
        
        try:
            # Fetch all assets concurrently
            tasks = [self.fetch_oracle_data(asset) for asset in self.target_assets]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            oracle_data = {}
            successful_fetches = 0
            
            for i, result in enumerate(results):
                asset = self.target_assets[i]
                
                if isinstance(result, Exception):
                    self.data_logger.log_data_collection("qie_oracle", asset, 0, False, str(result))
                    continue
                
                if result is not None:
                    oracle_data[asset] = result
                    successful_fetches += 1
            
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time("qie_oracle_all_feeds", duration_ms, True)
            self.data_logger.log_data_collection("qie_oracle", "all_feeds", successful_fetches, True)
            
            return oracle_data
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time("qie_oracle_all_feeds", duration_ms, False)
            self.data_logger.log_data_collection("qie_oracle", "all_feeds", 0, False, str(e))
            raise

class TechnicalIndicatorCalculator:
    """Calculate comprehensive technical indicators from price data"""
    
    def __init__(self):
        self.data_logger = get_data_logger()
        self.performance_logger = get_performance_logger()
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return None
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None
    
    def calculate_moving_averages(self, prices: pd.Series) -> Dict[str, float]:
        """Calculate various moving averages"""
        mas = {}
        
        # Simple Moving Averages
        for period in [5, 10, 20, 50, 200]:
            if len(prices) >= period:
                ma = prices.rolling(window=period).mean().iloc[-1]
                mas[f"sma_{period}"] = float(ma) if not pd.isna(ma) else None
        
        # Exponential Moving Averages
        for period in [12, 26]:
            if len(prices) >= period:
                ema = prices.ewm(span=period).mean().iloc[-1]
                mas[f"ema_{period}"] = float(ema) if not pd.isna(ema) else None
        
        return mas
    
    def calculate_volatility(self, prices: pd.Series, periods: List[int] = [10, 30]) -> Dict[str, float]:
        """Calculate volatility measures"""
        volatilities = {}
        
        for period in periods:
            if len(prices) >= period:
                returns = prices.pct_change().dropna()
                if len(returns) >= period:
                    vol = returns.rolling(window=period).std().iloc[-1] * np.sqrt(252)  # Annualized
                    volatilities[f"volatility_{period}d"] = float(vol) if not pd.isna(vol) else None
        
        return volatilities
    
    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, float]:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        if len(prices) < slow + signal:
            return {"macd": None, "macd_signal": None, "macd_histogram": None}
        
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return {
            "macd": float(macd_line.iloc[-1]) if not pd.isna(macd_line.iloc[-1]) else None,
            "macd_signal": float(signal_line.iloc[-1]) if not pd.isna(signal_line.iloc[-1]) else None,
            "macd_histogram": float(histogram.iloc[-1]) if not pd.isna(histogram.iloc[-1]) else None
        }
    
    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> Dict[str, float]:
        """Calculate Bollinger Bands"""
        if len(prices) < period:
            return {"bb_upper": None, "bb_middle": None, "bb_lower": None, "bb_width": None}
        
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        width = (upper_band - lower_band) / sma
        
        return {
            "bb_upper": float(upper_band.iloc[-1]) if not pd.isna(upper_band.iloc[-1]) else None,
            "bb_middle": float(sma.iloc[-1]) if not pd.isna(sma.iloc[-1]) else None,
            "bb_lower": float(lower_band.iloc[-1]) if not pd.isna(lower_band.iloc[-1]) else None,
            "bb_width": float(width.iloc[-1]) if not pd.isna(width.iloc[-1]) else None
        }
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
        """Calculate Average True Range"""
        if len(high) < period + 1:
            return None
        
        high_low = high - low
        high_close_prev = np.abs(high - close.shift(1))
        low_close_prev = np.abs(low - close.shift(1))
        
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean().iloc[-1]
        
        return float(atr) if not pd.isna(atr) else None
    
    def calculate_all_indicators(self, price_data: List[Any], asset_symbol: str) -> TechnicalIndicators:
        """Calculate all technical indicators for an asset"""
        start_time = time.time()
        
        try:
            if not price_data or len(price_data) < 2:
                return TechnicalIndicators(asset_symbol=asset_symbol, timestamp=datetime.now())
            
            # Convert to DataFrame
            df = pd.DataFrame([
                {
                    "timestamp": getattr(data, 'timestamp', datetime.now()),
                    "price": getattr(data, 'price', getattr(data, 'current_price', 0)),
                    "volume": getattr(data, 'volume_24h', 0),
                    "high": getattr(data, 'price', getattr(data, 'current_price', 0)),  # Simplified
                    "low": getattr(data, 'price', getattr(data, 'current_price', 0)),   # Simplified
                }
                for data in price_data
            ]).sort_values("timestamp")
            
            if len(df) < 2:
                return TechnicalIndicators(asset_symbol=asset_symbol, timestamp=datetime.now())
            
            prices = df["price"]
            volumes = df["volume"]
            highs = df["high"]
            lows = df["low"]
            
            # Calculate all indicators
            indicators = TechnicalIndicators(
                asset_symbol=asset_symbol,
                timestamp=datetime.now()
            )
            
            # RSI
            indicators.rsi = self.calculate_rsi(prices, 14)
            indicators.rsi_14 = indicators.rsi
            indicators.rsi_21 = self.calculate_rsi(prices, 21)
            
            # Moving Averages
            mas = self.calculate_moving_averages(prices)
            indicators.sma_5 = mas.get("sma_5")
            indicators.sma_10 = mas.get("sma_10")
            indicators.sma_20 = mas.get("sma_20")
            indicators.sma_50 = mas.get("sma_50")
            indicators.sma_200 = mas.get("sma_200")
            indicators.ema_12 = mas.get("ema_12")
            indicators.ema_26 = mas.get("ema_26")
            
            # Volatility
            vols = self.calculate_volatility(prices)
            indicators.volatility_10d = vols.get("volatility_10d")
            indicators.volatility_30d = vols.get("volatility_30d")
            
            # MACD
            macd_data = self.calculate_macd(prices)
            indicators.macd = macd_data["macd"]
            indicators.macd_signal = macd_data["macd_signal"]
            indicators.macd_histogram = macd_data["macd_histogram"]
            
            # Bollinger Bands
            bb_data = self.calculate_bollinger_bands(prices)
            indicators.bb_upper = bb_data["bb_upper"]
            indicators.bb_middle = bb_data["bb_middle"]
            indicators.bb_lower = bb_data["bb_lower"]
            indicators.bb_width = bb_data["bb_width"]
            
            # ATR
            indicators.atr = self.calculate_atr(highs, lows, prices)
            
            # Volume indicators
            if len(volumes) >= 20:
                indicators.volume_sma_20 = float(volumes.rolling(window=20).mean().iloc[-1])
                current_volume = volumes.iloc[-1]
                avg_volume = indicators.volume_sma_20
                indicators.volume_ratio = float(current_volume / avg_volume) if avg_volume > 0 else None
            
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time(f"technical_indicators_{asset_symbol.lower()}", duration_ms, True)
            
            return indicators
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time(f"technical_indicators_{asset_symbol.lower()}", duration_ms, False)
            self.data_logger.log_data_collection("technical_indicators", asset_symbol, 0, False, str(e))
            return TechnicalIndicators(asset_symbol=asset_symbol, timestamp=datetime.now())

class FeatureEngineer:
    """Feature engineering pipeline with lag features and correlations"""
    
    def __init__(self):
        self.data_logger = get_data_logger()
        self.performance_logger = get_performance_logger()
        self.advanced_engine = AdvancedFeatureEngine()
    
    def create_lag_features(self, price_data: List[Any], periods: List[int] = [1, 3, 7]) -> Dict[str, float]:
        """Create lag features for previous periods"""
        if len(price_data) < max(periods) + 1:
            return {f"price_lag_{p}": None for p in periods} | {f"volume_lag_{p}": None for p in periods[:2]}
        
        # Sort by timestamp
        sorted_data = sorted(price_data, key=lambda x: getattr(x, 'timestamp', datetime.now()))
        
        lag_features = {}
        
        for period in periods:
            if len(sorted_data) > period:
                lag_data = sorted_data[-(period + 1)]
                lag_features[f"price_lag_{period}"] = getattr(lag_data, 'price', getattr(lag_data, 'current_price', None))
                
                if period <= 3:  # Only short-term volume lags
                    lag_features[f"volume_lag_{period}"] = getattr(lag_data, 'volume_24h', None)
        
        return lag_features
    
    def calculate_price_changes(self, price_data: List[Any]) -> Dict[str, float]:
        """Calculate price changes over different periods"""
        if len(price_data) < 2:
            return {"price_change_1h": None, "price_change_4h": None, "price_change_24h": None, "price_change_7d": None}
        
        # Convert to DataFrame for easier calculation
        df = pd.DataFrame([
            {
                "timestamp": getattr(data, 'timestamp', datetime.now()),
                "price": getattr(data, 'price', getattr(data, 'current_price', 0))
            }
            for data in price_data
        ]).sort_values("timestamp")
        
        current_price = df["price"].iloc[-1]
        changes = {}
        
        # Define time periods (approximate based on data frequency)
        time_periods = {
            "price_change_1h": timedelta(hours=1),
            "price_change_4h": timedelta(hours=4), 
            "price_change_24h": timedelta(hours=24),
            "price_change_7d": timedelta(days=7)
        }
        
        current_time = df["timestamp"].iloc[-1]
        
        for change_name, time_delta in time_periods.items():
            target_time = current_time - time_delta
            
            # Find closest data point to target time
            time_diffs = (df["timestamp"] - target_time).abs()
            closest_idx = time_diffs.idxmin()
            
            if time_diffs.loc[closest_idx] <= time_delta * 0.5:  # Within 50% of target period
                past_price = df.loc[closest_idx, "price"]
                if past_price > 0:
                    changes[change_name] = float((current_price - past_price) / past_price * 100)
                else:
                    changes[change_name] = None
            else:
                changes[change_name] = None
        
        return changes
    
    def normalize_technical_indicators(self, indicators: TechnicalIndicators) -> Dict[str, float]:
        """Normalize technical indicators for ML features"""
        normalized = {}
        
        # RSI is already 0-100, normalize to 0-1
        if indicators.rsi is not None:
            normalized["rsi_normalized"] = indicators.rsi / 100.0
        
        # MACD normalization (relative to price)
        if indicators.macd is not None and hasattr(indicators, 'current_price'):
            current_price = getattr(indicators, 'current_price', 1)
            normalized["macd_normalized"] = indicators.macd / current_price if current_price > 0 else None
        
        # Bollinger Band position (0 = lower band, 1 = upper band)
        if all(x is not None for x in [indicators.bb_upper, indicators.bb_lower]):
            current_price = getattr(indicators, 'current_price', 
                                  (indicators.bb_upper + indicators.bb_lower) / 2)
            bb_range = indicators.bb_upper - indicators.bb_lower
            if bb_range > 0:
                normalized["bb_position"] = (current_price - indicators.bb_lower) / bb_range
        
        return normalized
    
    def calculate_cross_asset_correlations(self, asset_data: Dict[str, List[Any]], 
                                         target_asset: str, window: int = 30) -> Dict[str, float]:
        """Calculate correlations with other assets using advanced correlation engine"""
        try:
            # Use the advanced correlation engine for more sophisticated correlation calculation
            correlation_matrix = self.advanced_engine.correlation_engine.calculate_correlation_matrix(asset_data)
            
            correlations = {}
            
            # Extract correlations for the target asset
            if target_asset in correlation_matrix.asset_pairs:
                asset_correlations = correlation_matrix.asset_pairs[target_asset]
                
                # Map to expected feature names
                correlations["btc_correlation"] = asset_correlations.get("BTC")
                correlations["gold_correlation"] = asset_correlations.get("GOLD")
                
                # Calculate market correlation (average with all other assets)
                other_correlations = [corr for asset, corr in asset_correlations.items() 
                                    if asset != target_asset and corr is not None]
                correlations["market_correlation"] = float(np.mean(other_correlations)) if other_correlations else None
            else:
                correlations = {"btc_correlation": None, "gold_correlation": None, "market_correlation": None}
            
            return correlations
            
        except Exception as e:
            self.data_logger.log_data_collection("correlation_calculation", target_asset, 0, False, str(e))
            # Fallback to original correlation calculation
            return self._calculate_basic_correlations(asset_data, target_asset, window)
    
    def _calculate_basic_correlations(self, asset_data: Dict[str, List[Any]], 
                                    target_asset: str, window: int = 30) -> Dict[str, float]:
        """Fallback basic correlation calculation"""
        correlations = {}
        
        if target_asset not in asset_data or len(asset_data[target_asset]) < window:
            return {"btc_correlation": None, "gold_correlation": None, "market_correlation": None}
        
        # Convert target asset to DataFrame
        target_df = pd.DataFrame([
            {
                "timestamp": getattr(data, 'timestamp', datetime.now()),
                "price": getattr(data, 'price', getattr(data, 'current_price', 0))
            }
            for data in asset_data[target_asset]
        ]).sort_values("timestamp")
        
        target_df["returns"] = target_df["price"].pct_change()
        
        # Calculate correlations with key assets
        correlation_assets = {"BTC": "btc_correlation", "GOLD": "gold_correlation"}
        
        for asset, corr_name in correlation_assets.items():
            if asset in asset_data and asset != target_asset and len(asset_data[asset]) >= window:
                asset_df = pd.DataFrame([
                    {
                        "timestamp": getattr(data, 'timestamp', datetime.now()),
                        "price": getattr(data, 'price', getattr(data, 'current_price', 0))
                    }
                    for data in asset_data[asset]
                ]).sort_values("timestamp")
                
                asset_df["returns"] = asset_df["price"].pct_change()
                
                # Merge on timestamp and calculate correlation
                merged = pd.merge_asof(
                    target_df[["timestamp", "returns"]].rename(columns={"returns": "target_returns"}),
                    asset_df[["timestamp", "returns"]].rename(columns={"returns": "asset_returns"}),
                    on="timestamp",
                    direction="nearest"
                ).dropna()
                
                if len(merged) >= window:
                    corr = merged["target_returns"].rolling(window=window).corr(merged["asset_returns"]).iloc[-1]
                    correlations[corr_name] = float(corr) if not pd.isna(corr) else None
                else:
                    correlations[corr_name] = None
            else:
                correlations[corr_name] = None
        
        # Market correlation (average correlation with all other assets)
        all_correlations = []
        for asset in asset_data:
            if asset != target_asset and len(asset_data[asset]) >= window:
                # Similar correlation calculation as above
                try:
                    asset_df = pd.DataFrame([
                        {
                            "timestamp": getattr(data, 'timestamp', datetime.now()),
                            "price": getattr(data, 'price', getattr(data, 'current_price', 0))
                        }
                        for data in asset_data[asset]
                    ]).sort_values("timestamp")
                    
                    asset_df["returns"] = asset_df["price"].pct_change()
                    
                    merged = pd.merge_asof(
                        target_df[["timestamp", "returns"]].rename(columns={"returns": "target_returns"}),
                        asset_df[["timestamp", "returns"]].rename(columns={"returns": "asset_returns"}),
                        on="timestamp",
                        direction="nearest"
                    ).dropna()
                    
                    if len(merged) >= window:
                        corr = merged["target_returns"].rolling(window=window).corr(merged["asset_returns"]).iloc[-1]
                        if not pd.isna(corr):
                            all_correlations.append(corr)
                
                except Exception:
                    continue
        
        correlations["market_correlation"] = float(np.mean(all_correlations)) if all_correlations else None
        
        return correlations
        
        # Market correlation (average correlation with all other assets)
        all_correlations = []
        for asset in asset_data:
            if asset != target_asset and len(asset_data[asset]) >= window:
                # Similar correlation calculation as above
                try:
                    asset_df = pd.DataFrame([
                        {
                            "timestamp": getattr(data, 'timestamp', datetime.now()),
                            "price": getattr(data, 'price', getattr(data, 'current_price', 0))
                        }
                        for data in asset_data[asset]
                    ]).sort_values("timestamp")
                    
                    asset_df["returns"] = asset_df["price"].pct_change()
                    
                    merged = pd.merge_asof(
                        target_df[["timestamp", "returns"]].rename(columns={"returns": "target_returns"}),
                        asset_df[["timestamp", "returns"]].rename(columns={"returns": "asset_returns"}),
                        on="timestamp",
                        direction="nearest"
                    ).dropna()
                    
                    if len(merged) >= window:
                        corr = merged["target_returns"].rolling(window=window).corr(merged["asset_returns"]).iloc[-1]
                        if not pd.isna(corr):
                            all_correlations.append(corr)
                
                except Exception:
                    continue
        
        correlations["market_correlation"] = float(np.mean(all_correlations)) if all_correlations else None
        
        return correlations
    
    def calculate_risk_features(self, price_data: List[Any], indicators: TechnicalIndicators) -> Dict[str, float]:
        """Calculate risk-related features"""
        risk_features = {}
        
        if len(price_data) < 10:
            return {"volatility_percentile": None, "volume_anomaly_score": None, "price_momentum_score": None}
        
        # Convert to DataFrame
        df = pd.DataFrame([
            {
                "timestamp": getattr(data, 'timestamp', datetime.now()),
                "price": getattr(data, 'price', getattr(data, 'current_price', 0)),
                "volume": getattr(data, 'volume_24h', 0)
            }
            for data in price_data
        ]).sort_values("timestamp")
        
        # Volatility percentile (current volatility vs historical)
        if indicators.volatility_10d is not None:
            returns = df["price"].pct_change().dropna()
            if len(returns) >= 30:
                rolling_vol = returns.rolling(window=10).std()
                current_vol = indicators.volatility_10d / np.sqrt(252)  # De-annualize
                vol_percentile = (rolling_vol <= current_vol).mean()
                risk_features["volatility_percentile"] = float(vol_percentile)
        
        # Volume anomaly score (current volume vs average)
        if len(df) >= 20:
            avg_volume = df["volume"].rolling(window=20).mean().iloc[-1]
            current_volume = df["volume"].iloc[-1]
            if avg_volume > 0:
                volume_ratio = current_volume / avg_volume
                # Z-score based anomaly detection
                volume_std = df["volume"].rolling(window=20).std().iloc[-1]
                if volume_std > 0:
                    z_score = (current_volume - avg_volume) / volume_std
                    risk_features["volume_anomaly_score"] = float(abs(z_score))
        
        # Price momentum score
        if len(df) >= 5:
            recent_returns = df["price"].pct_change().tail(5)
            momentum_score = recent_returns.mean() / recent_returns.std() if recent_returns.std() > 0 else 0
            risk_features["price_momentum_score"] = float(momentum_score)
        
        return risk_features
    
    def create_feature_set(self, asset_symbol: str, price_data: List[Any], 
                          indicators: TechnicalIndicators, all_asset_data: Dict[str, List[Any]]) -> FeatureSet:
        """Create comprehensive feature set for ML model"""
        start_time = time.time()
        
        try:
            if not price_data:
                return FeatureSet(
                    asset_symbol=asset_symbol,
                    timestamp=datetime.now(),
                    current_price=0,
                    current_volume=0,
                    current_volatility=0
                )
            
            # Get current values
            latest_data = price_data[-1]
            current_price = getattr(latest_data, 'price', getattr(latest_data, 'current_price', 0))
            current_volume = getattr(latest_data, 'volume_24h', 0)
            current_volatility = indicators.volatility_10d or 0
            
            # Create feature set
            features = FeatureSet(
                asset_symbol=asset_symbol,
                timestamp=datetime.now(),
                current_price=current_price,
                current_volume=current_volume,
                current_volatility=current_volatility
            )
            
            # Add lag features
            lag_features = self.create_lag_features(price_data)
            features.price_lag_1 = lag_features.get("price_lag_1")
            features.price_lag_3 = lag_features.get("price_lag_3")
            features.price_lag_7 = lag_features.get("price_lag_7")
            features.volume_lag_1 = lag_features.get("volume_lag_1")
            features.volume_lag_3 = lag_features.get("volume_lag_3")
            
            # Add price change features
            price_changes = self.calculate_price_changes(price_data)
            features.price_change_1h = price_changes.get("price_change_1h")
            features.price_change_4h = price_changes.get("price_change_4h")
            features.price_change_24h = price_changes.get("price_change_24h")
            features.price_change_7d = price_changes.get("price_change_7d")
            
            # Add normalized technical indicators
            normalized_indicators = self.normalize_technical_indicators(indicators)
            features.rsi_normalized = normalized_indicators.get("rsi_normalized")
            features.macd_normalized = normalized_indicators.get("macd_normalized")
            features.bb_position = normalized_indicators.get("bb_position")
            
            # Add cross-asset correlations
            correlations = self.calculate_cross_asset_correlations(all_asset_data, asset_symbol)
            features.btc_correlation = correlations.get("btc_correlation")
            features.gold_correlation = correlations.get("gold_correlation")
            features.market_correlation = correlations.get("market_correlation")
            
            # Add risk features
            risk_features = self.calculate_risk_features(price_data, indicators)
            features.volatility_percentile = risk_features.get("volatility_percentile")
            features.volume_anomaly_score = risk_features.get("volume_anomaly_score")
            features.price_momentum_score = risk_features.get("price_momentum_score")
            
            # Add advanced features using the advanced feature engine
            try:
                # Generate advanced features for this asset
                advanced_features = self.advanced_engine.generate_advanced_features({asset_symbol: price_data})
                
                # Extract volatility regime features
                volatility_regimes = advanced_features.get("volatility_regimes", {})
                if asset_symbol in volatility_regimes:
                    regime = volatility_regimes[asset_symbol]
                    features.volatility_regime_score = regime.get_regime_score()
                    features.volatility_regime_percentile = regime.volatility_percentile / 100.0
                    features.current_volatility_normalized = regime.current_volatility
                
                # Extract momentum indicator features
                momentum_indicators = advanced_features.get("momentum_indicators", {})
                if asset_symbol in momentum_indicators:
                    momentum = momentum_indicators[asset_symbol]
                    
                    # Multi-timeframe RSI
                    features.rsi_1h = getattr(momentum, 'rsi_1h', None)
                    features.rsi_4h = getattr(momentum, 'rsi_4h', None)
                    features.rsi_24h = getattr(momentum, 'rsi_24h', None)
                    
                    # Multi-timeframe MACD signals
                    macd_1h = getattr(momentum, 'macd_1h', None)
                    macd_signal_1h = getattr(momentum, 'macd_signal_1h', None)
                    if macd_1h is not None and macd_signal_1h is not None:
                        features.macd_1h = 1.0 if macd_1h > macd_signal_1h else 0.0
                    
                    macd_4h = getattr(momentum, 'macd_4h', None)
                    macd_signal_4h = getattr(momentum, 'macd_signal_4h', None)
                    if macd_4h is not None and macd_signal_4h is not None:
                        features.macd_4h = 1.0 if macd_4h > macd_signal_4h else 0.0
                    
                    macd_24h = getattr(momentum, 'macd_24h', None)
                    macd_signal_24h = getattr(momentum, 'macd_signal_24h', None)
                    if macd_24h is not None and macd_signal_24h is not None:
                        features.macd_24h = 1.0 if macd_24h > macd_signal_24h else 0.0
                    
                    # Bollinger Band positions
                    features.bb_position_1h = getattr(momentum, 'bb_position_1h', None)
                    features.bb_position_4h = getattr(momentum, 'bb_position_4h', None)
                    features.bb_position_24h = getattr(momentum, 'bb_position_24h', None)
                    
                    # Momentum scores
                    features.momentum_score_1h = momentum.get_momentum_score("1h")
                    features.momentum_score_4h = momentum.get_momentum_score("4h")
                    features.momentum_score_24h = momentum.get_momentum_score("24h")
                
                # Extract volume anomaly features
                volume_anomalies = advanced_features.get("volume_anomalies", {})
                if asset_symbol in volume_anomalies:
                    volume_anomaly = volume_anomalies[asset_symbol]
                    features.volume_zscore = volume_anomaly.volume_zscore
                    features.volume_anomaly_strength = volume_anomaly.get_anomaly_strength()
                    features.is_volume_anomaly = 1.0 if volume_anomaly.is_anomaly else 0.0
                
                # Extract enhanced correlation features (multiple windows)
                correlation_matrix = advanced_features.get("correlation_matrix")
                if correlation_matrix and asset_symbol in correlation_matrix.asset_pairs:
                    # 7-day correlations
                    if hasattr(correlation_matrix, 'window_7d') and asset_symbol in correlation_matrix.window_7d:
                        features.btc_correlation_7d = correlation_matrix.window_7d[asset_symbol].get("BTC")
                        features.gold_correlation_7d = correlation_matrix.window_7d[asset_symbol].get("GOLD")
                    
                    # 30-day correlations
                    if hasattr(correlation_matrix, 'window_30d') and asset_symbol in correlation_matrix.window_30d:
                        features.btc_correlation_30d = correlation_matrix.window_30d[asset_symbol].get("BTC")
                        features.gold_correlation_30d = correlation_matrix.window_30d[asset_symbol].get("GOLD")
                    
                    # 90-day correlations
                    if hasattr(correlation_matrix, 'window_90d') and asset_symbol in correlation_matrix.window_90d:
                        features.btc_correlation_90d = correlation_matrix.window_90d[asset_symbol].get("BTC")
                        features.gold_correlation_90d = correlation_matrix.window_90d[asset_symbol].get("GOLD")
            
            except Exception as e:
                # Log error but continue with basic features
                self.data_logger.log_data_collection("advanced_features", asset_symbol, 0, False, str(e))
            
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time(f"feature_engineering_{asset_symbol.lower()}", duration_ms, True)
            
            return features
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time(f"feature_engineering_{asset_symbol.lower()}", duration_ms, False)
            self.data_logger.log_data_collection("feature_engineering", asset_symbol, 0, False, str(e))
            
            return FeatureSet(
                asset_symbol=asset_symbol,
                timestamp=datetime.now(),
                current_price=0,
                current_volume=0,
                current_volatility=0
            )

class OracleDataProcessor:
    """Main oracle data processor combining all components"""
    
    def __init__(self):
        self.oracle_client = QIEOracleClient()
        self.indicator_calculator = TechnicalIndicatorCalculator()
        self.feature_engineer = FeatureEngineer()
        self.config = get_config()
        self.data_logger = get_data_logger()
        self.performance_logger = get_performance_logger()
        
        # Historical data storage for calculations
        self.historical_data: Dict[str, List[Any]] = {}
        self.max_history_length = 200  # Keep last 200 data points per asset
    
    async def process_all_oracle_feeds(self) -> Dict[str, Dict[str, Any]]:
        """Process all seven QIE oracle feeds with technical indicators and features"""
        start_time = time.time()
        
        try:
            # Fetch current oracle data
            oracle_data = await self.oracle_client.fetch_all_oracle_data()
            
            if not oracle_data:
                raise Exception("No oracle data received from any feed")
            
            # Update historical data
            self._update_historical_data(oracle_data)
            
            # Process each asset
            processed_data = {}
            
            for asset, current_data in oracle_data.items():
                try:
                    # Get historical data for this asset
                    asset_history = self.historical_data.get(asset, [current_data])
                    
                    # Calculate technical indicators
                    indicators = self.indicator_calculator.calculate_all_indicators(asset_history, asset)
                    
                    # Create feature set
                    features = self.feature_engineer.create_feature_set(
                        asset, asset_history, indicators, self.historical_data
                    )
                    
                    processed_data[asset] = {
                        "oracle_data": current_data,
                        "technical_indicators": indicators,
                        "feature_set": features,
                        "processing_timestamp": datetime.now()
                    }
                
                except Exception as e:
                    self.data_logger.log_data_collection("oracle_processing", asset, 0, False, str(e))
                    # Continue processing other assets
                    continue
            
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time("oracle_processing_all", duration_ms, True)
            self.data_logger.log_data_collection("oracle_processing", "all_assets", len(processed_data), True)
            
            return processed_data
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time("oracle_processing_all", duration_ms, False)
            self.data_logger.log_data_collection("oracle_processing", "all_assets", 0, False, str(e))
            raise
    
    def _update_historical_data(self, new_oracle_data: Dict[str, OracleData]):
        """Update historical data storage with new oracle data"""
        for asset, data in new_oracle_data.items():
            if asset not in self.historical_data:
                self.historical_data[asset] = []
            
            # Add new data point
            self.historical_data[asset].append(data)
            
            # Trim to max length
            if len(self.historical_data[asset]) > self.max_history_length:
                self.historical_data[asset] = self.historical_data[asset][-self.max_history_length:]
    
    async def get_asset_processing_result(self, asset: str) -> Optional[Dict[str, Any]]:
        """Get processing result for a specific asset"""
        try:
            oracle_data = await self.oracle_client.fetch_oracle_data(asset)
            if not oracle_data:
                return None
            
            # Update historical data
            self._update_historical_data({asset: oracle_data})
            
            # Get historical data
            asset_history = self.historical_data.get(asset, [oracle_data])
            
            # Calculate indicators and features
            indicators = self.indicator_calculator.calculate_all_indicators(asset_history, asset)
            features = self.feature_engineer.create_feature_set(
                asset, asset_history, indicators, self.historical_data
            )
            
            return {
                "oracle_data": oracle_data,
                "technical_indicators": indicators,
                "feature_set": features,
                "processing_timestamp": datetime.now()
            }
        
        except Exception as e:
            self.data_logger.log_data_collection("oracle_processing", asset, 0, False, str(e))
            return None
    
    def validate_processing_completeness(self, processed_data: Dict[str, Dict[str, Any]]) -> bool:
        """Validate that all seven asset feeds were processed successfully"""
        required_assets = {"BTC", "ETH", "XRP", "SOL", "QIE", "GOLD", "BNB"}
        processed_assets = set(processed_data.keys())
        
        missing_assets = required_assets - processed_assets
        if missing_assets:
            self.data_logger.log_data_collection("validation", "completeness", 0, False, 
                                               f"Missing assets: {missing_assets}")
            return False
        
        # Validate each asset has all required components
        for asset, data in processed_data.items():
            required_components = {"oracle_data", "technical_indicators", "feature_set"}
            if not all(component in data for component in required_components):
                self.data_logger.log_data_collection("validation", asset, 0, False, 
                                                   "Missing processing components")
                return False
        
        return True