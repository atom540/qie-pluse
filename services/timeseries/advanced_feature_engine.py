"""
Advanced Feature Engineering Engine for Enhanced AI Risk Oracle Model Training
Implements cross-asset correlations, volatility regimes, momentum indicators, and volume anomaly detection
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import time
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
import talib

from config import get_config
from logging_config import get_data_logger, get_performance_logger

@dataclass
class CorrelationMatrix:
    """Cross-asset correlation matrix with multiple time windows"""
    asset_pairs: Dict[str, Dict[str, float]]
    window_7d: Dict[str, Dict[str, float]]
    window_30d: Dict[str, Dict[str, float]]
    window_90d: Dict[str, Dict[str, float]]
    calculation_timestamp: datetime
    data_points_used: int
    
    def get_correlation(self, asset1: str, asset2: str, window: str = "30d") -> Optional[float]:
        """Get correlation between two assets for specified window"""
        window_data = getattr(self, f"window_{window}", self.window_30d)
        return window_data.get(asset1, {}).get(asset2)
    
    def get_strongest_correlations(self, asset: str, window: str = "30d", top_n: int = 3) -> List[Tuple[str, float]]:
        """Get strongest correlations for an asset"""
        window_data = getattr(self, f"window_{window}", self.window_30d)
        if asset not in window_data:
            return []
        
        correlations = [(other_asset, corr) for other_asset, corr in window_data[asset].items() 
                       if corr is not None and other_asset != asset]
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        return correlations[:top_n]

@dataclass
class VolatilityRegime:
    """Volatility regime classification"""
    asset: str
    regime: str  # "low", "medium", "high"
    volatility_percentile: float
    current_volatility: float
    historical_volatility_stats: Dict[str, float]
    timestamp: datetime
    
    def is_high_volatility(self) -> bool:
        return self.regime == "high"
    
    def get_regime_score(self) -> float:
        """Get numerical score for regime (0=low, 0.5=medium, 1=high)"""
        regime_scores = {"low": 0.0, "medium": 0.5, "high": 1.0}
        return regime_scores.get(self.regime, 0.5)

@dataclass
class MomentumIndicators:
    """Multi-timeframe momentum indicators"""
    asset: str
    timestamp: datetime
    
    # 1-hour timeframe
    rsi_1h: Optional[float] = None
    macd_1h: Optional[float] = None
    macd_signal_1h: Optional[float] = None
    bb_position_1h: Optional[float] = None  # Position within Bollinger Bands (0-1)
    
    # 4-hour timeframe  
    rsi_4h: Optional[float] = None
    macd_4h: Optional[float] = None
    macd_signal_4h: Optional[float] = None
    bb_position_4h: Optional[float] = None
    
    # 24-hour timeframe
    rsi_24h: Optional[float] = None
    macd_24h: Optional[float] = None
    macd_signal_24h: Optional[float] = None
    bb_position_24h: Optional[float] = None
    
    def get_momentum_score(self, timeframe: str = "4h") -> float:
        """Get combined momentum score for timeframe"""
        rsi = getattr(self, f"rsi_{timeframe}", None)
        macd = getattr(self, f"macd_{timeframe}", None)
        macd_signal = getattr(self, f"macd_signal_{timeframe}", None)
        bb_position = getattr(self, f"bb_position_{timeframe}", None)
        
        scores = []
        
        # RSI momentum (0-1 scale)
        if rsi is not None:
            scores.append(rsi / 100.0)
        
        # MACD momentum (bullish/bearish)
        if macd is not None and macd_signal is not None:
            macd_momentum = 1.0 if macd > macd_signal else 0.0
            scores.append(macd_momentum)
        
        # Bollinger Band position
        if bb_position is not None:
            scores.append(bb_position)
        
        return np.mean(scores) if scores else 0.5

@dataclass
class VolumeAnomalyScore:
    """Volume anomaly detection results"""
    asset: str
    current_volume: float
    volume_zscore: float
    historical_avg_volume: float
    historical_std_volume: float
    anomaly_threshold: float
    is_anomaly: bool
    timestamp: datetime
    
    def get_anomaly_strength(self) -> float:
        """Get anomaly strength (0-1 scale)"""
        return min(abs(self.volume_zscore) / 3.0, 1.0)  # Normalize to 0-1

class CrossAssetCorrelationEngine:
    """Advanced cross-asset correlation calculation engine"""
    
    def __init__(self):
        self.config = get_config()
        self.data_logger = get_data_logger()
        self.performance_logger = get_performance_logger()
        
        # Configuration
        self.correlation_windows = [7, 30, 90]  # days
        self.min_data_points = 20
        self.significance_threshold = 0.05
    
    def calculate_correlation_matrix(self, asset_data: Dict[str, List[Any]]) -> CorrelationMatrix:
        """Calculate comprehensive correlation matrix for all asset pairs"""
        start_time = time.time()
        
        try:
            # Prepare price series for all assets
            asset_series = {}
            for asset, data in asset_data.items():
                if len(data) >= self.min_data_points:
                    series = self._prepare_price_series(data, asset)
                    if not series.empty:
                        asset_series[asset] = series
            
            if len(asset_series) < 2:
                return self._empty_correlation_matrix()
            
            # Calculate correlations for each window
            correlation_matrices = {}
            for window_days in self.correlation_windows:
                correlation_matrices[f"window_{window_days}d"] = self._calculate_window_correlations(
                    asset_series, window_days
                )
            
            # Create overall correlation matrix (using 30d as default)
            asset_pairs = correlation_matrices.get("window_30d", {})
            
            correlation_matrix = CorrelationMatrix(
                asset_pairs=asset_pairs,
                window_7d=correlation_matrices.get("window_7d", {}),
                window_30d=correlation_matrices.get("window_30d", {}),
                window_90d=correlation_matrices.get("window_90d", {}),
                calculation_timestamp=datetime.now(),
                data_points_used=sum(len(series) for series in asset_series.values())
            )
            
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time("correlation_matrix_calculation", duration_ms, True)
            
            return correlation_matrix
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time("correlation_matrix_calculation", duration_ms, False)
            self.data_logger.log_data_collection("correlation_matrix", "calculation", 0, False, str(e))
            return self._empty_correlation_matrix()
    
    def _calculate_window_correlations(self, asset_series: Dict[str, pd.DataFrame], 
                                     window_days: int) -> Dict[str, Dict[str, float]]:
        """Calculate correlations for a specific time window"""
        correlations = {}
        assets = list(asset_series.keys())
        
        for i, asset1 in enumerate(assets):
            correlations[asset1] = {}
            
            for asset2 in assets:
                if asset1 == asset2:
                    correlations[asset1][asset2] = 1.0
                    continue
                
                try:
                    # Get recent data for the window
                    cutoff_date = datetime.now() - timedelta(days=window_days)
                    
                    series1 = asset_series[asset1]
                    series2 = asset_series[asset2]
                    
                    # Filter to window period
                    series1_window = series1[series1['timestamp'] >= cutoff_date]
                    series2_window = series2[series2['timestamp'] >= cutoff_date]
                    
                    if len(series1_window) < 10 or len(series2_window) < 10:
                        correlations[asset1][asset2] = None
                        continue
                    
                    # Merge on timestamp for alignment
                    merged = pd.merge_asof(
                        series1_window.sort_values('timestamp'),
                        series2_window.sort_values('timestamp'),
                        on='timestamp',
                        suffixes=('_1', '_2'),
                        direction='nearest',
                        tolerance=pd.Timedelta(hours=2)
                    ).dropna()
                    
                    if len(merged) < 10:
                        correlations[asset1][asset2] = None
                        continue
                    
                    # Calculate Pearson correlation
                    corr_coef, p_value = pearsonr(merged['returns_1'], merged['returns_2'])
                    
                    # Only include significant correlations
                    if p_value <= self.significance_threshold:
                        correlations[asset1][asset2] = float(corr_coef)
                    else:
                        correlations[asset1][asset2] = None
                
                except Exception as e:
                    self.data_logger.log_data_collection(
                        "correlation_calculation", f"{asset1}_{asset2}", 0, False, str(e)
                    )
                    correlations[asset1][asset2] = None
        
        return correlations
    
    def _prepare_price_series(self, asset_data: List[Any], asset_name: str) -> pd.DataFrame:
        """Convert asset data to DataFrame with returns"""
        try:
            data_points = []
            
            for item in asset_data:
                if hasattr(item, 'current_price'):
                    # OracleData format
                    data_points.append({
                        'timestamp': getattr(item, 'last_updated', datetime.now()),
                        'price': item.current_price,
                        'volume': getattr(item, 'volume_24h', 0)
                    })
                elif hasattr(item, 'price'):
                    # CryptoPriceData or TraditionalAssetData format
                    data_points.append({
                        'timestamp': item.timestamp,
                        'price': item.price,
                        'volume': getattr(item, 'volume_24h', 0)
                    })
                else:
                    continue
            
            if not data_points:
                return pd.DataFrame()
            
            df = pd.DataFrame(data_points).sort_values('timestamp')
            
            # Calculate returns
            df['returns'] = df['price'].pct_change()
            df = df.dropna()
            
            return df
        
        except Exception as e:
            self.data_logger.log_data_collection("price_series_preparation", asset_name, 0, False, str(e))
            return pd.DataFrame()
    
    def _empty_correlation_matrix(self) -> CorrelationMatrix:
        """Return empty correlation matrix"""
        return CorrelationMatrix(
            asset_pairs={},
            window_7d={},
            window_30d={},
            window_90d={},
            calculation_timestamp=datetime.now(),
            data_points_used=0
        )
    
    def get_rolling_correlations(self, asset_data: Dict[str, List[Any]], 
                               asset1: str, asset2: str, window_days: int = 30) -> pd.Series:
        """Calculate rolling correlations between two assets"""
        try:
            if asset1 not in asset_data or asset2 not in asset_data:
                return pd.Series(dtype=float)
            
            series1 = self._prepare_price_series(asset_data[asset1], asset1)
            series2 = self._prepare_price_series(asset_data[asset2], asset2)
            
            if series1.empty or series2.empty:
                return pd.Series(dtype=float)
            
            # Merge series
            merged = pd.merge_asof(
                series1.sort_values('timestamp'),
                series2.sort_values('timestamp'),
                on='timestamp',
                suffixes=('_1', '_2'),
                direction='nearest',
                tolerance=pd.Timedelta(hours=2)
            ).dropna()
            
            if len(merged) < window_days:
                return pd.Series(dtype=float)
            
            # Calculate rolling correlation
            rolling_corr = merged['returns_1'].rolling(window=window_days).corr(merged['returns_2'])
            rolling_corr.index = merged['timestamp']
            
            return rolling_corr
        
        except Exception as e:
            self.data_logger.log_data_collection(
                "rolling_correlation", f"{asset1}_{asset2}", 0, False, str(e)
            )
            return pd.Series(dtype=float)

class VolatilityRegimeClassifier:
    """Classify volatility regimes (low, medium, high)"""
    
    def __init__(self):
        self.data_logger = get_data_logger()
        self.performance_logger = get_performance_logger()
        
        # Configuration
        self.lookback_period = 90  # days for historical volatility calculation
        self.low_threshold = 20    # percentile
        self.high_threshold = 80   # percentile
    
    def classify_volatility_regime(self, asset_data: List[Any], asset: str) -> VolatilityRegime:
        """Classify current volatility regime for an asset"""
        start_time = time.time()
        
        try:
            if len(asset_data) < 30:  # Need minimum data
                return self._default_regime(asset)
            
            # Prepare price series
            df = self._prepare_price_series(asset_data, asset)
            if df.empty:
                return self._default_regime(asset)
            
            # Calculate rolling volatility (20-day window)
            df['volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(252)  # Annualized
            df = df.dropna()
            
            if len(df) < 20:
                return self._default_regime(asset)
            
            # Get current volatility
            current_volatility = df['volatility'].iloc[-1]
            
            # Calculate historical percentiles
            historical_volatilities = df['volatility'].values
            percentile_rank = (historical_volatilities < current_volatility).mean() * 100
            
            # Classify regime
            if percentile_rank <= self.low_threshold:
                regime = "low"
            elif percentile_rank >= self.high_threshold:
                regime = "high"
            else:
                regime = "medium"
            
            # Calculate historical stats
            historical_stats = {
                "mean": float(np.mean(historical_volatilities)),
                "std": float(np.std(historical_volatilities)),
                "min": float(np.min(historical_volatilities)),
                "max": float(np.max(historical_volatilities)),
                "median": float(np.median(historical_volatilities))
            }
            
            volatility_regime = VolatilityRegime(
                asset=asset,
                regime=regime,
                volatility_percentile=percentile_rank,
                current_volatility=current_volatility,
                historical_volatility_stats=historical_stats,
                timestamp=datetime.now()
            )
            
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time("volatility_regime_classification", duration_ms, True)
            
            return volatility_regime
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time("volatility_regime_classification", duration_ms, False)
            self.data_logger.log_data_collection("volatility_regime", asset, 0, False, str(e))
            return self._default_regime(asset)
    
    def _prepare_price_series(self, asset_data: List[Any], asset_name: str) -> pd.DataFrame:
        """Convert asset data to DataFrame with returns"""
        try:
            data_points = []
            
            for item in asset_data:
                if hasattr(item, 'current_price'):
                    data_points.append({
                        'timestamp': getattr(item, 'last_updated', datetime.now()),
                        'price': item.current_price
                    })
                elif hasattr(item, 'price'):
                    data_points.append({
                        'timestamp': item.timestamp,
                        'price': item.price
                    })
                else:
                    continue
            
            if not data_points:
                return pd.DataFrame()
            
            df = pd.DataFrame(data_points).sort_values('timestamp')
            df['returns'] = df['price'].pct_change()
            df = df.dropna()
            
            return df
        
        except Exception as e:
            self.data_logger.log_data_collection("price_series_preparation", asset_name, 0, False, str(e))
            return pd.DataFrame()
    
    def _default_regime(self, asset: str) -> VolatilityRegime:
        """Return default medium volatility regime"""
        return VolatilityRegime(
            asset=asset,
            regime="medium",
            volatility_percentile=50.0,
            current_volatility=0.0,
            historical_volatility_stats={
                "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0
            },
            timestamp=datetime.now()
        )

class MultiTimeframeMomentumCalculator:
    """Calculate momentum indicators across multiple timeframes"""
    
    def __init__(self):
        self.data_logger = get_data_logger()
        self.performance_logger = get_performance_logger()
        
        # Timeframe configurations (in hours)
        self.timeframes = {
            "1h": 1,
            "4h": 4,
            "24h": 24
        }
    
    def calculate_momentum_indicators(self, asset_data: List[Any], asset: str) -> MomentumIndicators:
        """Calculate momentum indicators for all timeframes"""
        start_time = time.time()
        
        try:
            if len(asset_data) < 50:  # Need sufficient data for indicators
                return self._default_momentum_indicators(asset)
            
            # Prepare price series
            df = self._prepare_ohlcv_series(asset_data, asset)
            if df.empty or len(df) < 50:
                return self._default_momentum_indicators(asset)
            
            momentum_indicators = MomentumIndicators(asset=asset, timestamp=datetime.now())
            
            # Calculate indicators for each timeframe
            for timeframe, hours in self.timeframes.items():
                try:
                    # Resample data to timeframe
                    resampled_df = self._resample_to_timeframe(df, hours)
                    
                    if len(resampled_df) < 30:  # Need minimum data for indicators
                        continue
                    
                    # Calculate RSI
                    rsi = self._calculate_rsi(resampled_df['close'].values)
                    if len(rsi) > 0:
                        setattr(momentum_indicators, f"rsi_{timeframe}", float(rsi[-1]))
                    
                    # Calculate MACD
                    macd, macd_signal, _ = self._calculate_macd(resampled_df['close'].values)
                    if len(macd) > 0 and len(macd_signal) > 0:
                        setattr(momentum_indicators, f"macd_{timeframe}", float(macd[-1]))
                        setattr(momentum_indicators, f"macd_signal_{timeframe}", float(macd_signal[-1]))
                    
                    # Calculate Bollinger Band position
                    bb_position = self._calculate_bb_position(resampled_df['close'].values)
                    if bb_position is not None:
                        setattr(momentum_indicators, f"bb_position_{timeframe}", float(bb_position))
                
                except Exception as e:
                    self.data_logger.log_data_collection(
                        "momentum_calculation", f"{asset}_{timeframe}", 0, False, str(e)
                    )
                    continue
            
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time("momentum_indicators_calculation", duration_ms, True)
            
            return momentum_indicators
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time("momentum_indicators_calculation", duration_ms, False)
            self.data_logger.log_data_collection("momentum_indicators", asset, 0, False, str(e))
            return self._default_momentum_indicators(asset)
    
    def _prepare_ohlcv_series(self, asset_data: List[Any], asset_name: str) -> pd.DataFrame:
        """Convert asset data to OHLCV DataFrame"""
        try:
            data_points = []
            
            for item in asset_data:
                if hasattr(item, 'current_price'):
                    price = item.current_price
                    volume = getattr(item, 'volume_24h', 0)
                    timestamp = getattr(item, 'last_updated', datetime.now())
                elif hasattr(item, 'price'):
                    price = item.price
                    volume = getattr(item, 'volume_24h', 0)
                    timestamp = item.timestamp
                else:
                    continue
                
                # For simplicity, use price as OHLC (can be enhanced with actual OHLC data)
                data_points.append({
                    'timestamp': timestamp,
                    'open': price,
                    'high': price,
                    'low': price,
                    'close': price,
                    'volume': volume
                })
            
            if not data_points:
                return pd.DataFrame()
            
            df = pd.DataFrame(data_points).sort_values('timestamp')
            return df
        
        except Exception as e:
            self.data_logger.log_data_collection("ohlcv_preparation", asset_name, 0, False, str(e))
            return pd.DataFrame()
    
    def _resample_to_timeframe(self, df: pd.DataFrame, hours: int) -> pd.DataFrame:
        """Resample data to specified timeframe"""
        try:
            df_copy = df.copy()
            df_copy.set_index('timestamp', inplace=True)
            
            # Resample to timeframe
            resampled = df_copy.resample(f'{hours}h').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            return resampled.reset_index()
        
        except Exception as e:
            self.data_logger.log_data_collection("timeframe_resampling", f"{hours}h", 0, False, str(e))
            return pd.DataFrame()
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate RSI using TA-Lib"""
        try:
            if len(prices) < period + 1:
                return np.array([])
            return talib.RSI(prices.astype(float), timeperiod=period)
        except Exception:
            # Fallback manual calculation
            return self._manual_rsi(prices, period)
    
    def _manual_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Manual RSI calculation fallback"""
        try:
            if len(prices) < period + 1:
                return np.array([])
            
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gains = pd.Series(gains).rolling(window=period).mean()
            avg_losses = pd.Series(losses).rolling(window=period).mean()
            
            rs = avg_gains / avg_losses
            rsi = 100 - (100 / (1 + rs))
            
            return rsi.values
        except Exception:
            return np.array([])
    
    def _calculate_macd(self, prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate MACD using TA-Lib"""
        try:
            if len(prices) < slow + signal:
                return np.array([]), np.array([]), np.array([])
            return talib.MACD(prices.astype(float), fastperiod=fast, slowperiod=slow, signalperiod=signal)
        except Exception:
            # Fallback manual calculation
            return self._manual_macd(prices, fast, slow, signal)
    
    def _manual_macd(self, prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Manual MACD calculation fallback"""
        try:
            if len(prices) < slow + signal:
                return np.array([]), np.array([]), np.array([])
            
            prices_series = pd.Series(prices)
            ema_fast = prices_series.ewm(span=fast).mean()
            ema_slow = prices_series.ewm(span=slow).mean()
            
            macd = ema_fast - ema_slow
            macd_signal = macd.ewm(span=signal).mean()
            macd_histogram = macd - macd_signal
            
            return macd.values, macd_signal.values, macd_histogram.values
        except Exception:
            return np.array([]), np.array([]), np.array([])
    
    def _calculate_bb_position(self, prices: np.ndarray, period: int = 20, std_dev: int = 2) -> Optional[float]:
        """Calculate position within Bollinger Bands"""
        try:
            if len(prices) < period:
                return None
            
            # Calculate Bollinger Bands
            prices_series = pd.Series(prices)
            sma = prices_series.rolling(window=period).mean()
            std = prices_series.rolling(window=period).std()
            
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
            
            # Calculate position (0 = lower band, 1 = upper band)
            current_price = prices[-1]
            current_upper = upper_band.iloc[-1]
            current_lower = lower_band.iloc[-1]
            
            if pd.isna(current_upper) or pd.isna(current_lower):
                return None
            
            band_width = current_upper - current_lower
            if band_width <= 0:
                return 0.5
            
            position = (current_price - current_lower) / band_width
            return max(0.0, min(1.0, position))  # Clamp to 0-1
        
        except Exception:
            return None
    
    def _default_momentum_indicators(self, asset: str) -> MomentumIndicators:
        """Return default momentum indicators"""
        return MomentumIndicators(asset=asset, timestamp=datetime.now())

class VolumeAnomalyDetector:
    """Detect volume anomalies relative to historical averages"""
    
    def __init__(self):
        self.data_logger = get_data_logger()
        self.performance_logger = get_performance_logger()
        
        # Configuration
        self.lookback_period = 30  # days for historical average
        self.anomaly_threshold = 2.0  # Z-score threshold
    
    def detect_volume_anomaly(self, asset_data: List[Any], asset: str) -> VolumeAnomalyScore:
        """Detect volume anomalies for an asset"""
        start_time = time.time()
        
        try:
            if len(asset_data) < self.lookback_period:
                return self._default_volume_score(asset)
            
            # Extract volume data
            volumes = []
            timestamps = []
            
            for item in asset_data:
                volume = None
                timestamp = None
                
                if hasattr(item, 'volume_24h'):
                    volume = item.volume_24h
                    timestamp = getattr(item, 'timestamp', getattr(item, 'last_updated', datetime.now()))
                elif hasattr(item, 'volume'):
                    volume = item.volume
                    timestamp = getattr(item, 'timestamp', datetime.now())
                
                if volume is not None and volume > 0:
                    volumes.append(volume)
                    timestamps.append(timestamp)
            
            if len(volumes) < self.lookback_period:
                return self._default_volume_score(asset)
            
            # Calculate historical statistics
            volumes_array = np.array(volumes)
            historical_avg = np.mean(volumes_array[:-1])  # Exclude current volume
            historical_std = np.std(volumes_array[:-1])
            
            # Current volume
            current_volume = volumes[-1]
            
            # Calculate Z-score
            if historical_std > 0:
                volume_zscore = (current_volume - historical_avg) / historical_std
            else:
                volume_zscore = 0.0
            
            # Determine if anomaly
            is_anomaly = abs(volume_zscore) >= self.anomaly_threshold
            
            volume_anomaly_score = VolumeAnomalyScore(
                asset=asset,
                current_volume=current_volume,
                volume_zscore=volume_zscore,
                historical_avg_volume=historical_avg,
                historical_std_volume=historical_std,
                anomaly_threshold=self.anomaly_threshold,
                is_anomaly=is_anomaly,
                timestamp=datetime.now()
            )
            
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time("volume_anomaly_detection", duration_ms, True)
            
            return volume_anomaly_score
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time("volume_anomaly_detection", duration_ms, False)
            self.data_logger.log_data_collection("volume_anomaly", asset, 0, False, str(e))
            return self._default_volume_score(asset)
    
    def _default_volume_score(self, asset: str) -> VolumeAnomalyScore:
        """Return default volume anomaly score"""
        return VolumeAnomalyScore(
            asset=asset,
            current_volume=0.0,
            volume_zscore=0.0,
            historical_avg_volume=0.0,
            historical_std_volume=0.0,
            anomaly_threshold=self.anomaly_threshold,
            is_anomaly=False,
            timestamp=datetime.now()
        )

class AdvancedFeatureEngine:
    """Main advanced feature engineering engine"""
    
    def __init__(self):
        self.correlation_engine = CrossAssetCorrelationEngine()
        self.volatility_classifier = VolatilityRegimeClassifier()
        self.momentum_calculator = MultiTimeframeMomentumCalculator()
        self.volume_detector = VolumeAnomalyDetector()
        
        self.data_logger = get_data_logger()
        self.performance_logger = get_performance_logger()
    
    def generate_advanced_features(self, asset_data: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Generate all advanced features for the asset data"""
        start_time = time.time()
        
        try:
            advanced_features = {
                "correlation_matrix": None,
                "volatility_regimes": {},
                "momentum_indicators": {},
                "volume_anomalies": {},
                "generation_timestamp": datetime.now()
            }
            
            # Generate correlation matrix
            advanced_features["correlation_matrix"] = self.correlation_engine.calculate_correlation_matrix(asset_data)
            
            # Generate features for each asset
            for asset, data in asset_data.items():
                if not data:
                    continue
                
                try:
                    # Volatility regime classification
                    volatility_regime = self.volatility_classifier.classify_volatility_regime(data, asset)
                    advanced_features["volatility_regimes"][asset] = volatility_regime
                    
                    # Multi-timeframe momentum indicators
                    momentum_indicators = self.momentum_calculator.calculate_momentum_indicators(data, asset)
                    advanced_features["momentum_indicators"][asset] = momentum_indicators
                    
                    # Volume anomaly detection
                    volume_anomaly = self.volume_detector.detect_volume_anomaly(data, asset)
                    advanced_features["volume_anomalies"][asset] = volume_anomaly
                
                except Exception as e:
                    self.data_logger.log_data_collection(
                        "advanced_features", asset, 0, False, str(e)
                    )
                    continue
            
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time("advanced_feature_generation", duration_ms, True)
            
            return advanced_features
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time("advanced_feature_generation", duration_ms, False)
            self.data_logger.log_data_collection("advanced_features", "all", 0, False, str(e))
            return {
                "correlation_matrix": self.correlation_engine._empty_correlation_matrix(),
                "volatility_regimes": {},
                "momentum_indicators": {},
                "volume_anomalies": {},
                "generation_timestamp": datetime.now()
            }
    
    def extract_feature_vector(self, advanced_features: Dict[str, Any], target_asset: str) -> Dict[str, float]:
        """Extract numerical feature vector for ML training"""
        try:
            feature_vector = {}
            
            # Correlation features
            correlation_matrix = advanced_features.get("correlation_matrix")
            if correlation_matrix and target_asset in correlation_matrix.asset_pairs:
                asset_correlations = correlation_matrix.asset_pairs[target_asset]
                for other_asset, corr in asset_correlations.items():
                    if other_asset != target_asset and corr is not None:
                        feature_vector[f"corr_{other_asset.lower()}_30d"] = corr
                
                # Add different window correlations
                for window in ["7d", "30d", "90d"]:
                    window_data = getattr(correlation_matrix, f"window_{window}", {})
                    if target_asset in window_data:
                        for other_asset, corr in window_data[target_asset].items():
                            if other_asset != target_asset and corr is not None:
                                feature_vector[f"corr_{other_asset.lower()}_{window}"] = corr
            
            # Volatility regime features
            volatility_regimes = advanced_features.get("volatility_regimes", {})
            if target_asset in volatility_regimes:
                regime = volatility_regimes[target_asset]
                feature_vector["volatility_regime_score"] = regime.get_regime_score()
                feature_vector["volatility_percentile"] = regime.volatility_percentile / 100.0
                feature_vector["current_volatility"] = regime.current_volatility
            
            # Momentum indicator features
            momentum_indicators = advanced_features.get("momentum_indicators", {})
            if target_asset in momentum_indicators:
                momentum = momentum_indicators[target_asset]
                
                # Add all timeframe indicators
                for timeframe in ["1h", "4h", "24h"]:
                    rsi = getattr(momentum, f"rsi_{timeframe}", None)
                    if rsi is not None:
                        feature_vector[f"rsi_{timeframe}"] = rsi / 100.0
                    
                    macd = getattr(momentum, f"macd_{timeframe}", None)
                    macd_signal = getattr(momentum, f"macd_signal_{timeframe}", None)
                    if macd is not None and macd_signal is not None:
                        feature_vector[f"macd_{timeframe}"] = 1.0 if macd > macd_signal else 0.0
                    
                    bb_position = getattr(momentum, f"bb_position_{timeframe}", None)
                    if bb_position is not None:
                        feature_vector[f"bb_position_{timeframe}"] = bb_position
                    
                    # Combined momentum score
                    momentum_score = momentum.get_momentum_score(timeframe)
                    feature_vector[f"momentum_score_{timeframe}"] = momentum_score
            
            # Volume anomaly features
            volume_anomalies = advanced_features.get("volume_anomalies", {})
            if target_asset in volume_anomalies:
                volume_anomaly = volume_anomalies[target_asset]
                feature_vector["volume_zscore"] = volume_anomaly.volume_zscore
                feature_vector["volume_anomaly_strength"] = volume_anomaly.get_anomaly_strength()
                feature_vector["is_volume_anomaly"] = 1.0 if volume_anomaly.is_anomaly else 0.0
            
            return feature_vector
        
        except Exception as e:
            self.data_logger.log_data_collection("feature_extraction", target_asset, 0, False, str(e))
            return {}