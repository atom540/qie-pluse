"""
Cross-Asset Correlation Detection Engine
Implements correlation analysis functions for Gold-crypto relationships and pattern recognition
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

from config import get_config
from logging_config import get_data_logger, get_performance_logger
from services.timeseries.oracle_processor import OracleData, FeatureSet

@dataclass
class CrossAssetCorrelation:
    """Cross-asset correlation analysis results"""
    asset1: str
    asset2: str
    correlation_coefficient: float
    p_value: float
    analysis_period_days: int
    timestamp: datetime
    confidence_level: float
    correlation_type: str = "pearson"  # pearson, spearman, kendall
    window_size: int = 30
    
    def is_significant(self, alpha: float = 0.05) -> bool:
        """Check if correlation is statistically significant"""
        return self.p_value < alpha
    
    def get_strength_category(self) -> str:
        """Categorize correlation strength"""
        abs_corr = abs(self.correlation_coefficient)
        if abs_corr >= 0.8:
            return "very_strong"
        elif abs_corr >= 0.6:
            return "strong"
        elif abs_corr >= 0.4:
            return "moderate"
        elif abs_corr >= 0.2:
            return "weak"
        else:
            return "very_weak"

@dataclass
class CorrelationPattern:
    """Detected correlation pattern between assets"""
    pattern_type: str  # "gold_spike_crypto_dump", "crypto_correlation_break", "safe_haven_flow"
    primary_asset: str
    secondary_asset: str
    pattern_strength: float  # 0.0 to 1.0
    confidence_score: float  # 0.0 to 1.0
    detection_timestamp: datetime
    
    # Pattern-specific data
    trigger_event: Optional[Dict[str, Any]] = None
    lag_hours: Optional[int] = None
    magnitude: Optional[float] = None
    historical_occurrences: int = 0
    success_rate: float = 0.0
    
    # Supporting evidence
    volume_confirmation: bool = False
    volatility_confirmation: bool = False
    market_context: Optional[str] = None

class CorrelationCalculator:
    """Calculate various types of correlations between asset price series"""
    
    def __init__(self):
        self.data_logger = get_data_logger()
        self.performance_logger = get_performance_logger()
    
    def calculate_pearson_correlation(self, series1: pd.Series, series2: pd.Series, 
                                    min_periods: int = 10) -> Tuple[float, float]:
        """Calculate Pearson correlation coefficient and p-value"""
        if len(series1) < min_periods or len(series2) < min_periods:
            return np.nan, 1.0
        
        # Align series and remove NaN values
        aligned_data = pd.DataFrame({'s1': series1, 's2': series2}).dropna()
        
        if len(aligned_data) < min_periods:
            return np.nan, 1.0
        
        try:
            correlation, p_value = pearsonr(aligned_data['s1'], aligned_data['s2'])
            return float(correlation), float(p_value)
        except Exception:
            return np.nan, 1.0
    
    def calculate_rolling_correlation(self, series1: pd.Series, series2: pd.Series, 
                                    window: int = 30) -> pd.Series:
        """Calculate rolling correlation between two series"""
        if len(series1) < window or len(series2) < window:
            return pd.Series(dtype=float)
        
        # Align series
        aligned_data = pd.DataFrame({'s1': series1, 's2': series2}).dropna()
        
        if len(aligned_data) < window:
            return pd.Series(dtype=float)
        
        return aligned_data['s1'].rolling(window=window).corr(aligned_data['s2'])
    
    def calculate_lagged_correlation(self, series1: pd.Series, series2: pd.Series, 
                                   max_lag: int = 24) -> Dict[int, float]:
        """Calculate correlation at different lags"""
        correlations = {}
        
        for lag in range(0, max_lag + 1):
            if lag == 0:
                corr, _ = self.calculate_pearson_correlation(series1, series2)
            else:
                # Lag series2 by 'lag' periods
                lagged_series2 = series2.shift(lag)
                corr, _ = self.calculate_pearson_correlation(series1, lagged_series2)
            
            correlations[lag] = corr
        
        return correlations
    
    def detect_correlation_breakdowns(self, series1: pd.Series, series2: pd.Series, 
                                    window: int = 30, threshold: float = 0.3) -> List[Dict]:
        """Detect periods where correlation breaks down significantly"""
        rolling_corr = self.calculate_rolling_correlation(series1, series2, window)
        
        if rolling_corr.empty:
            return []
        
        # Find periods where correlation drops below threshold
        breakdowns = []
        baseline_corr = rolling_corr.median()
        
        for i, corr in enumerate(rolling_corr):
            if not pd.isna(corr) and abs(corr - baseline_corr) > threshold:
                breakdowns.append({
                    'timestamp': rolling_corr.index[i] if hasattr(rolling_corr.index[i], 'to_pydatetime') else rolling_corr.index[i],
                    'correlation': corr,
                    'baseline': baseline_corr,
                    'deviation': abs(corr - baseline_corr)
                })
        
        return breakdowns

class GoldCryptoPatternDetector:
    """Specialized detector for Gold-crypto correlation patterns"""
    
    def __init__(self):
        self.data_logger = get_data_logger()
        self.performance_logger = get_performance_logger()
        self.correlation_calculator = CorrelationCalculator()
    
    def detect_gold_spike_crypto_dump_pattern(self, gold_data: List[Any], 
                                            crypto_data: List[Any],
                                            spike_threshold: float = 0.03,
                                            dump_threshold: float = -0.03,
                                            max_lag_hours: int = 72) -> List[CorrelationPattern]:
        """Detect patterns where Gold spikes precede crypto dumps"""
        start_time = time.time()
        
        try:
            if len(gold_data) < 10 or len(crypto_data) < 10:
                return []
            
            # Convert to DataFrames with returns
            gold_df = self._prepare_asset_dataframe(gold_data, "GOLD")
            crypto_df = self._prepare_asset_dataframe(crypto_data, crypto_data[0].asset_symbol if hasattr(crypto_data[0], 'asset_symbol') else "CRYPTO")
            
            if gold_df.empty or crypto_df.empty:
                return []
            
            # Identify Gold spikes
            gold_spikes = gold_df[gold_df['returns'] > spike_threshold].copy()
            
            patterns = []
            
            for _, spike in gold_spikes.iterrows():
                spike_time = spike['timestamp']
                
                # Look for crypto dumps in the following hours
                future_crypto = crypto_df[
                    (crypto_df['timestamp'] > spike_time) & 
                    (crypto_df['timestamp'] <= spike_time + timedelta(hours=max_lag_hours))
                ]
                
                if not future_crypto.empty:
                    # Find significant dumps
                    dumps = future_crypto[future_crypto['returns'] < dump_threshold]
                    
                    if not dumps.empty:
                        # Get the most significant dump
                        worst_dump = dumps.loc[dumps['returns'].idxmin()]
                        
                        lag_hours = int((worst_dump['timestamp'] - spike_time).total_seconds() / 3600)
                        
                        # Calculate pattern strength based on magnitude and timing
                        gold_magnitude = spike['returns']
                        crypto_magnitude = abs(worst_dump['returns'])
                        pattern_strength = min(1.0, (gold_magnitude + crypto_magnitude) / 0.1)  # Normalize
                        
                        # Calculate confidence based on historical success rate (simplified)
                        confidence_score = min(1.0, pattern_strength * 0.8)  # Simplified confidence
                        
                        pattern = CorrelationPattern(
                            pattern_type="gold_spike_crypto_dump",
                            primary_asset="GOLD",
                            secondary_asset=crypto_df.iloc[0]['asset'] if 'asset' in crypto_df.columns else "CRYPTO",
                            pattern_strength=pattern_strength,
                            confidence_score=confidence_score,
                            detection_timestamp=datetime.now(),
                            trigger_event={
                                "gold_spike_time": spike_time,
                                "gold_spike_magnitude": gold_magnitude,
                                "crypto_dump_time": worst_dump['timestamp'],
                                "crypto_dump_magnitude": worst_dump['returns']
                            },
                            lag_hours=lag_hours,
                            magnitude=crypto_magnitude
                        )
                        
                        patterns.append(pattern)
            
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time("gold_crypto_pattern_detection", duration_ms, True)
            
            return patterns
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time("gold_crypto_pattern_detection", duration_ms, False)
            self.data_logger.log_data_collection("pattern_detection", "gold_crypto", 0, False, str(e))
            return []
    
    def detect_safe_haven_flow_pattern(self, gold_data: List[Any], 
                                     crypto_data: List[Any],
                                     market_stress_threshold: float = 0.05) -> List[CorrelationPattern]:
        """Detect safe haven flows from crypto to gold during market stress"""
        start_time = time.time()
        
        try:
            if len(gold_data) < 20 or len(crypto_data) < 20:
                return []
            
            gold_df = self._prepare_asset_dataframe(gold_data, "GOLD")
            crypto_df = self._prepare_asset_dataframe(crypto_data, crypto_data[0].asset_symbol if hasattr(crypto_data[0], 'asset_symbol') else "CRYPTO")
            
            if gold_df.empty or crypto_df.empty:
                return []
            
            # Calculate volatility for market stress detection
            crypto_df['volatility'] = crypto_df['returns'].rolling(window=10).std()
            gold_df['volatility'] = gold_df['returns'].rolling(window=10).std()
            
            # Identify high stress periods (high crypto volatility)
            stress_periods = crypto_df[crypto_df['volatility'] > market_stress_threshold]
            
            patterns = []
            
            for _, stress_period in stress_periods.iterrows():
                stress_time = stress_period['timestamp']
                
                # Look for simultaneous gold strength and crypto weakness
                time_window = timedelta(hours=24)
                
                gold_window = gold_df[
                    (gold_df['timestamp'] >= stress_time - time_window) &
                    (gold_df['timestamp'] <= stress_time + time_window)
                ]
                
                crypto_window = crypto_df[
                    (crypto_df['timestamp'] >= stress_time - time_window) &
                    (crypto_df['timestamp'] <= stress_time + time_window)
                ]
                
                if not gold_window.empty and not crypto_window.empty:
                    gold_performance = gold_window['returns'].mean()
                    crypto_performance = crypto_window['returns'].mean()
                    
                    # Safe haven pattern: gold up, crypto down during stress
                    if gold_performance > 0 and crypto_performance < 0:
                        pattern_strength = min(1.0, (gold_performance - crypto_performance) / 0.1)
                        confidence_score = min(1.0, pattern_strength * stress_period['volatility'] / market_stress_threshold)
                        
                        pattern = CorrelationPattern(
                            pattern_type="safe_haven_flow",
                            primary_asset="GOLD",
                            secondary_asset=crypto_df.iloc[0]['asset'] if 'asset' in crypto_df.columns else "CRYPTO",
                            pattern_strength=pattern_strength,
                            confidence_score=confidence_score,
                            detection_timestamp=datetime.now(),
                            trigger_event={
                                "stress_period": stress_time,
                                "crypto_volatility": stress_period['volatility'],
                                "gold_performance": gold_performance,
                                "crypto_performance": crypto_performance
                            },
                            market_context="high_volatility"
                        )
                        
                        patterns.append(pattern)
            
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time("safe_haven_pattern_detection", duration_ms, True)
            
            return patterns
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time("safe_haven_pattern_detection", duration_ms, False)
            self.data_logger.log_data_collection("pattern_detection", "safe_haven", 0, False, str(e))
            return []
    
    def _prepare_asset_dataframe(self, asset_data: List[Any], asset_name: str) -> pd.DataFrame:
        """Convert asset data to DataFrame with returns calculation"""
        try:
            # Handle different data types
            data_points = []
            
            for item in asset_data:
                if hasattr(item, 'current_price'):
                    # OracleData format
                    data_points.append({
                        'timestamp': getattr(item, 'last_updated', datetime.now()),
                        'price': item.current_price,
                        'volume': getattr(item, 'volume_24h', 0),
                        'asset': asset_name
                    })
                elif hasattr(item, 'price'):
                    # CryptoPriceData or TraditionalAssetData format
                    data_points.append({
                        'timestamp': item.timestamp,
                        'price': item.price,
                        'volume': getattr(item, 'volume_24h', 0),
                        'asset': asset_name
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
            self.data_logger.log_data_collection("dataframe_preparation", asset_name, 0, False, str(e))
            return pd.DataFrame()

class CrossAssetCorrelationEngine:
    """Main engine for cross-asset correlation analysis and pattern detection"""
    
    def __init__(self):
        self.correlation_calculator = CorrelationCalculator()
        self.gold_crypto_detector = GoldCryptoPatternDetector()
        self.config = get_config()
        self.data_logger = get_data_logger()
        self.performance_logger = get_performance_logger()
        
        # Configuration
        self.correlation_window = 30
        self.min_data_points = 20
        self.significance_threshold = 0.05
    
    def analyze_asset_correlations(self, asset_data: Dict[str, List[Any]], 
                                 target_asset: str) -> Dict[str, CrossAssetCorrelation]:
        """Analyze correlations between target asset and all other assets"""
        start_time = time.time()
        
        try:
            if target_asset not in asset_data:
                raise ValueError(f"Target asset {target_asset} not found in data")
            
            target_data = asset_data[target_asset]
            if len(target_data) < self.min_data_points:
                raise ValueError(f"Insufficient data for target asset {target_asset}")
            
            # Prepare target asset series
            target_df = self._prepare_price_series(target_data, target_asset)
            
            correlations = {}
            
            for asset_name, data in asset_data.items():
                if asset_name == target_asset or len(data) < self.min_data_points:
                    continue
                
                try:
                    # Prepare comparison asset series
                    asset_df = self._prepare_price_series(data, asset_name)
                    
                    if asset_df.empty or target_df.empty:
                        continue
                    
                    # Merge on timestamp for alignment
                    merged = pd.merge_asof(
                        target_df.sort_values('timestamp'),
                        asset_df.sort_values('timestamp'),
                        on='timestamp',
                        suffixes=('_target', '_asset'),
                        direction='nearest',
                        tolerance=pd.Timedelta(hours=1)  # Allow 1 hour tolerance
                    ).dropna()
                    
                    if len(merged) < self.min_data_points:
                        continue
                    
                    # Calculate correlation
                    corr_coef, p_value = self.correlation_calculator.calculate_pearson_correlation(
                        merged['returns_target'], merged['returns_asset']
                    )
                    
                    if not pd.isna(corr_coef):
                        correlation = CrossAssetCorrelation(
                            asset1=target_asset,
                            asset2=asset_name,
                            correlation_coefficient=corr_coef,
                            p_value=p_value,
                            analysis_period_days=len(merged),
                            timestamp=datetime.now(),
                            confidence_level=1 - p_value,
                            window_size=self.correlation_window
                        )
                        
                        correlations[asset_name] = correlation
                
                except Exception as e:
                    self.data_logger.log_data_collection("correlation_analysis", f"{target_asset}_{asset_name}", 0, False, str(e))
                    continue
            
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time("asset_correlation_analysis", duration_ms, True)
            
            return correlations
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time("asset_correlation_analysis", duration_ms, False)
            self.data_logger.log_data_collection("correlation_analysis", target_asset, 0, False, str(e))
            return {}
    
    def detect_correlation_patterns(self, asset_data: Dict[str, List[Any]]) -> List[CorrelationPattern]:
        """Detect various correlation patterns across all assets"""
        start_time = time.time()
        
        try:
            all_patterns = []
            
            # Gold-crypto patterns
            if "GOLD" in asset_data:
                gold_data = asset_data["GOLD"]
                
                for asset_name, crypto_data in asset_data.items():
                    if asset_name != "GOLD" and asset_name in ["BTC", "ETH", "XRP", "SOL", "BNB", "QIE"]:
                        # Gold spike -> crypto dump patterns
                        spike_dump_patterns = self.gold_crypto_detector.detect_gold_spike_crypto_dump_pattern(
                            gold_data, crypto_data
                        )
                        all_patterns.extend(spike_dump_patterns)
                        
                        # Safe haven flow patterns
                        safe_haven_patterns = self.gold_crypto_detector.detect_safe_haven_flow_pattern(
                            gold_data, crypto_data
                        )
                        all_patterns.extend(safe_haven_patterns)
            
            # Cross-crypto correlation breakdown patterns
            crypto_assets = ["BTC", "ETH", "XRP", "SOL", "BNB", "QIE"]
            available_crypto = [asset for asset in crypto_assets if asset in asset_data]
            
            for i, asset1 in enumerate(available_crypto):
                for asset2 in available_crypto[i+1:]:
                    breakdown_patterns = self._detect_correlation_breakdown_patterns(
                        asset_data[asset1], asset_data[asset2], asset1, asset2
                    )
                    all_patterns.extend(breakdown_patterns)
            
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time("pattern_detection_all", duration_ms, True)
            
            return all_patterns
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time("pattern_detection_all", duration_ms, False)
            self.data_logger.log_data_collection("pattern_detection", "all", 0, False, str(e))
            return []
    
    def _detect_correlation_breakdown_patterns(self, data1: List[Any], data2: List[Any], 
                                             asset1: str, asset2: str) -> List[CorrelationPattern]:
        """Detect correlation breakdown patterns between two crypto assets"""
        try:
            if len(data1) < 30 or len(data2) < 30:
                return []
            
            df1 = self._prepare_price_series(data1, asset1)
            df2 = self._prepare_price_series(data2, asset2)
            
            if df1.empty or df2.empty:
                return []
            
            # Merge data
            merged = pd.merge_asof(
                df1.sort_values('timestamp'),
                df2.sort_values('timestamp'),
                on='timestamp',
                suffixes=('_1', '_2'),
                direction='nearest',
                tolerance=pd.Timedelta(hours=1)
            ).dropna()
            
            if len(merged) < 30:
                return []
            
            # Detect correlation breakdowns
            breakdowns = self.correlation_calculator.detect_correlation_breakdowns(
                merged['returns_1'], merged['returns_2']
            )
            
            patterns = []
            for breakdown in breakdowns:
                pattern = CorrelationPattern(
                    pattern_type="crypto_correlation_break",
                    primary_asset=asset1,
                    secondary_asset=asset2,
                    pattern_strength=breakdown['deviation'],
                    confidence_score=min(1.0, breakdown['deviation'] / 0.5),
                    detection_timestamp=datetime.now(),
                    trigger_event={
                        "breakdown_time": breakdown['timestamp'],
                        "correlation_drop": breakdown['correlation'],
                        "baseline_correlation": breakdown['baseline']
                    }
                )
                patterns.append(pattern)
            
            return patterns
        
        except Exception as e:
            self.data_logger.log_data_collection("correlation_breakdown", f"{asset1}_{asset2}", 0, False, str(e))
            return []
    
    def _prepare_price_series(self, asset_data: List[Any], asset_name: str) -> pd.DataFrame:
        """Prepare price series DataFrame with returns"""
        try:
            data_points = []
            
            for item in asset_data:
                if hasattr(item, 'current_price'):
                    # OracleData format
                    data_points.append({
                        'timestamp': getattr(item, 'last_updated', datetime.now()),
                        'price': item.current_price
                    })
                elif hasattr(item, 'price'):
                    # CryptoPriceData or TraditionalAssetData format
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
    
    def calculate_confidence_scores(self, patterns: List[CorrelationPattern]) -> List[CorrelationPattern]:
        """Calculate and update confidence scores for detected patterns"""
        try:
            for pattern in patterns:
                # Base confidence on pattern strength and historical success
                base_confidence = pattern.pattern_strength
                
                # Adjust based on pattern type
                if pattern.pattern_type == "gold_spike_crypto_dump":
                    # Higher confidence for well-documented patterns
                    base_confidence *= 1.2
                elif pattern.pattern_type == "safe_haven_flow":
                    # Moderate confidence for safe haven flows
                    base_confidence *= 1.0
                elif pattern.pattern_type == "crypto_correlation_break":
                    # Lower confidence for correlation breaks (more noise)
                    base_confidence *= 0.8
                
                # Adjust for lag timing (shorter lags are more reliable)
                if pattern.lag_hours is not None:
                    if pattern.lag_hours <= 24:
                        base_confidence *= 1.1
                    elif pattern.lag_hours <= 48:
                        base_confidence *= 1.0
                    else:
                        base_confidence *= 0.9
                
                # Cap confidence at 1.0
                pattern.confidence_score = min(1.0, base_confidence)
            
            return patterns
        
        except Exception as e:
            self.data_logger.log_data_collection("confidence_calculation", "patterns", 0, False, str(e))
            return patterns

# Main correlation engine class for external use
class CorrelationEngine:
    """Main interface for cross-asset correlation analysis"""
    
    def __init__(self):
        self.engine = CrossAssetCorrelationEngine()
    
    def analyze_correlations(self, asset_data: Dict[str, List[Any]], 
                           target_asset: str) -> Dict[str, CrossAssetCorrelation]:
        """Analyze correlations for a target asset"""
        return self.engine.analyze_asset_correlations(asset_data, target_asset)
    
    def detect_patterns(self, asset_data: Dict[str, List[Any]]) -> List[CorrelationPattern]:
        """Detect correlation patterns across assets"""
        patterns = self.engine.detect_correlation_patterns(asset_data)
        return self.engine.calculate_confidence_scores(patterns)
    
    def get_gold_crypto_patterns(self, asset_data: Dict[str, List[Any]]) -> List[CorrelationPattern]:
        """Get specifically Gold-crypto correlation patterns"""
        if "GOLD" not in asset_data:
            return []
        
        patterns = []
        gold_data = asset_data["GOLD"]
        
        for asset_name, crypto_data in asset_data.items():
            if asset_name != "GOLD" and asset_name in ["BTC", "ETH", "XRP", "SOL", "BNB", "QIE"]:
                gold_patterns = self.engine.gold_crypto_detector.detect_gold_spike_crypto_dump_pattern(
                    gold_data, crypto_data
                )
                patterns.extend(gold_patterns)
        
        return self.engine.calculate_confidence_scores(patterns)