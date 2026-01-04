"""
Traditional Asset Data Collector
Implements Alpha Vantage API client for Gold historical prices and Yahoo Finance integration
"""

import asyncio
import aiohttp
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time

from config import get_config
from logging_config import get_data_logger, get_performance_logger

@dataclass
class TraditionalAssetData:
    """Data structure for traditional asset price information"""
    symbol: str
    price: float
    timestamp: datetime
    volume_24h: float
    price_change_24h: float
    market_cap: Optional[float] = None
    source: str = "unknown"

@dataclass
class CorrelationAnalysis:
    """Cross-asset correlation analysis results"""
    asset1: str
    asset2: str
    correlation_coefficient: float
    p_value: float
    analysis_period_days: int
    timestamp: datetime
    confidence_level: float

class AlphaVantageCollector:
    """Alpha Vantage API client for Gold historical prices"""
    
    def __init__(self):
        self.config = get_config()
        self.data_logger = get_data_logger()
        self.performance_logger = get_performance_logger()
        self.base_url = "https://www.alphavantage.co/query"
        self.api_key = self.config.api.alpha_vantage_api_key
        
        if not self.api_key:
            raise ValueError("Alpha Vantage API key not configured")
    
    async def get_gold_current_price(self) -> TraditionalAssetData:
        """Get current Gold price from Alpha Vantage"""
        start_time = time.time()
        
        try:
            params = {
                "function": "GLOBAL_QUOTE",
                "symbol": "GLD",  # SPDR Gold Trust ETF as proxy for Gold
                "apikey": self.api_key
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if "Global Quote" in data:
                            quote = data["Global Quote"]
                            
                            price_data = TraditionalAssetData(
                                symbol="GOLD",
                                price=float(quote["05. price"]),
                                timestamp=datetime.now(),
                                volume_24h=float(quote["06. volume"]),
                                price_change_24h=float(quote["09. change"]),
                                source="alphavantage"
                            )
                            
                            duration_ms = (time.time() - start_time) * 1000
                            self.performance_logger.log_inference_time("alphavantage_current_gold", duration_ms, True)
                            self.data_logger.log_data_collection("alphavantage", "GOLD", 1, True)
                            
                            return price_data
                        
                        elif "Error Message" in data:
                            raise Exception(f"Alpha Vantage API error: {data['Error Message']}")
                        elif "Note" in data:
                            raise Exception(f"Alpha Vantage API rate limit: {data['Note']}")
                        else:
                            raise Exception(f"Unexpected Alpha Vantage response format: {data}")
                    
                    else:
                        error_text = await response.text()
                        raise Exception(f"Alpha Vantage API error {response.status}: {error_text}")
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time("alphavantage_current_gold", duration_ms, False)
            self.data_logger.log_data_collection("alphavantage", "GOLD", 0, False, str(e))
            raise
    
    async def get_gold_historical_prices(self, days: int = 30) -> List[TraditionalAssetData]:
        """Get historical Gold prices from Alpha Vantage"""
        start_time = time.time()
        
        try:
            params = {
                "function": "TIME_SERIES_DAILY",
                "symbol": "GLD",  # SPDR Gold Trust ETF as proxy for Gold
                "apikey": self.api_key,
                "outputsize": "compact" if days <= 100 else "full"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if "Time Series (Daily)" in data:
                            time_series = data["Time Series (Daily)"]
                            
                            results = []
                            sorted_dates = sorted(time_series.keys(), reverse=True)[:days]
                            
                            for i, date_str in enumerate(sorted_dates):
                                daily_data = time_series[date_str]
                                
                                # Calculate 24h change
                                price_change_24h = 0
                                if i < len(sorted_dates) - 1:
                                    prev_date = sorted_dates[i + 1]
                                    prev_close = float(time_series[prev_date]["4. close"])
                                    current_close = float(daily_data["4. close"])
                                    price_change_24h = ((current_close - prev_close) / prev_close) * 100
                                
                                price_data = TraditionalAssetData(
                                    symbol="GOLD",
                                    price=float(daily_data["4. close"]),
                                    timestamp=datetime.strptime(date_str, "%Y-%m-%d"),
                                    volume_24h=float(daily_data["5. volume"]),
                                    price_change_24h=price_change_24h,
                                    source="alphavantage"
                                )
                                results.append(price_data)
                            
                            duration_ms = (time.time() - start_time) * 1000
                            self.performance_logger.log_inference_time("alphavantage_historical_gold", duration_ms, True)
                            self.data_logger.log_data_collection("alphavantage", "GOLD", len(results), True)
                            
                            return results
                        
                        elif "Error Message" in data:
                            raise Exception(f"Alpha Vantage API error: {data['Error Message']}")
                        elif "Note" in data:
                            raise Exception(f"Alpha Vantage API rate limit: {data['Note']}")
                        else:
                            raise Exception(f"Unexpected Alpha Vantage response format: {data}")
                    
                    else:
                        error_text = await response.text()
                        raise Exception(f"Alpha Vantage API error {response.status}: {error_text}")
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time("alphavantage_historical_gold", duration_ms, False)
            self.data_logger.log_data_collection("alphavantage", "GOLD", 0, False, str(e))
            raise

class YahooFinanceCollector:
    """Yahoo Finance integration for additional commodity data"""
    
    def __init__(self):
        self.config = get_config()
        self.data_logger = get_data_logger()
        self.performance_logger = get_performance_logger()
        
        # Yahoo Finance symbol mapping for commodities
        self.symbol_mapping = {
            "GOLD": "GC=F",  # Gold Futures
            "SILVER": "SI=F",  # Silver Futures
            "OIL": "CL=F",   # Crude Oil Futures
            "COPPER": "HG=F"  # Copper Futures
        }
    
    def get_current_prices(self, assets: List[str]) -> List[TraditionalAssetData]:
        """Get current prices for traditional assets from Yahoo Finance"""
        start_time = time.time()
        
        try:
            results = []
            
            for asset in assets:
                yahoo_symbol = self.symbol_mapping.get(asset)
                if not yahoo_symbol:
                    continue
                
                ticker = yf.Ticker(yahoo_symbol)
                info = ticker.info
                hist = ticker.history(period="2d")
                
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    volume = hist['Volume'].iloc[-1]
                    
                    # Calculate 24h change
                    price_change_24h = 0
                    if len(hist) >= 2:
                        prev_price = hist['Close'].iloc[-2]
                        price_change_24h = ((current_price - prev_price) / prev_price) * 100
                    
                    price_data = TraditionalAssetData(
                        symbol=asset,
                        price=float(current_price),
                        timestamp=datetime.now(),
                        volume_24h=float(volume),
                        price_change_24h=float(price_change_24h),
                        market_cap=info.get('marketCap'),
                        source="yahoo_finance"
                    )
                    results.append(price_data)
            
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time("yahoo_finance_current_prices", duration_ms, True)
            self.data_logger.log_data_collection("yahoo_finance", ",".join(assets), len(results), True)
            
            return results
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time("yahoo_finance_current_prices", duration_ms, False)
            self.data_logger.log_data_collection("yahoo_finance", ",".join(assets), 0, False, str(e))
            raise
    
    def get_historical_prices(self, asset: str, days: int = 30) -> List[TraditionalAssetData]:
        """Get historical prices for a traditional asset"""
        start_time = time.time()
        
        try:
            yahoo_symbol = self.symbol_mapping.get(asset)
            if not yahoo_symbol:
                raise ValueError(f"Asset {asset} not supported")
            
            ticker = yf.Ticker(yahoo_symbol)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            hist = ticker.history(start=start_date, end=end_date)
            
            results = []
            for i, (date, row) in enumerate(hist.iterrows()):
                # Calculate 24h change
                price_change_24h = 0
                if i > 0:
                    prev_close = hist['Close'].iloc[i-1]
                    current_close = row['Close']
                    price_change_24h = ((current_close - prev_close) / prev_close) * 100
                
                price_data = TraditionalAssetData(
                    symbol=asset,
                    price=float(row['Close']),
                    timestamp=date.to_pydatetime(),
                    volume_24h=float(row['Volume']),
                    price_change_24h=float(price_change_24h),
                    source="yahoo_finance"
                )
                results.append(price_data)
            
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time("yahoo_finance_historical_prices", duration_ms, True)
            self.data_logger.log_data_collection("yahoo_finance", asset, len(results), True)
            
            return results
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time("yahoo_finance_historical_prices", duration_ms, False)
            self.data_logger.log_data_collection("yahoo_finance", asset, 0, False, str(e))
            raise

class CorrelationAnalyzer:
    """Cross-asset correlation analysis between Gold and crypto assets"""
    
    def __init__(self):
        self.config = get_config()
        self.data_logger = get_data_logger()
        self.performance_logger = get_performance_logger()
    
    def calculate_correlation(self, asset1_data: List[TraditionalAssetData], 
                            asset2_data: List[TraditionalAssetData],
                            min_periods: int = 10) -> CorrelationAnalysis:
        """Calculate correlation between two asset price series"""
        start_time = time.time()
        
        try:
            if len(asset1_data) < min_periods or len(asset2_data) < min_periods:
                raise ValueError(f"Insufficient data for correlation analysis. Need at least {min_periods} periods")
            
            # Convert to pandas DataFrames for easier analysis
            df1 = pd.DataFrame([
                {"timestamp": data.timestamp, "price": data.price, "returns": 0}
                for data in asset1_data
            ]).sort_values("timestamp")
            
            df2 = pd.DataFrame([
                {"timestamp": data.timestamp, "price": data.price, "returns": 0}
                for data in asset2_data
            ]).sort_values("timestamp")
            
            # Calculate returns
            df1["returns"] = df1["price"].pct_change()
            df2["returns"] = df2["price"].pct_change()
            
            # Align timestamps and merge
            df1["date"] = df1["timestamp"].dt.date
            df2["date"] = df2["timestamp"].dt.date
            
            merged = pd.merge(df1[["date", "returns"]], df2[["date", "returns"]], 
                            on="date", suffixes=("_1", "_2"))
            
            # Remove NaN values
            merged = merged.dropna()
            
            if len(merged) < min_periods:
                raise ValueError(f"Insufficient overlapping data for correlation analysis")
            
            # Calculate Pearson correlation
            correlation_matrix = merged[["returns_1", "returns_2"]].corr()
            correlation_coefficient = correlation_matrix.iloc[0, 1]
            
            # Calculate p-value using scipy if available, otherwise approximate
            try:
                from scipy.stats import pearsonr
                _, p_value = pearsonr(merged["returns_1"], merged["returns_2"])
            except ImportError:
                # Approximate p-value calculation
                n = len(merged)
                t_stat = correlation_coefficient * np.sqrt((n - 2) / (1 - correlation_coefficient**2))
                # Rough approximation for p-value
                p_value = 2 * (1 - abs(t_stat) / (abs(t_stat) + np.sqrt(n - 2)))
            
            # Calculate confidence level
            confidence_level = max(0.0, min(1.0, 1 - p_value))
            
            analysis = CorrelationAnalysis(
                asset1=asset1_data[0].symbol,
                asset2=asset2_data[0].symbol,
                correlation_coefficient=float(correlation_coefficient),
                p_value=float(p_value),
                analysis_period_days=len(merged),
                timestamp=datetime.now(),
                confidence_level=float(confidence_level)
            )
            
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time("correlation_analysis", duration_ms, True)
            
            return analysis
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time("correlation_analysis", duration_ms, False)
            raise
    
    def detect_gold_crypto_patterns(self, gold_data: List[TraditionalAssetData],
                                   crypto_data: List[TraditionalAssetData],
                                   spike_threshold: float = 0.05) -> Dict[str, any]:
        """Detect patterns where Gold spikes precede crypto dumps"""
        start_time = time.time()
        
        try:
            # Convert to DataFrames
            gold_df = pd.DataFrame([
                {"timestamp": data.timestamp, "price": data.price, "returns": 0}
                for data in gold_data
            ]).sort_values("timestamp")
            
            crypto_df = pd.DataFrame([
                {"timestamp": data.timestamp, "price": data.price, "returns": 0}
                for data in crypto_data
            ]).sort_values("timestamp")
            
            # Calculate returns
            gold_df["returns"] = gold_df["price"].pct_change()
            crypto_df["returns"] = crypto_df["price"].pct_change()
            
            # Identify Gold spikes (returns > spike_threshold)
            gold_spikes = gold_df[gold_df["returns"] > spike_threshold].copy()
            
            patterns_detected = []
            
            for _, spike in gold_spikes.iterrows():
                spike_date = spike["timestamp"]
                
                # Look for crypto dumps in the next 1-3 days
                future_crypto = crypto_df[
                    (crypto_df["timestamp"] > spike_date) & 
                    (crypto_df["timestamp"] <= spike_date + timedelta(days=3))
                ]
                
                if not future_crypto.empty:
                    # Check if there's a significant negative return
                    min_return = future_crypto["returns"].min()
                    if min_return < -spike_threshold:  # Crypto dump detected
                        patterns_detected.append({
                            "gold_spike_date": spike_date,
                            "gold_spike_return": spike["returns"],
                            "crypto_dump_date": future_crypto.loc[future_crypto["returns"].idxmin(), "timestamp"],
                            "crypto_dump_return": min_return,
                            "days_lag": (future_crypto.loc[future_crypto["returns"].idxmin(), "timestamp"] - spike_date).days
                        })
            
            pattern_analysis = {
                "total_gold_spikes": len(gold_spikes),
                "patterns_detected": len(patterns_detected),
                "pattern_success_rate": len(patterns_detected) / len(gold_spikes) if len(gold_spikes) > 0 else 0,
                "average_lag_days": np.mean([p["days_lag"] for p in patterns_detected]) if patterns_detected else 0,
                "patterns": patterns_detected,
                "analysis_timestamp": datetime.now()
            }
            
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time("gold_crypto_pattern_detection", duration_ms, True)
            
            return pattern_analysis
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time("gold_crypto_pattern_detection", duration_ms, False)
            raise

class TraditionalAssetCollector:
    """Main traditional asset data collector combining Alpha Vantage and Yahoo Finance"""
    
    def __init__(self):
        self.alphavantage = AlphaVantageCollector()
        self.yahoo_finance = YahooFinanceCollector()
        self.correlation_analyzer = CorrelationAnalyzer()
        self.config = get_config()
        self.data_logger = get_data_logger()
        
        # Traditional assets to collect
        self.target_assets = ["GOLD", "SILVER", "OIL"]
    
    async def collect_current_prices(self, source: str = "alphavantage") -> List[TraditionalAssetData]:
        """Collect current prices from specified source"""
        if source == "alphavantage":
            # Alpha Vantage only supports Gold through GLD ETF
            return [await self.alphavantage.get_gold_current_price()]
        elif source == "yahoo_finance":
            return self.yahoo_finance.get_current_prices(self.target_assets)
        else:
            raise ValueError(f"Unknown source: {source}")
    
    async def collect_gold_historical_data(self, days: int = 30, source: str = "alphavantage") -> List[TraditionalAssetData]:
        """Collect historical Gold data"""
        if source == "alphavantage":
            return await self.alphavantage.get_gold_historical_prices(days)
        elif source == "yahoo_finance":
            return self.yahoo_finance.get_historical_prices("GOLD", days)
        else:
            raise ValueError(f"Unknown source: {source}")
    
    def collect_commodity_historical_data(self, asset: str, days: int = 30) -> List[TraditionalAssetData]:
        """Collect historical data for commodities via Yahoo Finance"""
        return self.yahoo_finance.get_historical_prices(asset, days)
    
    async def collect_historical_prices(self, source: str, asset: str,
                                      start_date: datetime, end_date: datetime) -> List[TraditionalAssetData]:
        """Collect historical prices for a specific date range"""
        try:
            # Calculate days between dates
            days = (end_date - start_date).days
            
            if source == "alphavantage":
                if asset == "GOLD":
                    return await self.alphavantage.get_gold_historical_prices(days)
                else:
                    raise ValueError(f"Alpha Vantage does not support asset: {asset}")
            elif source == "yahoo_finance":
                return self.yahoo_finance.get_historical_prices(asset, days)
            else:
                raise ValueError(f"Unknown source: {source}")
        
        except Exception as e:
            self.data_logger.log_data_collection("traditional_historical", f"{source}_{asset}", 0, False, str(e))
            return []
    
    async def analyze_gold_crypto_correlation(self, crypto_data: List, days: int = 30) -> CorrelationAnalysis:
        """Analyze correlation between Gold and crypto assets"""
        # Get Gold historical data
        gold_data = await self.collect_gold_historical_data(days)
        
        # Convert crypto data to TraditionalAssetData format for consistency
        crypto_traditional_format = [
            TraditionalAssetData(
                symbol=data.symbol,
                price=data.price,
                timestamp=data.timestamp,
                volume_24h=data.volume_24h,
                price_change_24h=data.price_change_24h,
                source=data.source
            )
            for data in crypto_data
        ]
        
        return self.correlation_analyzer.calculate_correlation(gold_data, crypto_traditional_format)
    
    async def detect_gold_crypto_patterns(self, crypto_data: List, days: int = 90) -> Dict[str, any]:
        """Detect Gold spike -> crypto dump patterns"""
        # Get Gold historical data for pattern analysis
        gold_data = await self.collect_gold_historical_data(days)
        
        # Convert crypto data format
        crypto_traditional_format = [
            TraditionalAssetData(
                symbol=data.symbol,
                price=data.price,
                timestamp=data.timestamp,
                volume_24h=data.volume_24h,
                price_change_24h=data.price_change_24h,
                source=data.source
            )
            for data in crypto_data
        ]
        
        return self.correlation_analyzer.detect_gold_crypto_patterns(gold_data, crypto_traditional_format)
    
    async def collect_all_traditional_data(self) -> Dict[str, List[TraditionalAssetData]]:
        """Collect current data from all traditional asset sources"""
        results = {}
        
        try:
            # Alpha Vantage Gold data
            alphavantage_data = await self.collect_current_prices("alphavantage")
            results["alphavantage"] = alphavantage_data
        except Exception as e:
            self.data_logger.log_data_collection("alphavantage", "GOLD", 0, False, str(e))
            results["alphavantage"] = []
        
        try:
            # Yahoo Finance commodity data
            yahoo_data = self.yahoo_finance.get_current_prices(self.target_assets)
            results["yahoo_finance"] = yahoo_data
        except Exception as e:
            self.data_logger.log_data_collection("yahoo_finance", "commodities", 0, False, str(e))
            results["yahoo_finance"] = []
        
        return results
    
    def validate_data(self, data: List[TraditionalAssetData]) -> List[TraditionalAssetData]:
        """Validate and clean traditional asset data"""
        valid_data = []
        
        for item in data:
            # Basic validation
            if item.price <= 0:
                continue
            if item.volume_24h < 0:
                continue
            if abs(item.price_change_24h) > 50:  # More than 50% change seems suspicious for traditional assets
                continue
            
            valid_data.append(item)
        
        return valid_data