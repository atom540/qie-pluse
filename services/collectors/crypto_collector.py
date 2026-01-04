"""
Crypto Price Data Collector
Implements CoinGecko and Binance API clients for historical and real-time crypto data
"""

import asyncio
import aiohttp
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time
import hmac
import hashlib
from urllib.parse import urlencode

from config import get_config
from logging_config import get_data_logger, get_performance_logger

@dataclass
class CryptoPriceData:
    """Data structure for crypto price information"""
    symbol: str
    price: float
    timestamp: datetime
    volume_24h: float
    price_change_24h: float
    market_cap: Optional[float] = None
    source: str = "unknown"

@dataclass
class TechnicalIndicators:
    """Technical indicators for crypto assets"""
    symbol: str
    rsi: Optional[float] = None
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    volatility: Optional[float] = None
    timestamp: datetime = None

class CoinGeckoCollector:
    """CoinGecko API client for historical crypto price data"""
    
    def __init__(self):
        self.config = get_config()
        self.data_logger = get_data_logger()
        self.performance_logger = get_performance_logger()
        self.base_url = "https://api.coingecko.com/api/v3"
        self.api_key = self.config.api.coingecko_api_key
        
        # CoinGecko asset mapping
        self.asset_mapping = {
            "BTC": "bitcoin",
            "ETH": "ethereum", 
            "XRP": "ripple",
            "SOL": "solana",
            "BNB": "binancecoin"
        }
    
    async def get_current_prices(self, assets: List[str]) -> List[CryptoPriceData]:
        """Get current prices for specified crypto assets"""
        start_time = time.time()
        
        try:
            # Map symbols to CoinGecko IDs
            coin_ids = [self.asset_mapping.get(asset) for asset in assets if asset in self.asset_mapping]
            if not coin_ids:
                raise ValueError(f"No valid assets found in mapping: {assets}")
            
            url = f"{self.base_url}/simple/price"
            params = {
                "ids": ",".join(coin_ids),
                "vs_currencies": "usd",
                "include_24hr_vol": "true",
                "include_24hr_change": "true",
                "include_market_cap": "true"
            }
            
            if self.api_key:
                params["x_cg_demo_api_key"] = self.api_key
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        results = []
                        for asset in assets:
                            coin_id = self.asset_mapping.get(asset)
                            if coin_id and coin_id in data:
                                coin_data = data[coin_id]
                                
                                price_data = CryptoPriceData(
                                    symbol=asset,
                                    price=coin_data["usd"],
                                    timestamp=datetime.now(),
                                    volume_24h=coin_data.get("usd_24h_vol", 0),
                                    price_change_24h=coin_data.get("usd_24h_change", 0),
                                    market_cap=coin_data.get("usd_market_cap"),
                                    source="coingecko"
                                )
                                results.append(price_data)
                        
                        duration_ms = (time.time() - start_time) * 1000
                        self.performance_logger.log_inference_time("coingecko_current_prices", duration_ms, True)
                        self.data_logger.log_data_collection("coingecko", ",".join(assets), len(results), True)
                        
                        return results
                    
                    else:
                        error_text = await response.text()
                        raise Exception(f"CoinGecko API error {response.status}: {error_text}")
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time("coingecko_current_prices", duration_ms, False)
            self.data_logger.log_data_collection("coingecko", ",".join(assets), 0, False, str(e))
            raise
    
    async def get_historical_prices(self, asset: str, days: int = 30) -> List[CryptoPriceData]:
        """Get historical daily prices for a crypto asset"""
        start_time = time.time()
        
        try:
            coin_id = self.asset_mapping.get(asset)
            if not coin_id:
                raise ValueError(f"Asset {asset} not supported")
            
            url = f"{self.base_url}/coins/{coin_id}/market_chart"
            params = {
                "vs_currency": "usd",
                "days": str(days),
                "interval": "daily"
            }
            
            if self.api_key:
                params["x_cg_demo_api_key"] = self.api_key
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        results = []
                        prices = data.get("prices", [])
                        volumes = data.get("total_volumes", [])
                        
                        for i, (timestamp_ms, price) in enumerate(prices):
                            volume = volumes[i][1] if i < len(volumes) else 0
                            
                            # Calculate 24h change (approximate)
                            price_change_24h = 0
                            if i > 0:
                                prev_price = prices[i-1][1]
                                price_change_24h = ((price - prev_price) / prev_price) * 100
                            
                            price_data = CryptoPriceData(
                                symbol=asset,
                                price=price,
                                timestamp=datetime.fromtimestamp(timestamp_ms / 1000),
                                volume_24h=volume,
                                price_change_24h=price_change_24h,
                                source="coingecko"
                            )
                            results.append(price_data)
                        
                        duration_ms = (time.time() - start_time) * 1000
                        self.performance_logger.log_inference_time("coingecko_historical_prices", duration_ms, True)
                        self.data_logger.log_data_collection("coingecko", asset, len(results), True)
                        
                        return results
                    
                    else:
                        error_text = await response.text()
                        raise Exception(f"CoinGecko API error {response.status}: {error_text}")
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time("coingecko_historical_prices", duration_ms, False)
            self.data_logger.log_data_collection("coingecko", asset, 0, False, str(e))
            raise

class BinanceCollector:
    """Binance API client for high-frequency crypto data and technical indicators"""
    
    def __init__(self):
        self.config = get_config()
        self.data_logger = get_data_logger()
        self.performance_logger = get_performance_logger()
        self.base_url = "https://api.binance.com/api/v3"
        self.api_key = self.config.api.binance_api_key
        self.secret_key = self.config.api.binance_secret_key
        
        # Binance symbol mapping
        self.symbol_mapping = {
            "BTC": "BTCUSDT",
            "ETH": "ETHUSDT",
            "XRP": "XRPUSDT", 
            "SOL": "SOLUSDT",
            "BNB": "BNBUSDT"
        }
    
    def _generate_signature(self, query_string: str) -> str:
        """Generate HMAC SHA256 signature for authenticated requests"""
        return hmac.new(
            self.secret_key.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    async def get_current_prices(self, assets: List[str]) -> List[CryptoPriceData]:
        """Get current prices from Binance"""
        start_time = time.time()
        
        try:
            symbols = [self.symbol_mapping.get(asset) for asset in assets if asset in self.symbol_mapping]
            if not symbols:
                raise ValueError(f"No valid assets found in mapping: {assets}")
            
            url = f"{self.base_url}/ticker/24hr"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Filter for our symbols
                        symbol_data = {item["symbol"]: item for item in data if item["symbol"] in symbols}
                        
                        results = []
                        for asset in assets:
                            binance_symbol = self.symbol_mapping.get(asset)
                            if binance_symbol and binance_symbol in symbol_data:
                                ticker = symbol_data[binance_symbol]
                                
                                price_data = CryptoPriceData(
                                    symbol=asset,
                                    price=float(ticker["lastPrice"]),
                                    timestamp=datetime.now(),
                                    volume_24h=float(ticker["volume"]),
                                    price_change_24h=float(ticker["priceChangePercent"]),
                                    source="binance"
                                )
                                results.append(price_data)
                        
                        duration_ms = (time.time() - start_time) * 1000
                        self.performance_logger.log_inference_time("binance_current_prices", duration_ms, True)
                        self.data_logger.log_data_collection("binance", ",".join(assets), len(results), True)
                        
                        return results
                    
                    else:
                        error_text = await response.text()
                        raise Exception(f"Binance API error {response.status}: {error_text}")
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time("binance_current_prices", duration_ms, False)
            self.data_logger.log_data_collection("binance", ",".join(assets), 0, False, str(e))
            raise
    
    async def get_kline_data(self, asset: str, interval: str = "1m", limit: int = 100) -> List[CryptoPriceData]:
        """Get minute-level kline/candlestick data"""
        start_time = time.time()
        
        try:
            symbol = self.symbol_mapping.get(asset)
            if not symbol:
                raise ValueError(f"Asset {asset} not supported")
            
            url = f"{self.base_url}/klines"
            params = {
                "symbol": symbol,
                "interval": interval,
                "limit": limit
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        results = []
                        for kline in data:
                            # Kline format: [timestamp, open, high, low, close, volume, ...]
                            timestamp_ms = int(kline[0])
                            close_price = float(kline[4])
                            volume = float(kline[5])
                            
                            price_data = CryptoPriceData(
                                symbol=asset,
                                price=close_price,
                                timestamp=datetime.fromtimestamp(timestamp_ms / 1000),
                                volume_24h=volume,  # This is interval volume, not 24h
                                price_change_24h=0,  # Would need calculation
                                source="binance"
                            )
                            results.append(price_data)
                        
                        duration_ms = (time.time() - start_time) * 1000
                        self.performance_logger.log_inference_time("binance_kline_data", duration_ms, True)
                        self.data_logger.log_data_collection("binance", asset, len(results), True)
                        
                        return results
                    
                    else:
                        error_text = await response.text()
                        raise Exception(f"Binance API error {response.status}: {error_text}")
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time("binance_kline_data", duration_ms, False)
            self.data_logger.log_data_collection("binance", asset, 0, False, str(e))
            raise
    
    def calculate_technical_indicators(self, price_data: List[CryptoPriceData]) -> TechnicalIndicators:
        """Calculate technical indicators from price data"""
        if not price_data:
            return TechnicalIndicators(symbol="unknown")
        
        # Convert to pandas for easier calculation
        df = pd.DataFrame([
            {
                "price": data.price,
                "timestamp": data.timestamp,
                "volume": data.volume_24h
            }
            for data in price_data
        ])
        
        if len(df) < 2:
            return TechnicalIndicators(symbol=price_data[0].symbol, timestamp=datetime.now())
        
        df = df.sort_values("timestamp")
        prices = df["price"]
        
        indicators = TechnicalIndicators(
            symbol=price_data[0].symbol,
            timestamp=datetime.now()
        )
        
        # Simple Moving Averages
        if len(prices) >= 20:
            indicators.sma_20 = prices.tail(20).mean()
        if len(prices) >= 50:
            indicators.sma_50 = prices.tail(50).mean()
        
        # RSI calculation (simplified)
        if len(prices) >= 14:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            indicators.rsi = 100 - (100 / (1 + rs.iloc[-1]))
        
        # Volatility (standard deviation of returns)
        if len(prices) >= 10:
            returns = prices.pct_change().dropna()
            indicators.volatility = returns.std() * 100  # As percentage
        
        return indicators

class CryptoDataCollector:
    """Main crypto data collector combining CoinGecko and Binance"""
    
    def __init__(self):
        self.coingecko = CoinGeckoCollector()
        self.binance = BinanceCollector()
        self.config = get_config()
        self.data_logger = get_data_logger()
        
        # Assets to collect (from requirements)
        self.target_assets = ["BTC", "ETH", "XRP", "SOL", "BNB"]
    
    async def collect_current_prices(self, source: str = "coingecko") -> List[CryptoPriceData]:
        """Collect current prices from specified source"""
        if source == "coingecko":
            return await self.coingecko.get_current_prices(self.target_assets)
        elif source == "binance":
            return await self.binance.get_current_prices(self.target_assets)
        else:
            raise ValueError(f"Unknown source: {source}")
    
    async def collect_historical_data(self, asset: str, days: int = 30) -> List[CryptoPriceData]:
        """Collect historical data for an asset"""
        return await self.coingecko.get_historical_prices(asset, days)
    
    async def collect_historical_prices(self, source: str, asset: str, 
                                      start_date: datetime, end_date: datetime) -> List[CryptoPriceData]:
        """Collect historical prices for a specific date range"""
        try:
            # Calculate days between dates
            days = (end_date - start_date).days
            
            if source == "coingecko":
                return await self.coingecko.get_historical_prices(asset, days)
            elif source == "binance":
                # For now, use the same method - in production this would use Binance historical API
                return await self.coingecko.get_historical_prices(asset, days)
            else:
                raise ValueError(f"Unknown source: {source}")
        
        except Exception as e:
            self.data_logger.log_data_collection("crypto_historical", f"{source}_{asset}", 0, False, str(e))
            return []
    
    async def collect_minute_data(self, asset: str, limit: int = 100) -> Tuple[List[CryptoPriceData], TechnicalIndicators]:
        """Collect minute-level data and calculate technical indicators"""
        price_data = await self.binance.get_kline_data(asset, "1m", limit)
        indicators = self.binance.calculate_technical_indicators(price_data)
        return price_data, indicators
    
    async def collect_all_current_data(self) -> Dict[str, List[CryptoPriceData]]:
        """Collect current data from all sources"""
        results = {}
        
        try:
            # CoinGecko data
            coingecko_data = await self.coingecko.get_current_prices(self.target_assets)
            results["coingecko"] = coingecko_data
        except Exception as e:
            self.data_logger.log_data_collection("coingecko", "all", 0, False, str(e))
            results["coingecko"] = []
        
        try:
            # Binance data
            binance_data = await self.binance.get_current_prices(self.target_assets)
            results["binance"] = binance_data
        except Exception as e:
            self.data_logger.log_data_collection("binance", "all", 0, False, str(e))
            results["binance"] = []
        
        return results
    
    def validate_data(self, data: List[CryptoPriceData]) -> List[CryptoPriceData]:
        """Validate and clean price data"""
        valid_data = []
        
        for item in data:
            # Basic validation
            if item.price <= 0:
                continue
            if item.volume_24h < 0:
                continue
            if abs(item.price_change_24h) > 100:  # More than 100% change seems suspicious
                continue
            
            valid_data.append(item)
        
        return valid_data