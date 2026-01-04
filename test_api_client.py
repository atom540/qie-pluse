"""
Test client for the AI Risk Oracle API
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime

class RiskOracleClient:
    """Client for interacting with the AI Risk Oracle API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
    
    async def health_check(self):
        """Check API health"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/health") as response:
                return await response.json()
    
    async def get_status(self):
        """Get detailed service status"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/status") as response:
                return await response.json()
    
    async def get_risk_score(self, asset: str):
        """Get risk score for single asset"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/risk/{asset}") as response:
                return await response.json()
    
    async def get_multi_asset_risk(self, assets: list = None):
        """Get risk scores for multiple assets"""
        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}/risk"
            if assets:
                url += f"?assets={','.join(assets)}"
            
            async with session.get(url) as response:
                return await response.json()
    
    async def get_alerts(self, threshold: float = 0.7):
        """Get risk alerts"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/alerts?threshold={threshold}") as response:
                return await response.json()
    
    async def clear_cache(self):
        """Clear service cache"""
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.base_url}/cache/clear") as response:
                return await response.json()

async def test_api():
    """Test the API endpoints"""
    client = RiskOracleClient()
    
    print("🧪 Testing AI Risk Oracle API")
    print("=" * 50)
    
    try:
        # Test health check
        print("\n1. Health Check:")
        health = await client.health_check()
        print(f"   Status: {health['status']}")
        print(f"   Uptime: {health['uptime_seconds']:.1f}s")
        print(f"   Model: {health['model_info']['version']}")
        
        # Test detailed status
        print("\n2. Service Status:")
        status = await client.get_status()
        print(f"   Service: {status['service']}")
        print(f"   Model Version: {status['model_version']}")
        print(f"   Cache Size: {status['cache_size']}")
        
        # Test single asset risk
        print("\n3. Single Asset Risk (BTC):")
        start_time = time.time()
        btc_risk = await client.get_risk_score("BTC")
        latency = (time.time() - start_time) * 1000
        
        print(f"   Asset: {btc_risk['asset_symbol']}")
        print(f"   Risk Score: {btc_risk['combined_risk_score']:.3f}")
        print(f"   Risk Level: {btc_risk['risk_level']}")
        print(f"   ML Score: {btc_risk['ml_risk_score']:.3f}")
        print(f"   Sentiment Score: {btc_risk['sentiment_risk_score']:.3f}")
        print(f"   Confidence: {btc_risk['confidence']:.3f}")
        print(f"   API Latency: {latency:.1f}ms")
        print(f"   Inference Latency: {btc_risk['inference_latency_ms']:.1f}ms")
        
        if btc_risk['reasoning']:
            print(f"   Reasoning:")
            for reason in btc_risk['reasoning']:
                print(f"     • {reason}")
        
        # Test multi-asset risk
        print("\n4. Multi-Asset Risk:")
        start_time = time.time()
        multi_risk = await client.get_multi_asset_risk(["BTC", "ETH", "XRP"])
        latency = (time.time() - start_time) * 1000
        
        print(f"   Assets Assessed: {multi_risk['summary']['total_assets']}")
        print(f"   Average Risk: {multi_risk['summary']['average_risk_score']:.3f}")
        print(f"   Highest Risk: {multi_risk['summary']['highest_risk_score']:.3f}")
        print(f"   Market Regime: {multi_risk['summary']['market_regime']}")
        print(f"   Total Latency: {latency:.1f}ms")
        
        print(f"\n   Individual Scores:")
        for asset, assessment in multi_risk['assessments'].items():
            print(f"     {asset}: {assessment['combined_risk_score']:.3f} ({assessment['risk_level']})")
        
        # Test alerts
        print("\n5. Risk Alerts:")
        alerts = await client.get_alerts(threshold=0.6)
        print(f"   Alert Level: {alerts['alert_level']}")
        print(f"   Triggered Assets: {', '.join(alerts['triggered_assets']) if alerts['triggered_assets'] else 'None'}")
        print(f"   Max Risk Score: {alerts['max_risk_score']:.3f}")
        print(f"   Recommendation: {alerts['recommended_action']}")
        
        # Performance summary
        print("\n6. Performance Summary:")
        if btc_risk['inference_latency_ms'] < 500:
            print(f"   ✅ Inference Speed: {btc_risk['inference_latency_ms']:.1f}ms (target: <500ms)")
        else:
            print(f"   ⚠️  Inference Speed: {btc_risk['inference_latency_ms']:.1f}ms (target: <500ms)")
        
        if btc_risk['data_freshness_seconds'] < 60:
            print(f"   ✅ Data Freshness: {btc_risk['data_freshness_seconds']}s (excellent)")
        else:
            print(f"   ⚠️  Data Freshness: {btc_risk['data_freshness_seconds']}s (may be stale)")
        
        print(f"\n✅ API Test Complete - All endpoints working!")
        
    except aiohttp.ClientError as e:
        print(f"❌ Connection Error: {e}")
        print("   Make sure the API server is running: python start_api_server.py")
    except Exception as e:
        print(f"❌ Test Error: {e}")

async def stress_test():
    """Simple stress test"""
    client = RiskOracleClient()
    
    print("\n🔥 Running Stress Test (10 concurrent requests)")
    print("=" * 50)
    
    start_time = time.time()
    
    # Run 10 concurrent requests
    tasks = [client.get_risk_score("BTC") for _ in range(10)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    end_time = time.time()
    
    # Analyze results
    successful = sum(1 for r in results if not isinstance(r, Exception))
    failed = len(results) - successful
    total_time = (end_time - start_time) * 1000
    avg_time = total_time / len(results)
    
    print(f"   Total Requests: {len(results)}")
    print(f"   Successful: {successful}")
    print(f"   Failed: {failed}")
    print(f"   Total Time: {total_time:.1f}ms")
    print(f"   Average Time: {avg_time:.1f}ms")
    print(f"   Requests/Second: {len(results) / (total_time / 1000):.1f}")
    
    if failed == 0:
        print("   ✅ All requests successful!")
    else:
        print(f"   ⚠️  {failed} requests failed")

if __name__ == "__main__":
    print("🚀 AI Risk Oracle API Test Client")
    
    # Run basic tests
    asyncio.run(test_api())
    
    # Ask if user wants stress test
    print("\n" + "="*50)
    response = input("Run stress test? (y/n): ").lower().strip()
    if response in ['y', 'yes']:
        asyncio.run(stress_test())
    
    print("\nThank you for testing the AI Risk Oracle API! 🎯")