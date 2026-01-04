"""
Command-line interface for the AI Risk Oracle Inference Service
"""

import asyncio
import argparse
import json
from datetime import datetime
from services.inference.risk_inference_service import RiskInferenceService

async def get_risk_score(asset: str, model_path: str = None):
    """Get risk score for a single asset"""
    service = RiskInferenceService(model_path)
    
    if not service.current_model:
        if not service.load_latest_model():
            print(f"❌ Error: Could not load model")
            return
    
    print(f"🔍 Getting risk assessment for {asset}...")
    assessment = await service.get_risk_assessment(asset)
    
    if assessment:
        print(f"\n📊 Risk Assessment for {asset}")
        print(f"{'='*40}")
        print(f"Timestamp: {assessment.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Combined Risk Score: {assessment.combined_risk_score:.3f}")
        print(f"Risk Level: {assessment.risk_level}")
        print(f"ML Risk Score: {assessment.ml_risk_score:.3f}")
        print(f"Sentiment Risk Score: {assessment.sentiment_risk_score:.3f}")
        print(f"Confidence: {assessment.confidence:.3f}")
        print(f"Market Regime: {assessment.market_regime}")
        print(f"Inference Latency: {assessment.inference_latency_ms:.1f}ms")
        print(f"Data Freshness: {assessment.data_freshness_seconds}s")
        
        if assessment.correlation_signals:
            print(f"Correlation Signals: {', '.join(assessment.correlation_signals)}")
        
        if assessment.feature_importance:
            print(f"\nTop 5 Features:")
            sorted_features = sorted(assessment.feature_importance.items(), 
                                   key=lambda x: x[1], reverse=True)
            for i, (feature, importance) in enumerate(sorted_features[:5], 1):
                print(f"  {i}. {feature}: {importance:.3f}")
    else:
        print(f"❌ Failed to get risk assessment for {asset}")

async def get_multi_asset_scores(assets: list, model_path: str = None):
    """Get risk scores for multiple assets"""
    service = RiskInferenceService(model_path)
    
    if not service.current_model:
        if not service.load_latest_model():
            print(f"❌ Error: Could not load model")
            return
    
    print(f"🔍 Getting risk assessments for {len(assets)} assets...")
    assessments = await service.get_multi_asset_assessment(assets)
    
    if assessments:
        print(f"\n📊 Multi-Asset Risk Assessment")
        print(f"{'='*50}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\n{'Asset':<6} {'Risk Score':<10} {'Level':<8} {'Latency':<8}")
        print(f"{'-'*40}")
        
        for asset, assessment in assessments.items():
            print(f"{asset:<6} {assessment.combined_risk_score:<10.3f} {assessment.risk_level:<8} {assessment.inference_latency_ms:<8.1f}ms")
        
        # Summary statistics
        risk_scores = [a.combined_risk_score for a in assessments.values()]
        avg_risk = sum(risk_scores) / len(risk_scores)
        max_risk = max(risk_scores)
        min_risk = min(risk_scores)
        
        print(f"\n📈 Summary:")
        print(f"Average Risk: {avg_risk:.3f}")
        print(f"Highest Risk: {max_risk:.3f}")
        print(f"Lowest Risk: {min_risk:.3f}")
        
        # Alerts
        high_risk_assets = [asset for asset, assessment in assessments.items() 
                           if assessment.combined_risk_score >= 0.7]
        if high_risk_assets:
            print(f"\n🚨 HIGH RISK ALERTS: {', '.join(high_risk_assets)}")
    else:
        print(f"❌ Failed to get risk assessments")

async def monitor_continuous(asset: str, interval: int, iterations: int, model_path: str = None):
    """Continuously monitor risk for an asset"""
    service = RiskInferenceService(model_path)
    
    if not service.current_model:
        if not service.load_latest_model():
            print(f"❌ Error: Could not load model")
            return
    
    print(f"🔄 Starting continuous monitoring for {asset}")
    print(f"Interval: {interval}s, Iterations: {iterations}")
    print(f"Model: {service.model_version}")
    print(f"\n{'Time':<10} {'Risk Score':<10} {'Level':<8} {'Latency':<8}")
    print(f"{'-'*40}")
    
    for i in range(iterations):
        assessment = await service.get_risk_assessment(asset)
        
        if assessment:
            timestamp = assessment.timestamp.strftime("%H:%M:%S")
            print(f"{timestamp:<10} {assessment.combined_risk_score:<10.3f} {assessment.risk_level:<8} {assessment.inference_latency_ms:<8.1f}ms")
            
            # Alert on high risk
            if assessment.combined_risk_score >= 0.7:
                print(f"🚨 HIGH RISK ALERT: {assessment.combined_risk_score:.3f}")
        else:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"{timestamp:<10} {'ERROR':<10} {'N/A':<8} {'N/A':<8}")
        
        # Wait before next iteration (except for the last one)
        if i < iterations - 1:
            await asyncio.sleep(interval)
    
    print(f"\n✅ Monitoring complete")

async def service_status(model_path: str = None):
    """Show service status"""
    service = RiskInferenceService(model_path)
    
    if not service.current_model:
        service.load_latest_model()
    
    status = service.get_service_status()
    
    print(f"🔧 AI Risk Oracle Inference Service Status")
    print(f"{'='*45}")
    print(f"Service Status: {status['status']}")
    print(f"Model Version: {status['model_version']}")
    print(f"Model Loaded: {status['model_loaded_at'] or 'Not loaded'}")
    print(f"Cache Size: {status['cache_size']} items")
    
    print(f"\n📦 Components:")
    for component, status_val in status['components'].items():
        icon = "✅" if status_val == "available" else "❌"
        print(f"  {icon} {component}: {status_val}")

def main():
    parser = argparse.ArgumentParser(description="AI Risk Oracle Inference Service CLI")
    parser.add_argument("--model", "-m", help="Path to model file (optional)")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Single asset command
    single_parser = subparsers.add_parser("assess", help="Get risk assessment for single asset")
    single_parser.add_argument("asset", help="Asset symbol (e.g., BTC, ETH, XRP)")
    
    # Multi-asset command
    multi_parser = subparsers.add_parser("multi", help="Get risk assessments for multiple assets")
    multi_parser.add_argument("assets", nargs="+", help="Asset symbols (e.g., BTC ETH XRP)")
    
    # Monitor command
    monitor_parser = subparsers.add_parser("monitor", help="Continuously monitor an asset")
    monitor_parser.add_argument("asset", help="Asset symbol to monitor")
    monitor_parser.add_argument("--interval", "-i", type=int, default=30, help="Monitoring interval in seconds (default: 30)")
    monitor_parser.add_argument("--iterations", "-n", type=int, default=10, help="Number of iterations (default: 10)")
    
    # Status command
    subparsers.add_parser("status", help="Show service status")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Run the appropriate command
    if args.command == "assess":
        asyncio.run(get_risk_score(args.asset.upper(), args.model))
    elif args.command == "multi":
        assets = [asset.upper() for asset in args.assets]
        asyncio.run(get_multi_asset_scores(assets, args.model))
    elif args.command == "monitor":
        asyncio.run(monitor_continuous(args.asset.upper(), args.interval, args.iterations, args.model))
    elif args.command == "status":
        asyncio.run(service_status(args.model))

if __name__ == "__main__":
    main()