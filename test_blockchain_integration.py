#!/usr/bin/env python3
"""
Test script for blockchain integration functionality
Tests Web3 client and blockchain integration service
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from services.blockchain.web3_client import Web3Client
from services.blockchain.blockchain_integration import BlockchainIntegrationService
from services.inference.risk_inference_service import RiskAssessment

async def test_web3_client():
    """Test Web3 client functionality"""
    print("🔗 Testing Web3 Client...")
    
    client = Web3Client()
    
    # Test connection status (without actual connection)
    status = client.get_connection_status()
    print(f"Connection Status: {status}")
    
    # Test configuration loading
    print(f"Contract Address: {client.contract_address}")
    print(f"RPC URL: {client.rpc_url}")
    print(f"Backup RPC URL: {client.backup_rpc_url}")
    
    print("✅ Web3 Client tests completed")

async def test_blockchain_integration():
    """Test blockchain integration service"""
    print("🔧 Testing Blockchain Integration Service...")
    
    service = BlockchainIntegrationService()
    
    # Test configuration
    print(f"Risk Threshold: {service.risk_threshold}")
    print(f"Update Interval: {service.update_interval_seconds}s")
    
    # Test trigger logic with mock risk assessment
    mock_assessment = RiskAssessment(
        asset_symbol="BTC",
        timestamp=datetime.now(),
        ml_risk_score=0.9,
        sentiment_risk_score=0.8,
        combined_risk_score=0.85,
        confidence=0.9,
        risk_level="CRITICAL",
        market_regime="crisis",
        correlation_signals=["gold_spike_detected"],
        feature_importance={"price_change_24h": 0.3, "volatility": 0.2},
        inference_latency_ms=150.0,
        data_freshness_seconds=2
    )
    
    # Test triggering logic
    should_trigger, reason = service._should_trigger_update(mock_assessment)
    print(f"Should Trigger: {should_trigger}, Reason: {reason}")
    
    # Test status
    status = await service.get_blockchain_status()
    print(f"Service Status: {status['service_status']}")
    
    print("✅ Blockchain Integration Service tests completed")

async def test_risk_level_determination():
    """Test risk level determination"""
    print("📊 Testing Risk Level Determination...")
    
    service = BlockchainIntegrationService()
    
    test_scores = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    for score in test_scores:
        level = service._determine_risk_level(score)
        print(f"Risk Score: {score:.1f} → Level: {level}")
    
    print("✅ Risk Level Determination tests completed")

async def test_error_handling():
    """Test error handling scenarios"""
    print("⚠️  Testing Error Handling...")
    
    service = BlockchainIntegrationService()
    
    # Test with invalid risk assessment
    try:
        invalid_assessment = RiskAssessment(
            asset_symbol="INVALID",
            timestamp=datetime.now(),
            ml_risk_score=1.5,  # Invalid score > 1.0
            sentiment_risk_score=0.5,
            combined_risk_score=1.2,  # Invalid score > 1.0
            confidence=0.8,
            risk_level="UNKNOWN",
            market_regime="invalid",
            correlation_signals=[],
            feature_importance={},
            inference_latency_ms=0.0,
            data_freshness_seconds=0
        )
        
        should_trigger, reason = service._should_trigger_update(invalid_assessment)
        print(f"Invalid Assessment Handling: Should Trigger: {should_trigger}")
        
    except Exception as e:
        print(f"Error handling test: {e}")
    
    print("✅ Error Handling tests completed")

async def test_configuration_validation():
    """Test configuration validation"""
    print("⚙️  Testing Configuration Validation...")
    
    client = Web3Client()
    
    # Test configuration values
    configs = {
        "RPC URL": client.rpc_url,
        "Contract Address": client.contract_address,
        "Gas Limit": client.default_gas_limit,
        "Gas Price": client.default_gas_price,
        "Max Retries": client.max_retries,
        "Backoff Factor": client.retry_backoff_factor
    }
    
    for name, value in configs.items():
        print(f"{name}: {value}")
        
        # Basic validation
        if name == "Gas Limit":
            assert value > 0, "Gas limit should be positive"
        elif name == "Max Retries":
            assert value >= 0, "Max retries should be non-negative"
        elif name == "Backoff Factor":
            assert value >= 1.0, "Backoff factor should be >= 1.0"
    
    print("✅ Configuration Validation tests completed")

async def main():
    """Run all blockchain integration tests"""
    print("🚀 Starting Blockchain Integration Tests")
    print("=" * 50)
    
    try:
        await test_web3_client()
        print()
        
        await test_blockchain_integration()
        print()
        
        await test_risk_level_determination()
        print()
        
        await test_error_handling()
        print()
        
        await test_configuration_validation()
        print()
        
        print("=" * 50)
        print("✅ All Blockchain Integration Tests Completed Successfully!")
        
        # Summary
        print("\n📋 Test Summary:")
        print("- Web3 Client: Configuration and status checks")
        print("- Blockchain Integration: Service initialization and trigger logic")
        print("- Risk Level Determination: Score to level mapping")
        print("- Error Handling: Invalid input handling")
        print("- Configuration Validation: Parameter validation")
        
        print("\n🔧 Next Steps:")
        print("1. Configure blockchain environment variables in .env")
        print("2. Deploy or get AI Risk Oracle contract address")
        print("3. Set up QIE blockchain RPC endpoints")
        print("4. Test with actual blockchain connection")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)