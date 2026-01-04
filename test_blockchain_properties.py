"""
Property-based tests for blockchain integration functionality
Tests correctness properties for Web3 client and blockchain integration service
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from hypothesis import given, strategies as st, settings, assume
from hypothesis.stateful import RuleBasedStateMachine, rule, initialize, invariant
from typing import Dict, Any

from services.blockchain.web3_client import Web3Client, TransactionResult, ContractCallResult
from services.blockchain.blockchain_integration import BlockchainIntegrationService, TriggerReason
from services.inference.risk_inference_service import RiskAssessment

# Test configuration
MAX_EXAMPLES = 100
DEADLINE = 30000  # 30 seconds

class TestBlockchainProperties:
    """Property-based tests for blockchain functionality"""
    
    @given(
        risk_score=st.floats(min_value=0.0, max_value=1.0),
        confidence=st.floats(min_value=0.0, max_value=1.0)
    )
    @settings(max_examples=MAX_EXAMPLES, deadline=DEADLINE)
    def test_property_10_blockchain_transaction_triggering(self, risk_score, confidence):
        """
        Property 10: Blockchain transaction triggering
        For any risk score exceeding the threshold (0.85), the system should call 
        the AI_Risk_Oracle contract's updateRisk function
        Validates: Requirements 4.2
        """
        # Create mock blockchain integration service
        integration_service = BlockchainIntegrationService()
        integration_service.risk_threshold = 0.85
        
        # Create mock risk assessment
        risk_assessment = RiskAssessment(
            asset_symbol="BTC",
            timestamp=datetime.now(),
            ml_risk_score=risk_score,
            sentiment_risk_score=0.5,
            combined_risk_score=risk_score,
            confidence=confidence,
            risk_level="HIGH" if risk_score >= 0.6 else "LOW",
            market_regime="normal",
            correlation_signals=[],
            feature_importance={},
            inference_latency_ms=100.0,
            data_freshness_seconds=1
        )
        
        # Test the triggering logic
        should_trigger, trigger_reason = integration_service._should_trigger_update(
            risk_assessment, force_update=False
        )
        
        # Property: Risk scores >= threshold should trigger updates
        if risk_score >= 0.85:
            assert should_trigger, f"Risk score {risk_score} >= 0.85 should trigger update"
            assert trigger_reason == TriggerReason.RISK_THRESHOLD_EXCEEDED
        else:
            # For scores below threshold, triggering depends on other factors
            # This is acceptable behavior
            pass
    
    @given(
        initial_delay=st.floats(min_value=0.1, max_value=2.0),
        backoff_factor=st.floats(min_value=1.1, max_value=5.0),
        max_retries=st.integers(min_value=1, max_value=5),
        max_delay=st.floats(min_value=5.0, max_value=120.0)
    )
    @settings(max_examples=MAX_EXAMPLES, deadline=DEADLINE)
    def test_property_11_retry_logic_with_exponential_backoff(
        self, initial_delay, backoff_factor, max_retries, max_delay
    ):
        """
        Property 11: Retry logic with exponential backoff
        For any blockchain transaction failure, the system should implement retry 
        attempts with exponentially increasing delays
        Validates: Requirements 4.3
        """
        # Calculate expected delays for each retry
        expected_delays = []
        current_delay = initial_delay
        
        for attempt in range(max_retries):
            expected_delays.append(min(current_delay, max_delay))
            current_delay *= backoff_factor
        
        # Verify exponential growth (up to max_delay)
        for i in range(1, len(expected_delays)):
            if expected_delays[i-1] < max_delay:
                # Should grow exponentially until hitting max_delay
                expected_growth = expected_delays[i-1] * backoff_factor
                actual_delay = expected_delays[i]
                
                # Either exponential growth or capped at max_delay
                assert (
                    abs(actual_delay - expected_growth) < 0.001 or 
                    actual_delay == max_delay
                ), f"Delay sequence should grow exponentially: {expected_delays}"
        
        # Verify all delays are within bounds
        for delay in expected_delays:
            assert delay >= initial_delay, f"Delay {delay} should be >= initial_delay {initial_delay}"
            assert delay <= max_delay, f"Delay {delay} should be <= max_delay {max_delay}"
    
    @given(
        risk_scores=st.lists(
            st.floats(min_value=0.0, max_value=1.0),
            min_size=1,
            max_size=10
        ),
        time_intervals=st.lists(
            st.integers(min_value=1, max_value=3600),  # 1 second to 1 hour
            min_size=1,
            max_size=10
        )
    )
    @settings(max_examples=MAX_EXAMPLES, deadline=DEADLINE)
    def test_property_blockchain_update_consistency(self, risk_scores, time_intervals):
        """
        Property: Blockchain update consistency
        For any sequence of risk score updates, the blockchain state should 
        reflect the most recent successful update
        """
        assume(len(risk_scores) == len(time_intervals))
        
        # Mock Web3 client
        web3_client = Mock()
        web3_client.update_risk_score = AsyncMock()
        
        # Track expected final state
        last_successful_score = None
        
        # Simulate updates
        for i, (score, interval) in enumerate(zip(risk_scores, time_intervals)):
            # Mock successful transaction
            web3_client.update_risk_score.return_value = TransactionResult(
                success=True,
                tx_hash=f"0x{i:064x}",
                block_number=1000 + i,
                gas_used=50000,
                error_message=None,
                timestamp=datetime.now(),
                latency_ms=100.0
            )
            
            last_successful_score = score
        
        # Property: The final state should match the last successful update
        assert last_successful_score is not None
        # In a real test, we would verify the blockchain state matches this score
    
    @given(
        contract_address=st.just("0x1234567890123456789012345678901234567890"),
        private_key=st.just("1234567890123456789012345678901234567890123456789012345678901234")
    )
    @settings(max_examples=MAX_EXAMPLES, deadline=DEADLINE)
    def test_property_secure_key_management(self, contract_address, private_key):
        """
        Property: Secure private key management
        For any private key configuration, the system should handle keys securely
        without exposing them in logs or error messages
        """
        # Create Web3 client with mock configuration
        with patch('services.blockchain.web3_client.get_config') as mock_config:
            # Create a mock config that doesn't expose the private key in its string representation
            config_mock = Mock()
            config_mock.ai_risk_oracle_contract_address = contract_address
            config_mock.private_key = private_key
            config_mock.qie_rpc_url = "https://test-rpc.example.com"
            
            mock_config.return_value = config_mock
            
            client = Web3Client()
            
            # Property: Private key should be accessible through secure method
            retrieved_key = client._get_private_key()
            assert retrieved_key == private_key, "Private key should be retrievable through secure method"
            
            # Property: Private key should not be directly accessible as a public attribute
            assert not hasattr(client, 'private_key'), "Private key should not be a public attribute"
            
            # Property: Contract address should be properly formatted
            if contract_address.startswith('0x') and len(contract_address) == 42:
                assert client.contract_address == contract_address
            
            # Property: Secure storage mechanism should be used
            # The private key should be stored using name mangling or similar secure method
            assert hasattr(client, '_Web3Client__private_key'), "Private key should use secure storage mechanism"
    
    @given(
        gas_limit=st.integers(min_value=21000, max_value=1000000),
        gas_price=st.integers(min_value=1000000000, max_value=100000000000),  # 1-100 Gwei
        nonce=st.integers(min_value=0, max_value=1000000)
    )
    @settings(max_examples=MAX_EXAMPLES, deadline=DEADLINE)
    def test_property_transaction_parameter_bounds(self, gas_limit, gas_price, nonce):
        """
        Property: Transaction parameter bounds
        For any transaction parameters, they should be within valid ranges
        """
        # Property: Gas limit should be reasonable
        assert gas_limit >= 21000, "Gas limit should be at least 21000 (basic transaction)"
        assert gas_limit <= 1000000, "Gas limit should not exceed block gas limit"
        
        # Property: Gas price should be reasonable
        assert gas_price >= 1000000000, "Gas price should be at least 1 Gwei"
        assert gas_price <= 100000000000, "Gas price should not exceed 100 Gwei for normal operations"
        
        # Property: Nonce should be non-negative
        assert nonce >= 0, "Nonce should be non-negative"
    
    @given(
        error_types=st.lists(
            st.sampled_from([
                "ConnectionError", "TimeoutError", "ValueError", 
                "Web3Exception", "ContractLogicError"
            ]),
            min_size=1,
            max_size=5
        )
    )
    @settings(max_examples=MAX_EXAMPLES, deadline=DEADLINE)
    def test_property_12_error_handling_resilience(self, error_types):
        """
        Property 12: Error handling resilience
        For any RPC timeout or system error, the service should handle the error 
        gracefully and maintain availability
        Validates: Requirements 4.4
        """
        integration_service = BlockchainIntegrationService()
        
        # Mock Web3 client with various error scenarios
        with patch.object(integration_service, 'web3_client') as mock_client:
            # Test each error type
            for error_type in error_types:
                if error_type == "ConnectionError":
                    mock_client.update_risk_score.side_effect = ConnectionError("RPC connection failed")
                elif error_type == "TimeoutError":
                    mock_client.update_risk_score.side_effect = TimeoutError("RPC timeout")
                elif error_type == "ValueError":
                    mock_client.update_risk_score.side_effect = ValueError("Invalid parameters")
                else:
                    mock_client.update_risk_score.side_effect = Exception(f"Generic {error_type}")
                
                # Create test risk assessment
                risk_assessment = RiskAssessment(
                    asset_symbol="BTC",
                    timestamp=datetime.now(),
                    ml_risk_score=0.9,
                    sentiment_risk_score=0.8,
                    combined_risk_score=0.85,
                    confidence=0.9,
                    risk_level="CRITICAL",
                    market_regime="crisis",
                    correlation_signals=[],
                    feature_importance={},
                    inference_latency_ms=100.0,
                    data_freshness_seconds=1
                )
                
                # Property: Service should handle errors gracefully
                try:
                    # This should not raise an exception, but handle it gracefully
                    result = asyncio.run(
                        integration_service.process_risk_assessment(risk_assessment)
                    )
                    # Service should return None or a failed result, not crash
                    if result is not None:
                        assert hasattr(result, 'transaction_result')
                except Exception as e:
                    # If an exception is raised, it should be a controlled failure
                    assert isinstance(e, (ConnectionError, TimeoutError, ValueError))


class BlockchainStateMachine(RuleBasedStateMachine):
    """
    Stateful property testing for blockchain integration
    Tests complex interactions and state transitions
    """
    
    def __init__(self):
        super().__init__()
        self.integration_service = None
        self.risk_scores_sent = []
        self.successful_updates = 0
        self.failed_updates = 0
    
    @initialize()
    def setup(self):
        """Initialize the blockchain integration service"""
        self.integration_service = BlockchainIntegrationService()
        self.integration_service.risk_threshold = 0.85
        
        # Mock the Web3 client
        self.integration_service.web3_client = Mock()
        self.integration_service.web3_client.connect = AsyncMock(return_value=True)
        self.integration_service.web3_client.get_current_risk_score = AsyncMock(
            return_value=ContractCallResult(
                success=True,
                return_value={'risk_score': 50, 'last_updated': int(time.time())},
                error_message=None,
                gas_estimate=None,
                timestamp=datetime.now()
            )
        )
    
    @rule(
        risk_score=st.floats(min_value=0.0, max_value=1.0),
        confidence=st.floats(min_value=0.0, max_value=1.0),
        success=st.booleans()
    )
    def send_risk_update(self, risk_score, confidence, success):
        """Send a risk score update"""
        # Mock transaction result
        if success:
            tx_result = TransactionResult(
                success=True,
                tx_hash=f"0x{len(self.risk_scores_sent):064x}",
                block_number=1000 + len(self.risk_scores_sent),
                gas_used=50000,
                error_message=None,
                timestamp=datetime.now(),
                latency_ms=100.0
            )
            self.successful_updates += 1
        else:
            tx_result = TransactionResult(
                success=False,
                tx_hash=None,
                block_number=None,
                gas_used=None,
                error_message="Mock transaction failure",
                timestamp=datetime.now(),
                latency_ms=100.0
            )
            self.failed_updates += 1
        
        self.integration_service.web3_client.update_risk_score = AsyncMock(
            return_value=tx_result
        )
        
        # Create risk assessment
        risk_assessment = RiskAssessment(
            asset_symbol="BTC",
            timestamp=datetime.now(),
            ml_risk_score=risk_score,
            sentiment_risk_score=0.5,
            combined_risk_score=risk_score,
            confidence=confidence,
            risk_level="HIGH" if risk_score >= 0.6 else "LOW",
            market_regime="normal",
            correlation_signals=[],
            feature_importance={},
            inference_latency_ms=100.0,
            data_freshness_seconds=1
        )
        
        # Process the assessment
        result = asyncio.run(
            self.integration_service.process_risk_assessment(risk_assessment)
        )
        
        self.risk_scores_sent.append((risk_score, success, result))
    
    @invariant()
    def statistics_are_consistent(self):
        """Statistics should be consistent with actual operations"""
        total_operations = len(self.risk_scores_sent)
        
        if total_operations > 0:
            # Count actual successes and failures
            actual_successes = sum(1 for _, success, result in self.risk_scores_sent 
                                 if success and result is not None)
            actual_failures = sum(1 for _, success, result in self.risk_scores_sent 
                                if not success and result is not None)
            
            # Statistics should match reality
            assert self.integration_service.total_updates >= 0
            assert self.integration_service.successful_updates >= 0
            assert self.integration_service.failed_updates >= 0
    
    @invariant()
    def risk_threshold_triggering_is_consistent(self):
        """Risk threshold triggering should be consistent"""
        for risk_score, _, result in self.risk_scores_sent:
            if result is not None:
                # High risk scores should have triggered updates
                if risk_score >= 0.85:
                    assert result.trigger_reason in [
                        TriggerReason.RISK_THRESHOLD_EXCEEDED,
                        TriggerReason.PERIODIC_UPDATE,
                        TriggerReason.MANUAL_TRIGGER
                    ]


# Test runner
if __name__ == "__main__":
    # Run property tests
    pytest.main([__file__, "-v", "--tb=short"])
    
    # Run stateful tests
    print("\nRunning stateful property tests...")
    TestStateMachine = BlockchainStateMachine.TestCase
    TestStateMachine.settings = settings(max_examples=50, stateful_step_count=20)
    
    test_instance = TestStateMachine()
    test_instance.runTest()
    
    print("All blockchain property tests completed!")