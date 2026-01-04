"""
Blockchain Integration Service
Bridges AI Risk Oracle inference results to blockchain transactions
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from config import get_config
from logging_config import get_data_logger, get_performance_logger
from services.blockchain.web3_client import Web3Client, TransactionResult
from services.inference.risk_inference_service import RiskAssessment

class TriggerReason(Enum):
    """Reasons for triggering blockchain updates"""
    RISK_THRESHOLD_EXCEEDED = "risk_threshold_exceeded"
    PERIODIC_UPDATE = "periodic_update"
    MANUAL_TRIGGER = "manual_trigger"
    SYSTEM_RECOVERY = "system_recovery"

@dataclass
class BlockchainUpdate:
    """Record of a blockchain update"""
    trigger_reason: TriggerReason
    risk_assessment: RiskAssessment
    transaction_result: TransactionResult
    triggered_at: datetime
    completed_at: Optional[datetime]
    retry_count: int

class BlockchainIntegrationService:
    """Service for managing blockchain integration of AI risk assessments"""
    
    def __init__(self):
        self.config = get_config()
        self.data_logger = get_data_logger()
        self.performance_logger = get_performance_logger()
        
        # Initialize Web3 client
        self.web3_client = Web3Client()
        
        # Configuration
        self.risk_threshold = getattr(self.config, 'risk_threshold', 0.85)
        self.update_interval_seconds = getattr(self.config, 'blockchain_update_interval', 300)  # 5 minutes
        self.force_update_interval_seconds = getattr(self.config, 'blockchain_force_update_interval', 3600)  # 1 hour
        
        # State tracking
        self.last_update_time: Optional[datetime] = None
        self.last_risk_score: Optional[float] = None
        self.update_history: List[BlockchainUpdate] = []
        self.is_running = False
        
        # Performance tracking
        self.total_updates = 0
        self.successful_updates = 0
        self.failed_updates = 0
    
    async def initialize(self) -> bool:
        """Initialize the blockchain integration service"""
        try:
            start_time = time.time()
            
            # Connect to blockchain
            connected = await self.web3_client.connect()
            
            if not connected:
                raise ConnectionError("Failed to connect to QIE blockchain")
            
            # Verify contract access
            current_score_result = await self.web3_client.get_current_risk_score()
            if not current_score_result.success:
                self.data_logger.log_data_collection(
                    "contract_verification", "warning", 0, False,
                    f"Cannot read from contract: {current_score_result.error_message}"
                )
            else:
                self.data_logger.log_data_collection(
                    "contract_verification", "success", 1, True,
                    f"Contract accessible, current score: {current_score_result.return_value}"
                )
            
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time("blockchain_initialization", duration_ms, True)
            
            self.data_logger.log_data_collection(
                "blockchain_integration_init", "success", 1, True,
                "Blockchain integration service initialized"
            )
            
            return True
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time("blockchain_initialization", duration_ms, False)
            
            self.data_logger.log_data_collection(
                "blockchain_integration_init", "error", 0, False, str(e)
            )
            return False
    
    async def process_risk_assessment(self, 
                                    risk_assessment: RiskAssessment,
                                    force_update: bool = False) -> Optional[BlockchainUpdate]:
        """Process a risk assessment and potentially trigger blockchain update"""
        try:
            # Determine if we should trigger an update
            should_update, trigger_reason = self._should_trigger_update(
                risk_assessment, force_update
            )
            
            if not should_update:
                return None
            
            # Create update record
            update = BlockchainUpdate(
                trigger_reason=trigger_reason,
                risk_assessment=risk_assessment,
                transaction_result=None,  # Will be filled after transaction
                triggered_at=datetime.now(),
                completed_at=None,
                retry_count=0
            )
            
            # Send blockchain transaction
            transaction_result = await self.web3_client.update_risk_score(
                risk_assessment.combined_risk_score
            )
            
            # Update the record
            update.transaction_result = transaction_result
            update.completed_at = datetime.now()
            
            # Track statistics
            self.total_updates += 1
            if transaction_result.success:
                self.successful_updates += 1
                self.last_update_time = datetime.now()
                self.last_risk_score = risk_assessment.combined_risk_score
            else:
                self.failed_updates += 1
            
            # Store in history
            self.update_history.append(update)
            
            # Keep only recent history (last 100 updates)
            if len(self.update_history) > 100:
                self.update_history = self.update_history[-100:]
            
            # Log the update
            if transaction_result.success:
                self.data_logger.log_data_collection(
                    "blockchain_update", "success", 1, True,
                    f"Risk score {risk_assessment.combined_risk_score:.3f} updated on blockchain, "
                    f"reason: {trigger_reason.value}, tx: {transaction_result.tx_hash}"
                )
            else:
                self.data_logger.log_data_collection(
                    "blockchain_update", "failed", 0, False,
                    f"Failed to update blockchain: {transaction_result.error_message}, "
                    f"reason: {trigger_reason.value}"
                )
            
            return update
            
        except Exception as e:
            self.data_logger.log_data_collection(
                "blockchain_update", "error", 0, False, str(e)
            )
            return None
    
    def _should_trigger_update(self, 
                             risk_assessment: RiskAssessment,
                             force_update: bool = False) -> Tuple[bool, TriggerReason]:
        """Determine if blockchain update should be triggered"""
        
        # Force update always triggers
        if force_update:
            return True, TriggerReason.MANUAL_TRIGGER
        
        # Check risk threshold
        if risk_assessment.combined_risk_score >= self.risk_threshold:
            return True, TriggerReason.RISK_THRESHOLD_EXCEEDED
        
        # Check if enough time has passed for periodic update
        if self.last_update_time is None:
            return True, TriggerReason.PERIODIC_UPDATE
        
        time_since_last = (datetime.now() - self.last_update_time).total_seconds()
        
        # Force update if too much time has passed
        if time_since_last >= self.force_update_interval_seconds:
            return True, TriggerReason.PERIODIC_UPDATE
        
        # Regular periodic update if interval passed and risk changed significantly
        if time_since_last >= self.update_interval_seconds:
            if self.last_risk_score is None:
                return True, TriggerReason.PERIODIC_UPDATE
            
            # Update if risk score changed by more than 10%
            risk_change = abs(risk_assessment.combined_risk_score - self.last_risk_score)
            if risk_change >= 0.1:
                return True, TriggerReason.PERIODIC_UPDATE
        
        return False, None
    
    async def get_blockchain_status(self) -> Dict[str, Any]:
        """Get current blockchain integration status"""
        try:
            # Get Web3 client status
            connection_status = self.web3_client.get_connection_status()
            
            # Get current contract state
            contract_result = await self.web3_client.get_current_risk_score()
            
            # Calculate performance metrics
            success_rate = 0.0
            if self.total_updates > 0:
                success_rate = self.successful_updates / self.total_updates
            
            # Get recent update history
            recent_updates = []
            for update in self.update_history[-5:]:  # Last 5 updates
                recent_updates.append({
                    "trigger_reason": update.trigger_reason.value,
                    "risk_score": update.risk_assessment.combined_risk_score,
                    "success": update.transaction_result.success if update.transaction_result else False,
                    "tx_hash": update.transaction_result.tx_hash if update.transaction_result else None,
                    "timestamp": update.triggered_at.isoformat(),
                    "latency_ms": update.transaction_result.latency_ms if update.transaction_result else 0
                })
            
            return {
                "service_status": "active" if self.is_running else "inactive",
                "blockchain_connection": connection_status,
                "contract_state": {
                    "accessible": contract_result.success,
                    "current_risk_score": contract_result.return_value if contract_result.success else None,
                    "error": contract_result.error_message if not contract_result.success else None
                },
                "update_statistics": {
                    "total_updates": self.total_updates,
                    "successful_updates": self.successful_updates,
                    "failed_updates": self.failed_updates,
                    "success_rate": success_rate,
                    "last_update_time": self.last_update_time.isoformat() if self.last_update_time else None,
                    "last_risk_score": self.last_risk_score
                },
                "configuration": {
                    "risk_threshold": self.risk_threshold,
                    "update_interval_seconds": self.update_interval_seconds,
                    "force_update_interval_seconds": self.force_update_interval_seconds
                },
                "recent_updates": recent_updates
            }
            
        except Exception as e:
            return {
                "service_status": "error",
                "error": str(e)
            }
    
    async def manual_update(self, risk_score: float, reason: str = "Manual trigger") -> BlockchainUpdate:
        """Manually trigger a blockchain update"""
        try:
            # Create a mock risk assessment for manual update
            from services.inference.risk_inference_service import RiskAssessment
            
            mock_assessment = RiskAssessment(
                asset_symbol="MANUAL",
                timestamp=datetime.now(),
                ml_risk_score=risk_score,
                sentiment_risk_score=0.5,  # Neutral sentiment
                combined_risk_score=risk_score,
                confidence=1.0,  # High confidence for manual updates
                risk_level=self._determine_risk_level(risk_score),
                market_regime="manual",
                correlation_signals=[],
                feature_importance={},
                inference_latency_ms=0.0,
                data_freshness_seconds=0
            )
            
            # Process the manual update
            update = await self.process_risk_assessment(mock_assessment, force_update=True)
            
            if update:
                self.data_logger.log_data_collection(
                    "manual_blockchain_update", "triggered", 1, True,
                    f"Manual update triggered: {reason}, risk_score: {risk_score}"
                )
            
            return update
            
        except Exception as e:
            self.data_logger.log_data_collection(
                "manual_blockchain_update", "error", 0, False, str(e)
            )
            raise
    
    def _determine_risk_level(self, risk_score: float) -> str:
        """Determine risk level from score"""
        if risk_score >= 0.8:
            return "CRITICAL"
        elif risk_score >= 0.6:
            return "HIGH"
        elif risk_score >= 0.4:
            return "MEDIUM"
        else:
            return "LOW"
    
    async def start_monitoring(self):
        """Start the blockchain integration monitoring service"""
        self.is_running = True
        self.data_logger.log_data_collection(
            "blockchain_monitoring", "started", 1, True,
            "Blockchain integration monitoring started"
        )
    
    async def stop_monitoring(self):
        """Stop the blockchain integration monitoring service"""
        self.is_running = False
        await self.web3_client.disconnect()
        
        self.data_logger.log_data_collection(
            "blockchain_monitoring", "stopped", 1, True,
            "Blockchain integration monitoring stopped"
        )
    
    def get_update_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent update history"""
        history = []
        
        for update in self.update_history[-limit:]:
            history.append({
                "trigger_reason": update.trigger_reason.value,
                "asset_symbol": update.risk_assessment.asset_symbol,
                "risk_score": update.risk_assessment.combined_risk_score,
                "risk_level": update.risk_assessment.risk_level,
                "confidence": update.risk_assessment.confidence,
                "success": update.transaction_result.success if update.transaction_result else False,
                "tx_hash": update.transaction_result.tx_hash if update.transaction_result else None,
                "error": update.transaction_result.error_message if update.transaction_result and not update.transaction_result.success else None,
                "triggered_at": update.triggered_at.isoformat(),
                "completed_at": update.completed_at.isoformat() if update.completed_at else None,
                "latency_ms": update.transaction_result.latency_ms if update.transaction_result else 0,
                "retry_count": update.retry_count
            })
        
        return history
    
    def clear_history(self):
        """Clear update history"""
        self.update_history.clear()
        self.data_logger.log_data_collection(
            "blockchain_history", "cleared", 1, True, "Update history cleared"
        )