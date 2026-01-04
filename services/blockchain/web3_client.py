"""
Web3 Blockchain Client for QIE AI Risk Oracle
Handles secure blockchain transactions and contract interactions
"""

import asyncio
import time
from datetime import datetime
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import json

from web3 import Web3
from eth_account import Account
from eth_typing import ChecksumAddress

# Try to import PoA middleware (different names in different versions)
try:
    from web3.middleware import extra_data_to_hash_middleware as poa_middleware
except ImportError:
    try:
        from web3.middleware import geth_poa_middleware as poa_middleware
    except ImportError:
        try:
            from web3.middleware.geth_poa import geth_poa_middleware as poa_middleware
        except ImportError:
            # Fallback for versions without PoA middleware
            poa_middleware = None

from config import get_config
from logging_config import get_data_logger, get_performance_logger

@dataclass
class TransactionResult:
    """Result of a blockchain transaction"""
    success: bool
    tx_hash: Optional[str]
    block_number: Optional[int]
    gas_used: Optional[int]
    error_message: Optional[str]
    timestamp: datetime
    latency_ms: float

@dataclass
class ContractCallResult:
    """Result of a contract function call"""
    success: bool
    return_value: Any
    error_message: Optional[str]
    gas_estimate: Optional[int]
    timestamp: datetime

class Web3Client:
    """Web3 client for QIE blockchain interactions"""
    
    def __init__(self):
        self.config = get_config()
        self.data_logger = get_data_logger()
        self.performance_logger = get_performance_logger()
        
        # Initialize Web3 connection
        self.w3: Optional[Web3] = None
        self.account: Optional[Account] = None
        self.contract = None
        
        # Configuration
        self.rpc_url = getattr(self.config, 'qie_rpc_url', 'https://rpc.qie-pulse.com')
        self.backup_rpc_url = getattr(self.config, 'qie_rpc_backup_url', 'https://backup-rpc.qie-pulse.com')
        self.contract_address = getattr(self.config, 'ai_risk_oracle_contract_address', '')
        
        # Store private key securely (not in __dict__)
        private_key = getattr(self.config, 'private_key', '')
        if private_key:
            # Store in a way that doesn't appear in __dict__
            object.__setattr__(self, '_Web3Client__private_key', private_key)
        else:
            object.__setattr__(self, '_Web3Client__private_key', None)
        
        # Gas configuration
        self.default_gas_limit = getattr(self.config, 'default_gas_limit', 100000)
        self.default_gas_price = getattr(self.config, 'default_gas_price', 20000000000)  # 20 Gwei
        
        # Retry configuration
        self.max_retries = getattr(self.config, 'max_retries', 3)
        self.retry_backoff_factor = getattr(self.config, 'retry_backoff_factor', 2.0)
        self.retry_max_delay = getattr(self.config, 'retry_max_delay', 60.0)
        
        # Connection state
        self.is_connected = False
        self.current_rpc_url = self.rpc_url
        
        # Load contract ABI
        self.contract_abi = self._load_contract_abi()
    
    def _get_private_key(self) -> Optional[str]:
        """Securely retrieve the private key"""
        return getattr(self, '_Web3Client__private_key', None)
    
    def _load_contract_abi(self) -> list:
        """Load the AI Risk Oracle contract ABI"""
        try:
            # Simple ABI for the AIRiskOracle contract
            return [
                {
                    "inputs": [{"internalType": "address", "name": "_updater", "type": "address"}],
                    "stateMutability": "nonpayable",
                    "type": "constructor"
                },
                {
                    "anonymous": False,
                    "inputs": [
                        {"indexed": False, "internalType": "uint8", "name": "score", "type": "uint8"},
                        {"indexed": False, "internalType": "uint256", "name": "timestamp", "type": "uint256"}
                    ],
                    "name": "RiskUpdated",
                    "type": "event"
                },
                {
                    "inputs": [],
                    "name": "lastUpdated",
                    "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
                    "stateMutability": "view",
                    "type": "function"
                },
                {
                    "inputs": [],
                    "name": "riskScore",
                    "outputs": [{"internalType": "uint8", "name": "", "type": "uint8"}],
                    "stateMutability": "view",
                    "type": "function"
                },
                {
                    "inputs": [{"internalType": "uint8", "name": "_score", "type": "uint8"}],
                    "name": "updateRisk",
                    "outputs": [],
                    "stateMutability": "nonpayable",
                    "type": "function"
                },
                {
                    "inputs": [],
                    "name": "updater",
                    "outputs": [{"internalType": "address", "name": "", "type": "address"}],
                    "stateMutability": "view",
                    "type": "function"
                }
            ]
        except Exception as e:
            self.data_logger.log_data_collection(
                "contract_abi_loading", "error", 0, False, str(e)
            )
            return []
    
    async def connect(self) -> bool:
        """Establish connection to QIE blockchain"""
        try:
            start_time = time.time()
            
            # Try primary RPC first
            success = await self._try_connect(self.rpc_url)
            
            if not success:
                # Try backup RPC
                self.data_logger.log_data_collection(
                    "rpc_connection", "primary_failed", 0, False, 
                    f"Primary RPC failed, trying backup: {self.backup_rpc_url}"
                )
                success = await self._try_connect(self.backup_rpc_url)
                if success:
                    self.current_rpc_url = self.backup_rpc_url
            
            if success:
                # Load account from private key
                private_key = self._get_private_key()
                if private_key:
                    try:
                        self.account = Account.from_key(private_key)
                        self.data_logger.log_data_collection(
                            "account_loading", "success", 1, True,
                            f"Account loaded: {self.account.address}"
                        )
                    except Exception as e:
                        self.data_logger.log_data_collection(
                            "account_loading", "error", 0, False, str(e)
                        )
                        return False
                
                # Initialize contract
                if self.contract_address and self.contract_abi:
                    try:
                        self.contract = self.w3.eth.contract(
                            address=Web3.to_checksum_address(self.contract_address),
                            abi=self.contract_abi
                        )
                        self.data_logger.log_data_collection(
                            "contract_loading", "success", 1, True,
                            f"Contract loaded: {self.contract_address}"
                        )
                    except Exception as e:
                        self.data_logger.log_data_collection(
                            "contract_loading", "error", 0, False, str(e)
                        )
                        return False
                
                self.is_connected = True
                
                duration_ms = (time.time() - start_time) * 1000
                self.performance_logger.log_inference_time("blockchain_connection", duration_ms, True)
                
                self.data_logger.log_data_collection(
                    "blockchain_connection", "success", 1, True,
                    f"Connected to {self.current_rpc_url}"
                )
                
                return True
            
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time("blockchain_connection", duration_ms, False)
            
            return False
            
        except Exception as e:
            self.data_logger.log_data_collection(
                "blockchain_connection", "error", 0, False, str(e)
            )
            return False
    
    async def _try_connect(self, rpc_url: str) -> bool:
        """Try to connect to a specific RPC URL"""
        try:
            # Create Web3 instance
            self.w3 = Web3(Web3.HTTPProvider(rpc_url))
            
            # Add PoA middleware if available (common for private chains like QIE)
            if poa_middleware:
                self.w3.middleware_onion.inject(poa_middleware, layer=0)
            
            # Test connection
            if not self.w3.is_connected():
                return False
            
            # Test basic functionality
            latest_block = self.w3.eth.get_block('latest')
            if not latest_block:
                return False
            
            return True
            
        except Exception as e:
            self.data_logger.log_data_collection(
                "rpc_connection_attempt", rpc_url, 0, False, str(e)
            )
            return False
    
    async def update_risk_score(self, risk_score: float) -> TransactionResult:
        """Update risk score on the AI Risk Oracle contract"""
        start_time = time.time()
        
        try:
            if not self.is_connected:
                await self.connect()
            
            if not self.is_connected or not self.contract or not self.account:
                raise ValueError("Not connected to blockchain or missing configuration")
            
            # Convert risk score (0.0-1.0) to uint8 (0-100)
            score_uint8 = min(100, max(0, int(risk_score * 100)))
            
            # Build transaction
            transaction = await self._build_transaction(
                self.contract.functions.updateRisk(score_uint8)
            )
            
            if not transaction:
                raise ValueError("Failed to build transaction")
            
            # Send transaction with retry logic
            result = await self._send_transaction_with_retry(transaction)
            
            # Log performance
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time(
                "risk_score_update", duration_ms, result.success
            )
            
            if result.success:
                self.data_logger.log_data_collection(
                    "risk_score_update", "success", 1, True,
                    f"Risk score {risk_score:.3f} updated, tx: {result.tx_hash}"
                )
            else:
                self.data_logger.log_data_collection(
                    "risk_score_update", "failed", 0, False,
                    f"Failed to update risk score: {result.error_message}"
                )
            
            return result
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time("risk_score_update", duration_ms, False)
            
            error_msg = str(e)
            self.data_logger.log_data_collection(
                "risk_score_update", "error", 0, False, error_msg
            )
            
            return TransactionResult(
                success=False,
                tx_hash=None,
                block_number=None,
                gas_used=None,
                error_message=error_msg,
                timestamp=datetime.now(),
                latency_ms=duration_ms
            )
    
    async def _build_transaction(self, contract_function) -> Optional[Dict]:
        """Build a transaction for a contract function call"""
        try:
            # Get current nonce
            nonce = self.w3.eth.get_transaction_count(self.account.address)
            
            # Estimate gas
            try:
                gas_estimate = contract_function.estimate_gas({'from': self.account.address})
                gas_limit = min(gas_estimate + 20000, self.default_gas_limit)  # Add buffer
            except Exception:
                gas_limit = self.default_gas_limit
            
            # Get current gas price
            try:
                gas_price = self.w3.eth.gas_price
                # Use higher of current price or default
                gas_price = max(gas_price, self.default_gas_price)
            except Exception:
                gas_price = self.default_gas_price
            
            # Build transaction
            transaction = contract_function.build_transaction({
                'from': self.account.address,
                'nonce': nonce,
                'gas': gas_limit,
                'gasPrice': gas_price,
            })
            
            return transaction
            
        except Exception as e:
            self.data_logger.log_data_collection(
                "transaction_building", "error", 0, False, str(e)
            )
            return None
    
    async def _send_transaction_with_retry(self, transaction: Dict) -> TransactionResult:
        """Send transaction with exponential backoff retry logic"""
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                start_time = time.time()
                
                # Sign transaction
                private_key = self._get_private_key()
                if not private_key:
                    raise ValueError("Private key not available for signing")
                signed_txn = self.w3.eth.account.sign_transaction(transaction, private_key)
                
                # Send transaction
                tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
                
                # Wait for confirmation
                receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=30)
                
                duration_ms = (time.time() - start_time) * 1000
                
                return TransactionResult(
                    success=True,
                    tx_hash=tx_hash.hex(),
                    block_number=receipt.blockNumber,
                    gas_used=receipt.gasUsed,
                    error_message=None,
                    timestamp=datetime.now(),
                    latency_ms=duration_ms
                )
                
            except Exception as e:
                last_error = str(e)
                
                self.data_logger.log_data_collection(
                    "transaction_attempt", f"attempt_{attempt + 1}", 0, False,
                    f"Transaction failed: {last_error}"
                )
                
                # If this is not the last attempt, wait before retrying
                if attempt < self.max_retries:
                    # Exponential backoff with jitter
                    delay = min(
                        self.retry_backoff_factor ** attempt,
                        self.retry_max_delay
                    )
                    
                    self.data_logger.log_data_collection(
                        "transaction_retry", f"delay_{delay}s", 0, False,
                        f"Retrying in {delay}s (attempt {attempt + 1}/{self.max_retries})"
                    )
                    
                    await asyncio.sleep(delay)
                    
                    # Update nonce for retry (in case of nonce issues)
                    try:
                        transaction['nonce'] = self.w3.eth.get_transaction_count(self.account.address)
                    except Exception:
                        pass  # Continue with existing nonce
        
        # All retries failed
        return TransactionResult(
            success=False,
            tx_hash=None,
            block_number=None,
            gas_used=None,
            error_message=f"Transaction failed after {self.max_retries} retries: {last_error}",
            timestamp=datetime.now(),
            latency_ms=0.0
        )
    
    async def get_current_risk_score(self) -> ContractCallResult:
        """Get the current risk score from the contract"""
        try:
            if not self.is_connected:
                await self.connect()
            
            if not self.is_connected or not self.contract:
                raise ValueError("Not connected to blockchain")
            
            # Call contract function
            risk_score = self.contract.functions.riskScore().call()
            last_updated = self.contract.functions.lastUpdated().call()
            
            return ContractCallResult(
                success=True,
                return_value={
                    'risk_score': risk_score,
                    'last_updated': last_updated,
                    'risk_score_normalized': risk_score / 100.0  # Convert back to 0.0-1.0
                },
                error_message=None,
                gas_estimate=None,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            error_msg = str(e)
            self.data_logger.log_data_collection(
                "get_risk_score", "error", 0, False, error_msg
            )
            
            return ContractCallResult(
                success=False,
                return_value=None,
                error_message=error_msg,
                gas_estimate=None,
                timestamp=datetime.now()
            )
    
    async def estimate_gas_for_update(self, risk_score: float) -> ContractCallResult:
        """Estimate gas cost for updating risk score"""
        try:
            if not self.is_connected:
                await self.connect()
            
            if not self.is_connected or not self.contract or not self.account:
                raise ValueError("Not connected to blockchain")
            
            # Convert risk score
            score_uint8 = min(100, max(0, int(risk_score * 100)))
            
            # Estimate gas
            gas_estimate = self.contract.functions.updateRisk(score_uint8).estimate_gas({
                'from': self.account.address
            })
            
            return ContractCallResult(
                success=True,
                return_value=gas_estimate,
                error_message=None,
                gas_estimate=gas_estimate,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            error_msg = str(e)
            self.data_logger.log_data_collection(
                "gas_estimation", "error", 0, False, error_msg
            )
            
            return ContractCallResult(
                success=False,
                return_value=None,
                error_message=error_msg,
                gas_estimate=None,
                timestamp=datetime.now()
            )
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get current connection status"""
        try:
            status = {
                "connected": self.is_connected,
                "rpc_url": self.current_rpc_url,
                "account_address": self.account.address if self.account else None,
                "contract_address": self.contract_address,
                "latest_block": None,
                "network_id": None
            }
            
            if self.is_connected and self.w3:
                try:
                    status["latest_block"] = self.w3.eth.block_number
                    status["network_id"] = self.w3.eth.chain_id
                except Exception:
                    pass  # Connection might be unstable
            
            return status
            
        except Exception as e:
            return {
                "connected": False,
                "error": str(e)
            }
    
    async def disconnect(self):
        """Clean up connections"""
        try:
            self.is_connected = False
            self.w3 = None
            self.account = None
            self.contract = None
            
            self.data_logger.log_data_collection(
                "blockchain_disconnect", "success", 1, True, "Disconnected from blockchain"
            )
            
        except Exception as e:
            self.data_logger.log_data_collection(
                "blockchain_disconnect", "error", 0, False, str(e)
            )