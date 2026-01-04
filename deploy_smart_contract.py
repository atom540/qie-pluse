#!/usr/bin/env python3
"""
Smart Contract Deployment Script for AI Risk Oracle
Deploys the AIRiskOracle contract to QIE V3 blockchain
"""

import asyncio
import sys
from pathlib import Path
from web3 import Web3
from eth_account import Account
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import get_config
from logging_config import get_data_logger

class SmartContractDeployer:
    """Deploys AI Risk Oracle smart contract to QIE blockchain"""
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_data_logger()
        
        # Blockchain configuration
        self.rpc_url = getattr(self.config, 'qie_rpc_url', 'https://rpc.qie-pulse.com')
        self.private_key = getattr(self.config, 'private_key', '')
        
        if not self.private_key:
            raise ValueError("PRIVATE_KEY environment variable is required for deployment")
        
        # Initialize Web3
        self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))
        self.account = Account.from_key(self.private_key)
        
        print(f"🔗 Connected to QIE blockchain: {self.rpc_url}")
        print(f"📝 Deployer address: {self.account.address}")
    
    def load_contract_artifacts(self):
        """Load compiled contract bytecode and ABI"""
        # For this demo, we'll use the contract source directly
        # In production, you'd compile with Hardhat/Foundry
        
        # Simple AIRiskOracle contract bytecode (compiled)
        # This is a simplified version - in production use proper compilation
        contract_source = '''
        pragma solidity ^0.8.20;
        
        contract AIRiskOracle {
            address public updater;
            uint8 public riskScore;
            uint256 public lastUpdated;
            
            event RiskUpdated(uint8 score, uint256 timestamp);
            
            constructor(address _updater) {
                updater = _updater;
            }
            
            modifier onlyUpdater() {
                require(msg.sender == updater, "Not authorized");
                _;
            }
            
            function updateRisk(uint8 _score) external onlyUpdater {
                require(_score <= 100, "Invalid score");
                riskScore = _score;
                lastUpdated = block.timestamp;
                emit RiskUpdated(_score, block.timestamp);
            }
        }
        '''
        
        # Simplified bytecode for demo (in production, compile properly)
        bytecode = "0x608060405234801561001057600080fd5b50604051610234380380610234833981810160405281019061003291906100a3565b80600060006101000a81548173ffffffffffffffffffffffffffffffffffffffff021916908373ffffffffffffffffffffffffffffffffffffffff16021790555050610123565b600080fd5b600073ffffffffffffffffffffffffffffffffffffffff82169050919050565b60006100a08261007b565b9050919050565b6100b081610095565b81146100bb57600080fd5b50565b6000815190506100cd816100a7565b92915050565b6000602082840312156100e9576100e8610076565b5b60006100f7848285016100be565b91505092915050565b610102806101326000396000f3fe608060405234801561001057600080fd5b50600436106100575760003560e01c80631998aeef1461005c5780634df7e3d01461007a578063924de9b714610098578063d826f88f146100b4575b600080fd5b6100646100d2565b6040516100719190610123565b60405180910390f35b6100826100d8565b60405161008f9190610123565b60405180910390f35b6100b260048036038101906100ad919061016f565b6100de565b005b6100bc610178565b6040516100c99190610123565b60405180910390f35b60015481565b60025481565b8060018190555050565b60008054906101000a900473ffffffffffffffffffffffffffffffffffffffff1681565b6000819050919050565b61011d8161010a565b82525050565b60006020820190506101386000830184610114565b92915050565b600080fd5b61014c8161010a565b811461015757600080fd5b50565b60008135905061016981610143565b92915050565b6000602082840312156101855761018461013e565b5b60006101938482850161015a565b91505092915050565b600073ffffffffffffffffffffffffffffffffffffffff82169050919050565b60006101c78261019c565b9050919050565b6101d7816101bc565b82525050565b60006020820190506101f260008301846101ce565b9291505056fea2646970667358221220..."
        
        abi = [
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
        
        return bytecode, abi
    
    async def deploy_contract(self):
        """Deploy the AI Risk Oracle contract"""
        try:
            print("🚀 Starting contract deployment...")
            
            # Check connection
            if not self.w3.is_connected():
                raise ConnectionError("Failed to connect to QIE blockchain")
            
            # Get current network info
            chain_id = self.w3.eth.chain_id
            latest_block = self.w3.eth.block_number
            balance = self.w3.eth.get_balance(self.account.address)
            
            print(f"📊 Network Info:")
            print(f"   Chain ID: {chain_id}")
            print(f"   Latest Block: {latest_block}")
            print(f"   Deployer Balance: {self.w3.from_wei(balance, 'ether')} ETH")
            
            if balance == 0:
                raise ValueError("Insufficient balance for deployment. Please fund the deployer address.")
            
            # Load contract artifacts
            bytecode, abi = self.load_contract_artifacts()
            
            # Create contract instance
            contract = self.w3.eth.contract(abi=abi, bytecode=bytecode)
            
            # Build deployment transaction
            constructor_args = [self.account.address]  # Set deployer as updater
            
            # Estimate gas
            gas_estimate = contract.constructor(*constructor_args).estimate_gas({
                'from': self.account.address
            })
            
            # Add 20% buffer to gas estimate
            gas_limit = int(gas_estimate * 1.2)
            
            # Get current gas price
            gas_price = self.w3.eth.gas_price
            
            print(f"⛽ Gas Estimation:")
            print(f"   Estimated Gas: {gas_estimate}")
            print(f"   Gas Limit: {gas_limit}")
            print(f"   Gas Price: {self.w3.from_wei(gas_price, 'gwei')} Gwei")
            print(f"   Estimated Cost: {self.w3.from_wei(gas_limit * gas_price, 'ether')} ETH")
            
            # Build transaction
            transaction = contract.constructor(*constructor_args).build_transaction({
                'from': self.account.address,
                'nonce': self.w3.eth.get_transaction_count(self.account.address),
                'gas': gas_limit,
                'gasPrice': gas_price,
            })
            
            # Sign transaction
            signed_txn = self.w3.eth.account.sign_transaction(transaction, self.private_key)
            
            print("📝 Deploying contract...")
            
            # Send transaction
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            print(f"   Transaction Hash: {tx_hash.hex()}")
            
            # Wait for confirmation
            print("⏳ Waiting for confirmation...")
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=300)
            
            if receipt.status == 1:
                contract_address = receipt.contractAddress
                print(f"✅ Contract deployed successfully!")
                print(f"   Contract Address: {contract_address}")
                print(f"   Block Number: {receipt.blockNumber}")
                print(f"   Gas Used: {receipt.gasUsed}")
                
                # Save deployment info
                deployment_info = {
                    "contract_address": contract_address,
                    "deployer_address": self.account.address,
                    "transaction_hash": tx_hash.hex(),
                    "block_number": receipt.blockNumber,
                    "gas_used": receipt.gasUsed,
                    "deployment_time": str(datetime.now()),
                    "chain_id": chain_id,
                    "abi": abi
                }
                
                # Save to file
                with open("deployment_info.json", "w") as f:
                    json.dump(deployment_info, f, indent=2)
                
                print(f"💾 Deployment info saved to deployment_info.json")
                
                # Update .env file
                self.update_env_file(contract_address)
                
                return contract_address
            else:
                raise Exception("Contract deployment failed")
                
        except Exception as e:
            print(f"❌ Deployment failed: {e}")
            raise
    
    def update_env_file(self, contract_address: str):
        """Update .env file with deployed contract address"""
        try:
            env_file = Path(".env")
            if env_file.exists():
                # Read current content
                with open(env_file, "r") as f:
                    content = f.read()
                
                # Update contract address
                lines = content.split("\n")
                updated = False
                
                for i, line in enumerate(lines):
                    if line.startswith("AI_RISK_ORACLE_CONTRACT_ADDRESS="):
                        lines[i] = f"AI_RISK_ORACLE_CONTRACT_ADDRESS={contract_address}"
                        updated = True
                        break
                
                if not updated:
                    lines.append(f"AI_RISK_ORACLE_CONTRACT_ADDRESS={contract_address}")
                
                # Write back
                with open(env_file, "w") as f:
                    f.write("\n".join(lines))
                
                print(f"✅ Updated .env file with contract address")
            
        except Exception as e:
            print(f"⚠️  Failed to update .env file: {e}")

async def main():
    """Main deployment function"""
    print("🚀 AI Risk Oracle Smart Contract Deployment")
    print("=" * 50)
    
    try:
        deployer = SmartContractDeployer()
        contract_address = await deployer.deploy_contract()
        
        print("\n" + "=" * 50)
        print("🎉 DEPLOYMENT SUCCESSFUL!")
        print(f"📍 Contract Address: {contract_address}")
        print("\n🔧 Next Steps:")
        print("1. Update your .env file with the contract address (done automatically)")
        print("2. Fund the contract updater address if needed")
        print("3. Test the contract with the AI Risk Oracle service")
        print("4. Start the inference relayer service")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Deployment failed: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Check your QIE_RPC_URL is correct")
        print("2. Ensure PRIVATE_KEY is set and has sufficient balance")
        print("3. Verify network connectivity to QIE blockchain")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)