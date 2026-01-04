#!/usr/bin/env python3
"""
QIE Blockchain Endpoint Configuration Script
Configures and tests QIE V3 blockchain RPC endpoints
"""

import asyncio
import sys
from pathlib import Path
from web3 import Web3
import json
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

class QIEEndpointConfigurator:
    """Configures and validates QIE blockchain endpoints"""
    
    def __init__(self):
        # QIE V3 blockchain endpoints (update with actual endpoints)
        self.endpoints = {
            "mainnet": {
                "primary": "https://rpc.qie-pulse.com",
                "backup": "https://backup-rpc.qie-pulse.com",
                "websocket": "wss://ws.qie-pulse.com",
                "explorer": "https://explorer.qie-pulse.com"
            },
            "testnet": {
                "primary": "https://testnet-rpc.qie-pulse.com", 
                "backup": "https://testnet-backup-rpc.qie-pulse.com",
                "websocket": "wss://testnet-ws.qie-pulse.com",
                "explorer": "https://testnet-explorer.qie-pulse.com"
            }
        }
    
    async def test_endpoint(self, url: str, timeout: int = 10) -> dict:
        """Test a single RPC endpoint"""
        result = {
            "url": url,
            "connected": False,
            "chain_id": None,
            "latest_block": None,
            "response_time_ms": None,
            "error": None
        }
        
        try:
            start_time = time.time()
            
            # Create Web3 instance
            w3 = Web3(Web3.HTTPProvider(url, request_kwargs={'timeout': timeout}))
            
            # Test connection
            if w3.is_connected():
                result["connected"] = True
                
                # Get chain info
                result["chain_id"] = w3.eth.chain_id
                result["latest_block"] = w3.eth.block_number
                
                # Calculate response time
                result["response_time_ms"] = (time.time() - start_time) * 1000
                
                print(f"✅ {url}")
                print(f"   Chain ID: {result['chain_id']}")
                print(f"   Latest Block: {result['latest_block']}")
                print(f"   Response Time: {result['response_time_ms']:.1f}ms")
            else:
                result["error"] = "Connection failed"
                print(f"❌ {url} - Connection failed")
                
        except Exception as e:
            result["error"] = str(e)
            print(f"❌ {url} - {str(e)}")
        
        return result
    
    async def test_all_endpoints(self, network: str = "mainnet") -> dict:
        """Test all endpoints for a network"""
        print(f"🔍 Testing {network} endpoints...")
        print("-" * 40)
        
        endpoints = self.endpoints.get(network, {})
        results = {}
        
        # Test HTTP RPC endpoints
        for name, url in endpoints.items():
            if name in ["primary", "backup"]:
                results[name] = await self.test_endpoint(url)
        
        return results
    
    def select_best_endpoints(self, results: dict) -> dict:
        """Select the best performing endpoints"""
        best_endpoints = {}
        
        # Find fastest working endpoint as primary
        working_endpoints = [
            (name, result) for name, result in results.items() 
            if result["connected"] and result["response_time_ms"] is not None
        ]
        
        if working_endpoints:
            # Sort by response time
            working_endpoints.sort(key=lambda x: x[1]["response_time_ms"])
            
            best_endpoints["primary"] = working_endpoints[0][1]["url"]
            
            # Use second fastest as backup if available
            if len(working_endpoints) > 1:
                best_endpoints["backup"] = working_endpoints[1][1]["url"]
            else:
                # Use the same endpoint as backup
                best_endpoints["backup"] = working_endpoints[0][1]["url"]
        
        return best_endpoints
    
    def update_env_file(self, endpoints: dict):
        """Update .env file with selected endpoints"""
        try:
            env_file = Path(".env")
            if env_file.exists():
                # Read current content
                with open(env_file, "r") as f:
                    content = f.read()
                
                # Update endpoints
                lines = content.split("\n")
                
                # Update or add QIE RPC URLs
                primary_updated = False
                backup_updated = False
                
                for i, line in enumerate(lines):
                    if line.startswith("QIE_RPC_URL="):
                        lines[i] = f"QIE_RPC_URL={endpoints['primary']}"
                        primary_updated = True
                    elif line.startswith("QIE_RPC_BACKUP_URL="):
                        lines[i] = f"QIE_RPC_BACKUP_URL={endpoints['backup']}"
                        backup_updated = True
                
                if not primary_updated:
                    lines.append(f"QIE_RPC_URL={endpoints['primary']}")
                if not backup_updated:
                    lines.append(f"QIE_RPC_BACKUP_URL={endpoints['backup']}")
                
                # Write back
                with open(env_file, "w") as f:
                    f.write("\n".join(lines))
                
                print(f"✅ Updated .env file with QIE endpoints")
            
        except Exception as e:
            print(f"⚠️  Failed to update .env file: {e}")
    
    def save_endpoint_config(self, network: str, results: dict, best_endpoints: dict):
        """Save endpoint configuration and test results"""
        config = {
            "network": network,
            "test_timestamp": str(datetime.now()),
            "test_results": results,
            "selected_endpoints": best_endpoints,
            "all_endpoints": self.endpoints[network]
        }
        
        config_file = Path(f"qie_{network}_endpoints.json")
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)
        
        print(f"💾 Endpoint configuration saved to {config_file}")

async def main():
    """Main configuration function"""
    print("🚀 QIE V3 Blockchain Endpoint Configuration")
    print("=" * 50)
    
    try:
        from datetime import datetime
        
        configurator = QIEEndpointConfigurator()
        
        # Test mainnet endpoints
        print("🌐 Testing QIE V3 Mainnet Endpoints...")
        mainnet_results = await configurator.test_all_endpoints("mainnet")
        
        print("\n🧪 Testing QIE V3 Testnet Endpoints...")
        testnet_results = await configurator.test_all_endpoints("testnet")
        
        # Select best endpoints for mainnet
        best_mainnet = configurator.select_best_endpoints(mainnet_results)
        
        if best_mainnet:
            print(f"\n🎯 Selected Mainnet Endpoints:")
            print(f"   Primary: {best_mainnet['primary']}")
            print(f"   Backup: {best_mainnet['backup']}")
            
            # Update configuration
            configurator.update_env_file(best_mainnet)
            configurator.save_endpoint_config("mainnet", mainnet_results, best_mainnet)
        else:
            print("\n❌ No working mainnet endpoints found!")
            
            # Try testnet as fallback
            best_testnet = configurator.select_best_endpoints(testnet_results)
            if best_testnet:
                print(f"🔄 Using testnet endpoints as fallback:")
                print(f"   Primary: {best_testnet['primary']}")
                print(f"   Backup: {best_testnet['backup']}")
                
                configurator.update_env_file(best_testnet)
                configurator.save_endpoint_config("testnet", testnet_results, best_testnet)
        
        print("\n" + "=" * 50)
        print("🎉 ENDPOINT CONFIGURATION COMPLETE!")
        print("\n🔧 Next Steps:")
        print("1. Set up production wallet")
        print("2. Deploy AI Risk Oracle smart contract")
        print("3. Test blockchain integration")
        print("4. Start live risk monitoring")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Configuration failed: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Check internet connectivity")
        print("2. Verify QIE blockchain is operational")
        print("3. Contact QIE team for correct RPC endpoints")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)