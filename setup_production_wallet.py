#!/usr/bin/env python3
"""
Production Wallet Setup Script
Creates a dedicated wallet for AI Risk Oracle transactions
"""

import secrets
import sys
from pathlib import Path
from eth_account import Account
from web3 import Web3
import json

def generate_secure_wallet():
    """Generate a cryptographically secure wallet"""
    print("🔐 Generating secure wallet for AI Risk Oracle...")
    
    # Generate secure private key
    private_key = "0x" + secrets.token_hex(32)
    
    # Create account
    account = Account.from_key(private_key)
    
    wallet_info = {
        "address": account.address,
        "private_key": private_key,
        "created_at": str(datetime.now()),
        "purpose": "AI Risk Oracle Transaction Wallet"
    }
    
    print(f"✅ Wallet generated successfully!")
    print(f"📍 Address: {account.address}")
    print(f"🔑 Private Key: {private_key}")
    print()
    print("⚠️  SECURITY WARNING:")
    print("- Store the private key securely")
    print("- Never share or commit the private key to version control")
    print("- Consider using a hardware wallet for production")
    print()
    
    return wallet_info

def save_wallet_config(wallet_info):
    """Save wallet configuration securely"""
    # Save to secure config file (not in git)
    config_file = Path("wallet_config.json")
    
    with open(config_file, "w") as f:
        json.dump(wallet_info, f, indent=2)
    
    print(f"💾 Wallet config saved to {config_file}")
    print("🔒 Make sure to add wallet_config.json to .gitignore")
    
    # Update .env file
    update_env_file(wallet_info["private_key"])

def update_env_file(private_key):
    """Update .env file with new private key"""
    try:
        env_file = Path(".env")
        if env_file.exists():
            # Read current content
            with open(env_file, "r") as f:
                content = f.read()
            
            # Update private key
            lines = content.split("\n")
            updated = False
            
            for i, line in enumerate(lines):
                if line.startswith("PRIVATE_KEY="):
                    lines[i] = f"PRIVATE_KEY={private_key}"
                    updated = True
                    break
            
            if not updated:
                lines.append(f"PRIVATE_KEY={private_key}")
            
            # Write back
            with open(env_file, "w") as f:
                f.write("\n".join(lines))
            
            print(f"✅ Updated .env file with new private key")
        
    except Exception as e:
        print(f"⚠️  Failed to update .env file: {e}")

def check_wallet_balance(address, rpc_url):
    """Check wallet balance on QIE blockchain"""
    try:
        w3 = Web3(Web3.HTTPProvider(rpc_url))
        
        if not w3.is_connected():
            print(f"⚠️  Could not connect to {rpc_url}")
            return
        
        balance = w3.eth.get_balance(address)
        balance_eth = w3.from_wei(balance, 'ether')
        
        print(f"💰 Wallet Balance: {balance_eth} ETH")
        
        if balance == 0:
            print("⚠️  Wallet has no balance. Please fund it before deployment.")
            print(f"   Send QIE tokens to: {address}")
        else:
            print("✅ Wallet has sufficient balance for transactions")
            
    except Exception as e:
        print(f"⚠️  Could not check balance: {e}")

def main():
    """Main wallet setup function"""
    print("🚀 AI Risk Oracle Production Wallet Setup")
    print("=" * 50)
    
    try:
        from datetime import datetime
        
        # Generate wallet
        wallet_info = generate_secure_wallet()
        
        # Save configuration
        save_wallet_config(wallet_info)
        
        # Check balance (if RPC is available)
        rpc_url = "https://rpc.qie-pulse.com"  # Default QIE RPC
        check_wallet_balance(wallet_info["address"], rpc_url)
        
        print("\n" + "=" * 50)
        print("🎉 WALLET SETUP COMPLETE!")
        print(f"📍 Address: {wallet_info['address']}")
        print("\n🔧 Next Steps:")
        print("1. Fund the wallet with QIE tokens for gas fees")
        print("2. Deploy the AI Risk Oracle smart contract")
        print("3. Test the blockchain integration")
        print("4. Start the production inference service")
        print("\n🔒 Security Reminders:")
        print("- Add wallet_config.json to .gitignore")
        print("- Store private key in secure environment variables")
        print("- Consider using hardware wallet for production")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Wallet setup failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)