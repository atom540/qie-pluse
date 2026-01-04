# AI Risk Oracle - Production Deployment Guide

## 🚀 Quick Start Deployment

Follow these steps to deploy the AI Risk Oracle to QIE V3 blockchain:

### Step 1: Configure QIE Blockchain Endpoints
```bash
python configure_qie_endpoints.py
```
This will:
- Test QIE V3 mainnet and testnet RPC endpoints
- Select the fastest working endpoints
- Update your `.env` file automatically
- Save endpoint configuration for future reference

### Step 2: Set Up Production Wallet
```bash
python setup_production_wallet.py
```
This will:
- Generate a cryptographically secure wallet
- Update your `.env` file with the private key
- Display the wallet address for funding
- Save wallet configuration securely

**⚠️ IMPORTANT**: Fund the generated wallet address with QIE tokens for gas fees!

### Step 3: Deploy Smart Contract
```bash
python deploy_smart_contract.py
```
This will:
- Deploy the AI Risk Oracle contract to QIE blockchain
- Set the deployer as the authorized updater
- Update your `.env` file with the contract address
- Save deployment information

### Step 4: Start Live Risk Monitoring
```bash
python start_live_monitoring.py
```
This will:
- Initialize the AI Risk Oracle inference service
- Connect to the deployed smart contract
- Begin real-time risk monitoring and blockchain updates
- Generate alerts for high-risk conditions

## 📋 Prerequisites

### System Requirements
- Python 3.10+ with conda environment
- Trained AI Risk Oracle model (run `python train_model.py` first)
- Internet connection for blockchain and API access
- QIE tokens for transaction gas fees

### API Keys Required
Ensure these are configured in your `.env` file:
- `COINGECKO_API_KEY`: For crypto price data
- `ALPHA_VANTAGE_API_KEY`: For traditional asset data
- `TELEGRAM_API_ID`, `TELEGRAM_API_HASH`: For sentiment analysis (optional)

### Environment Variables
The deployment scripts will automatically configure:
- `QIE_RPC_URL`: Primary QIE blockchain RPC endpoint
- `QIE_RPC_BACKUP_URL`: Backup RPC endpoint
- `PRIVATE_KEY`: Wallet private key for transactions
- `AI_RISK_ORACLE_CONTRACT_ADDRESS`: Deployed contract address

## 🔧 Manual Configuration

If you need to manually configure the system:

### 1. QIE Blockchain Configuration
```bash
# Update .env file
QIE_RPC_URL=https://rpc.qie-pulse.com
QIE_RPC_BACKUP_URL=https://backup-rpc.qie-pulse.com
```

### 2. Wallet Configuration
```bash
# Generate wallet manually
from eth_account import Account
import secrets

private_key = "0x" + secrets.token_hex(32)
account = Account.from_key(private_key)
print(f"Address: {account.address}")
print(f"Private Key: {private_key}")
```

### 3. Contract Deployment
```bash
# Deploy using Hardhat (alternative method)
npx hardhat compile
npx hardhat deploy --network qie-mainnet
```

## 🔍 Monitoring and Maintenance

### Health Checks
```bash
# Check service health
curl http://localhost:8001/health

# Check blockchain status
curl http://localhost:8001/blockchain/status

# Get current risk scores
curl http://localhost:8001/risk
```

### Log Monitoring
```bash
# Monitor risk alerts
tail -f risk_alerts.jsonl

# Monitor service logs
tail -f logs/data_collection.log
```

### Performance Monitoring
- **Target Latency**: < 3 seconds for blockchain updates
- **Uptime Goal**: 99.9% during market hours
- **Alert Response**: < 60 seconds for critical failures

## 🛡️ Security Best Practices

### Private Key Security
- Store private keys in secure environment variables
- Never commit private keys to version control
- Consider using hardware wallets for production
- Implement key rotation policies

### Network Security
- Use HTTPS endpoints only
- Implement rate limiting
- Monitor for unusual transaction patterns
- Set up alerting for failed transactions

### Access Control
- Restrict contract updater permissions
- Implement multi-signature for critical operations
- Monitor contract events and transactions
- Set up emergency pause mechanisms

## 🚨 Troubleshooting

### Common Issues

**Connection Failed**
```bash
# Check RPC endpoint
curl -X POST -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}' \
  https://rpc.qie-pulse.com
```

**Insufficient Balance**
```bash
# Check wallet balance
python -c "
from web3 import Web3
w3 = Web3(Web3.HTTPProvider('https://rpc.qie-pulse.com'))
balance = w3.eth.get_balance('YOUR_WALLET_ADDRESS')
print(f'Balance: {w3.from_wei(balance, \"ether\")} ETH')
"
```

**Contract Not Found**
- Verify contract address in `.env` file
- Check if contract was deployed successfully
- Ensure you're connected to the correct network

**Model Loading Failed**
```bash
# Retrain model if needed
python train_model.py --training-months 11 --extended-training
```

### Error Codes
- `503`: Service not initialized (check configuration)
- `400`: Invalid parameters (check input validation)
- `500`: Internal server error (check logs)

## 📊 Performance Optimization

### Gas Optimization
```python
# Adjust gas settings in .env
DEFAULT_GAS_LIMIT=100000
DEFAULT_GAS_PRICE=20000000000  # 20 Gwei
```

### Caching Configuration
```python
# Adjust cache TTL for better performance
CACHE_TTL_SECONDS=60  # 1 minute cache
```

### Monitoring Intervals
```python
# Adjust monitoring frequency
MONITORING_INTERVAL=30  # 30 seconds
RISK_THRESHOLD=0.85     # 85% risk threshold
```

## 🔄 Backup and Recovery

### Data Backup
- Backup wallet private keys securely
- Save deployment configuration files
- Archive model training data
- Export transaction history

### Disaster Recovery
1. Redeploy smart contract if needed
2. Restore wallet from private key
3. Retrain model from historical data
4. Resume monitoring service

## 📈 Scaling Considerations

### High Availability
- Deploy multiple inference service instances
- Use load balancers for API endpoints
- Implement database clustering for logs
- Set up monitoring and alerting

### Performance Scaling
- Optimize model inference speed
- Implement connection pooling
- Use caching for frequently accessed data
- Consider GPU acceleration for ML inference

## 🎯 Success Metrics

### Operational Metrics
- **Uptime**: > 99.9%
- **Response Time**: < 500ms for risk assessment
- **Blockchain Latency**: < 3 seconds for updates
- **Alert Accuracy**: > 90% true positive rate

### Business Metrics
- **Risk Detection**: Early warning for market crashes
- **Portfolio Protection**: Automated hedge triggering
- **Cost Efficiency**: Optimized gas usage
- **User Satisfaction**: Reliable risk signals

## 🏆 Production Checklist

- [ ] QIE blockchain endpoints configured and tested
- [ ] Production wallet created and funded
- [ ] AI Risk Oracle contract deployed successfully
- [ ] Model trained and loaded
- [ ] API endpoints tested and functional
- [ ] Monitoring service running
- [ ] Alerts configured and tested
- [ ] Security measures implemented
- [ ] Backup procedures established
- [ ] Documentation updated

## 📞 Support

For deployment support:
1. Check the troubleshooting section above
2. Review log files for error details
3. Test individual components separately
4. Contact QIE team for blockchain-specific issues

---

🎉 **Congratulations!** Your AI Risk Oracle is now ready for production deployment on QIE V3!

The system will provide real-time risk assessment and autonomous portfolio protection for the QIE ecosystem.