# API Keys Setup Guide

This guide walks you through obtaining all the API keys needed for the AI Risk Oracle system.

## 🔑 Required API Keys

### 1. CoinGecko API (Crypto Price Data)

**Purpose**: Historical and real-time cryptocurrency prices for BTC, ETH, XRP, SOL, BNB

**Steps**:
1. Go to [CoinGecko API](https://www.coingecko.com/en/api)
2. Click "Get Your API Key"
3. Sign up for a free account
4. Choose the "Demo" plan (free, 30 calls/minute)
5. Copy your API key from the dashboard

**Free Tier**: 30 calls/minute, sufficient for historical data collection

```env
COINGECKO_API_KEY=CG-your-api-key-here
```

### 2. Alpha Vantage API (Traditional Assets - Gold)

**Purpose**: Historical Gold prices and traditional asset data

**Steps**:
1. Go to [Alpha Vantage](https://www.alphavantage.co/support/#api-key)
2. Enter your email and click "GET FREE API KEY"
3. Check your email for the API key
4. Free tier: 25 requests per day

**Free Tier**: 25 requests/day, sufficient for daily Gold price updates

```env
ALPHA_VANTAGE_API_KEY=your-alpha-vantage-key-here
```

### 3. Binance API (High-Frequency Crypto Data)

**Purpose**: Minute-level price data and technical indicators

**Steps**:
1. Go to [Binance](https://www.binance.com/) and create an account
2. Complete identity verification (required for API access)
3. Go to Account → API Management
4. Create a new API key
5. **Important**: Only enable "Enable Reading" (no trading permissions needed)
6. Save both API Key and Secret Key

**Security**: Only enable read permissions, never enable trading for this application

```env
BINANCE_API_KEY=your-binance-api-key-here
BINANCE_SECRET_KEY=your-binance-secret-key-here
```

### 4. Twitter API (Social Sentiment Data)

**Purpose**: Historical tweets during market events for sentiment analysis

**Steps**:
1. Go to [Twitter Developer Portal](https://developer.twitter.com/)
2. Apply for a developer account (may take 1-2 days for approval)
3. Create a new project/app
4. In your app settings, go to "Keys and tokens" section
5. Generate/regenerate all tokens:
   - **API Key and Secret** (OAuth 1.0a)
   - **Access Token and Secret** (OAuth 1.0a)
   - **Bearer Token** (OAuth 2.0 - optional)

**Important OAuth 1.0a Setup**:
- Make sure your app has "Read" permissions
- Generate Access Token and Secret (not just API keys)
- We need OAuth 1.0a for user context (not just app-only Bearer token)

**Academic Research Access** (for historical data):
- Apply for Academic Research product track
- Provides access to full Twitter archive
- Required for historical market event analysis

```env
# OAuth 1.0a credentials (required)
TWITTER_API_KEY=your-api-key-here
TWITTER_API_SECRET=your-api-secret-here
TWITTER_ACCESS_TOKEN=your-access-token-here
TWITTER_ACCESS_TOKEN_SECRET=your-access-token-secret-here

# OAuth 2.0 Bearer Token (optional, for some endpoints)
TWITTER_BEARER_TOKEN=your-bearer-token-here
```

### 5. Telegram API (QIE Community Sentiment)

**Purpose**: Real-time sentiment analysis from QIE Telegram channels

**Steps**:
1. Go to [my.telegram.org](https://my.telegram.org/)
2. Log in with your phone number
3. Go to "API development tools"
4. Create a new application
5. Note down API ID and API Hash
6. You'll need your phone number for authentication

**Important**: You need to be a member of QIE Telegram channels to access messages

```env
TELEGRAM_API_ID=12345678
TELEGRAM_API_HASH=your-telegram-api-hash-here
TELEGRAM_PHONE_NUMBER=+1234567890
```

### 6. Reddit API (Cryptocurrency Community Data)

**Purpose**: Historical posts from r/cryptocurrency during market events

**Steps**:
1. Go to [Reddit Apps](https://www.reddit.com/prefs/apps)
2. Click "Create App" or "Create Another App"
3. Choose "script" as the app type
4. Fill in name and description
5. Note down the client ID (under the app name) and client secret

```env
REDDIT_CLIENT_ID=your-reddit-client-id-here
REDDIT_CLIENT_SECRET=your-reddit-client-secret-here
REDDIT_USER_AGENT=ai-risk-oracle/1.0
```

## 🔐 Blockchain Configuration

### QIE Blockchain Setup

**Purpose**: Sending risk scores to the AI_Risk_Oracle smart contract

**Steps**:
1. **Get QIE RPC URLs**: Contact QIE team for official RPC endpoints
2. **Deploy/Get Contract Address**: Get the AI_Risk_Oracle contract address
3. **Create Wallet**: Generate a new wallet specifically for the oracle
4. **Fund Wallet**: Add QIE tokens for gas fees
5. **Export Private Key**: Use for blockchain transactions

**Security Warning**: 
- Create a dedicated wallet for the oracle
- Only fund with minimal amounts needed for gas
- Never share or commit private keys to version control

```env
QIE_RPC_URL=https://rpc.qie-pulse.com
QIE_RPC_BACKUP_URL=https://backup-rpc.qie-pulse.com
AI_RISK_ORACLE_CONTRACT_ADDRESS=0x1234567890123456789012345678901234567890
PRIVATE_KEY=0xYOUR_PRIVATE_KEY_HERE
```

## 📋 Quick Setup Checklist

- [ ] **CoinGecko**: Free account, demo plan
- [ ] **Alpha Vantage**: Free API key via email
- [ ] **Binance**: Account + API key (read-only permissions)
- [ ] **Twitter**: Developer account + API keys
- [ ] **Telegram**: API ID/Hash from my.telegram.org
- [ ] **Reddit**: App registration for client ID/secret
- [ ] **QIE Blockchain**: RPC URLs + contract address + funded wallet

## 🚀 Testing Your Setup

After configuring all API keys, test the setup:

```bash
# Activate environment
conda activate ai-risk-oracle

# Test configuration loading
python -c "from config import get_config; config = get_config(); print('✓ Configuration loaded successfully')"

# Test API connections (once implemented)
python -c "from services.collectors import test_apis; test_apis()"
```

## 💡 Development Tips

### Free Tier Limitations
- **CoinGecko**: 30 calls/minute - batch requests efficiently
- **Alpha Vantage**: 25 calls/day - cache Gold prices daily
- **Binance**: No explicit limits for market data
- **Twitter**: Rate limits vary by endpoint
- **Telegram**: No explicit limits for reading
- **Reddit**: 60 requests/minute

### Cost Optimization
- Cache frequently requested data
- Use batch API calls when available
- Implement exponential backoff for rate limiting
- Consider upgrading to paid tiers for production use

### Security Best Practices
- Never commit API keys to version control
- Use environment variables for all sensitive data
- Rotate API keys regularly
- Monitor API usage for unusual activity
- Use read-only permissions where possible

## 🆘 Troubleshooting

### Common Issues

**"Invalid API Key" Errors**:
- Double-check key format and spelling
- Ensure no extra spaces or characters
- Verify key permissions and restrictions

**Rate Limiting**:
- Implement exponential backoff
- Cache responses to reduce API calls
- Consider upgrading to higher tier plans

**Telegram Authentication**:
- Ensure phone number format is correct (+country_code)
- You may need to complete 2FA verification
- Join QIE channels before attempting to read messages

**Blockchain Connection**:
- Verify RPC URLs are accessible
- Ensure wallet has sufficient gas funds
- Test with small transactions first

## 📞 Support

If you encounter issues:
1. Check API provider documentation
2. Verify account status and limits
3. Test with simple API calls first
4. Contact API provider support if needed

For QIE-specific configuration, contact the QIE development team for:
- Official RPC endpoints
- Contract addresses
- Community channel access