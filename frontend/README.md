# QIE V3 AI Risk Oracle Dashboard

A modern, responsive web dashboard for the QIE V3 AI Risk Oracle system.

## Features

- **Real-time Risk Assessment**: Live monitoring of multi-asset risk scores
- **Interactive Charts**: Visual representation of risk data and trends
- **System Status**: Comprehensive health monitoring and performance metrics
- **Blockchain Integration**: QIE V3 blockchain connection status and transaction history
- **Alert System**: Real-time risk alerts with recommended actions
- **Performance Metrics**: Detailed system performance tracking

## Quick Start

### Prerequisites

- Node.js 16+ and npm
- QIE V3 AI Risk Oracle backend running on port 8001

### Installation

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm start
```

The dashboard will be available at `http://localhost:3000`

### Production Build

```bash
# Create production build
npm run build

# Serve production build (optional)
npx serve -s build -l 3000
```

## Dashboard Sections

### 1. Risk Dashboard
- Multi-asset risk monitoring (BTC, ETH, XRP, SOL, BNB, GOLD)
- Combined risk scores with ML and sentiment analysis
- Real-time risk level indicators (LOW, MEDIUM, HIGH, CRITICAL)
- Interactive charts and visualizations

### 2. Alerts Panel
- Current risk alerts with severity levels
- Recommended actions based on risk levels
- Alert history and timeline
- Asset-specific risk notifications

### 3. Performance Metrics
- System performance monitoring
- Response time tracking
- Model accuracy metrics
- Historical performance trends

### 4. Blockchain Status
- QIE V3 blockchain connection status
- Smart contract interaction monitoring
- Transaction history and success rates
- On-chain risk score updates

### 5. System Status
- Overall system health monitoring
- Component status tracking
- Resource usage metrics
- Service uptime and availability

## API Integration

The dashboard connects to the QIE V3 AI Risk Oracle backend API:

- **Base URL**: `http://localhost:8001`
- **Health Check**: `GET /health`
- **Risk Assessment**: `GET /risk`
- **Alerts**: `GET /alerts`
- **System Status**: `GET /status`
- **Blockchain Status**: `GET /blockchain/status`

## Technology Stack

- **React 18**: Modern React with hooks
- **Tailwind CSS**: Utility-first CSS framework
- **Recharts**: Responsive chart library
- **Lucide React**: Beautiful icon library
- **Axios**: HTTP client for API requests

## Customization

### Colors and Themes

The dashboard uses a dark theme optimized for financial data visualization:

- **Primary**: Blue (#3B82F6)
- **Success**: Green (#10B981)
- **Warning**: Yellow (#F59E0B)
- **Danger**: Red (#EF4444)
- **Background**: Slate gradients

### Risk Level Colors

- **LOW**: Green (#10B981)
- **MEDIUM**: Yellow (#F59E0B)
- **HIGH**: Orange (#F97316)
- **CRITICAL**: Red (#EF4444) with pulse animation

## Development

### Project Structure

```
frontend/
├── public/
│   └── index.html
├── src/
│   ├── components/
│   │   ├── RiskDashboard.js
│   │   ├── AlertsPanel.js
│   │   ├── SystemStatus.js
│   │   ├── PerformanceMetrics.js
│   │   └── BlockchainStatus.js
│   ├── App.js
│   ├── index.js
│   └── index.css
├── package.json
└── tailwind.config.js
```

### Adding New Components

1. Create component in `src/components/`
2. Import and add to `App.js`
3. Add navigation tab if needed
4. Update API integration as required

## Deployment

### Development
```bash
npm start
```

### Production
```bash
npm run build
npx serve -s build
```

### Docker (Optional)
```dockerfile
FROM node:16-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npx", "serve", "-s", "build", "-l", "3000"]
```

## Hackathon Demo

For the QIE V3 hackathon presentation:

1. **Start Backend**: Ensure the AI Risk Oracle API is running on port 8001
2. **Start Frontend**: Run `npm start` to launch the dashboard
3. **Demo Flow**:
   - Show real-time risk assessment
   - Demonstrate alert system
   - Display blockchain integration
   - Highlight performance metrics

## Support

For issues or questions:
1. Check the backend API is running and accessible
2. Verify network connectivity
3. Check browser console for errors
4. Ensure all dependencies are installed

---

**Built for QIE V3 Hackathon - Predictive Security System for Money**