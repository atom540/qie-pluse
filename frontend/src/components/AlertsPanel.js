import React, { useState, useEffect } from 'react';
import { AlertTriangle, Shield, Clock, TrendingUp, RefreshCw } from 'lucide-react';

const AlertsPanel = () => {
  const [alerts, setAlerts] = useState(null);
  const [loading, setLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState(new Date());

  useEffect(() => {
    const fetchAlerts = async () => {
      try {
        const response = await fetch('/alerts');
        const data = await response.json();
        setAlerts(data);
        setLastUpdate(new Date());
      } catch (error) {
        console.error('Failed to fetch alerts:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchAlerts();
    const interval = setInterval(fetchAlerts, 15000); // Update every 15 seconds

    return () => clearInterval(interval);
  }, []);

  const getAlertColor = (level) => {
    switch (level) {
      case 'CRITICAL': return 'text-red-400 bg-red-400/10 border-red-400/20';
      case 'HIGH': return 'text-orange-400 bg-orange-400/10 border-orange-400/20';
      case 'MEDIUM': return 'text-yellow-400 bg-yellow-400/10 border-yellow-400/20';
      default: return 'text-green-400 bg-green-400/10 border-green-400/20';
    }
  };

  const getAlertIcon = (level) => {
    switch (level) {
      case 'CRITICAL': return <AlertTriangle className="w-6 h-6 animate-pulse" />;
      case 'HIGH': return <AlertTriangle className="w-6 h-6" />;
      case 'MEDIUM': return <TrendingUp className="w-6 h-6" />;
      default: return <Shield className="w-6 h-6" />;
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-400"></div>
        <span className="ml-4 text-slate-400">Loading alerts...</span>
      </div>
    );
  }

  if (!alerts) {
    return (
      <div className="text-center py-12">
        <AlertTriangle className="w-12 h-12 text-yellow-400 mx-auto mb-4" />
        <h3 className="text-lg font-medium text-white mb-2">Unable to load alerts</h3>
        <p className="text-slate-400">Please check the API connection</p>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Alert Summary */}
      <div className={`risk-card ${getAlertColor(alerts.alert_level)}`}>
        <div className="flex items-center space-x-4">
          <div className={getAlertColor(alerts.alert_level)}>
            {getAlertIcon(alerts.alert_level)}
          </div>
          <div className="flex-1">
            <h2 className="text-xl font-bold text-white mb-2">
              {alerts.alert_level} Risk Alert
            </h2>
            <p className="text-slate-300 mb-4">
              {alerts.recommended_action}
            </p>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <p className="text-sm text-slate-400">Max Risk Score</p>
                <p className="text-lg font-bold text-white">
                  {(alerts.max_risk_score * 100).toFixed(1)}%
                </p>
              </div>
              <div>
                <p className="text-sm text-slate-400">Triggered Assets</p>
                <p className="text-lg font-bold text-white">
                  {alerts.triggered_assets.length}
                </p>
              </div>
              <div>
                <p className="text-sm text-slate-400">Alert Time</p>
                <p className="text-lg font-bold text-white">
                  {new Date(alerts.timestamp).toLocaleTimeString()}
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Triggered Assets */}
      {alerts.triggered_assets.length > 0 && (
        <div className="risk-card">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
            <AlertTriangle className="w-5 h-5 mr-2 text-red-400" />
            High-Risk Assets
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {alerts.triggered_assets.map((asset) => (
              <div key={asset} className="bg-slate-700/50 rounded-lg p-4 border border-red-400/20">
                <div className="flex items-center justify-between">
                  <span className="font-medium text-white">{asset}</span>
                  <span className="px-2 py-1 bg-red-500/20 text-red-400 rounded text-xs font-medium">
                    HIGH RISK
                  </span>
                </div>
                <p className="text-sm text-slate-400 mt-2">
                  Immediate attention required
                </p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Recommended Actions */}
      <div className="risk-card">
        <h3 className="text-lg font-semibold text-white mb-4">Recommended Actions</h3>
        <div className="space-y-4">
          {alerts.alert_level === 'CRITICAL' && (
            <>
              <div className="flex items-start space-x-3">
                <div className="w-2 h-2 bg-red-400 rounded-full mt-2"></div>
                <div>
                  <p className="font-medium text-white">Immediate Portfolio Protection</p>
                  <p className="text-sm text-slate-400">
                    Consider emergency hedging positions or reduce exposure to high-risk assets
                  </p>
                </div>
              </div>
              <div className="flex items-start space-x-3">
                <div className="w-2 h-2 bg-red-400 rounded-full mt-2"></div>
                <div>
                  <p className="font-medium text-white">Activate Stop-Loss Orders</p>
                  <p className="text-sm text-slate-400">
                    Ensure all positions have appropriate stop-loss protection
                  </p>
                </div>
              </div>
              <div className="flex items-start space-x-3">
                <div className="w-2 h-2 bg-red-400 rounded-full mt-2"></div>
                <div>
                  <p className="font-medium text-white">Monitor Market Conditions</p>
                  <p className="text-sm text-slate-400">
                    Stay alert for rapid market changes and be ready to act quickly
                  </p>
                </div>
              </div>
            </>
          )}

          {alerts.alert_level === 'HIGH' && (
            <>
              <div className="flex items-start space-x-3">
                <div className="w-2 h-2 bg-orange-400 rounded-full mt-2"></div>
                <div>
                  <p className="font-medium text-white">Increase Defensive Positions</p>
                  <p className="text-sm text-slate-400">
                    Consider adding hedge positions or reducing leverage
                  </p>
                </div>
              </div>
              <div className="flex items-start space-x-3">
                <div className="w-2 h-2 bg-orange-400 rounded-full mt-2"></div>
                <div>
                  <p className="font-medium text-white">Review Risk Exposure</p>
                  <p className="text-sm text-slate-400">
                    Assess current portfolio risk and consider rebalancing
                  </p>
                </div>
              </div>
            </>
          )}

          {alerts.alert_level === 'MEDIUM' && (
            <>
              <div className="flex items-start space-x-3">
                <div className="w-2 h-2 bg-yellow-400 rounded-full mt-2"></div>
                <div>
                  <p className="font-medium text-white">Monitor Closely</p>
                  <p className="text-sm text-slate-400">
                    Keep a close eye on market conditions and be prepared for changes
                  </p>
                </div>
              </div>
              <div className="flex items-start space-x-3">
                <div className="w-2 h-2 bg-yellow-400 rounded-full mt-2"></div>
                <div>
                  <p className="font-medium text-white">Prepare for Volatility</p>
                  <p className="text-sm text-slate-400">
                    Ensure you have adequate liquidity and risk management in place
                  </p>
                </div>
              </div>
            </>
          )}

          {alerts.alert_level === 'LOW' && (
            <div className="flex items-start space-x-3">
              <div className="w-2 h-2 bg-green-400 rounded-full mt-2"></div>
              <div>
                <p className="font-medium text-white">Normal Operations</p>
                <p className="text-sm text-slate-400">
                  Continue regular monitoring and maintain current risk management practices
                </p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Alert History */}
      <div className="risk-card">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
          <Clock className="w-5 h-5 mr-2 text-blue-400" />
          Recent Alert History
        </h3>
        <div className="space-y-3">
          <div className="flex items-center justify-between p-3 bg-slate-700/30 rounded-lg">
            <div className="flex items-center space-x-3">
              <div className="w-3 h-3 bg-red-400 rounded-full"></div>
              <div>
                <p className="font-medium text-white">CRITICAL Alert</p>
                <p className="text-sm text-slate-400">Multiple assets exceeded risk threshold</p>
              </div>
            </div>
            <span className="text-sm text-slate-400">
              {new Date(alerts.timestamp).toLocaleString()}
            </span>
          </div>

          <div className="flex items-center justify-between p-3 bg-slate-700/30 rounded-lg">
            <div className="flex items-center space-x-3">
              <div className="w-3 h-3 bg-orange-400 rounded-full"></div>
              <div>
                <p className="font-medium text-white">HIGH Alert</p>
                <p className="text-sm text-slate-400">Elevated risk detected in crypto markets</p>
              </div>
            </div>
            <span className="text-sm text-slate-400">2 minutes ago</span>
          </div>

          <div className="flex items-center justify-between p-3 bg-slate-700/30 rounded-lg">
            <div className="flex items-center space-x-3">
              <div className="w-3 h-3 bg-yellow-400 rounded-full"></div>
              <div>
                <p className="font-medium text-white">MEDIUM Alert</p>
                <p className="text-sm text-slate-400">Increased volatility in traditional assets</p>
              </div>
            </div>
            <span className="text-sm text-slate-400">5 minutes ago</span>
          </div>
        </div>
      </div>

      {/* Refresh Button */}
      <div className="flex justify-center">
        <button
          onClick={() => window.location.reload()}
          className="flex items-center space-x-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors"
        >
          <RefreshCw className="w-4 h-4" />
          <span>Refresh Alerts</span>
        </button>
      </div>

      {/* Last Update */}
      <div className="text-center text-sm text-slate-400">
        Last updated: {lastUpdate.toLocaleTimeString()}
      </div>
    </div>
  );
};

export default AlertsPanel;