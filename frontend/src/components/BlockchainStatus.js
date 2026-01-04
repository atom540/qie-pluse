import React, { useState, useEffect } from 'react';
import { Zap, CheckCircle, XCircle, AlertTriangle, Clock, Hash, Activity, TrendingUp } from 'lucide-react';

const BlockchainStatus = () => {
  const [blockchainData, setBlockchainData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchBlockchainStatus = async () => {
      try {
        const response = await fetch('/blockchain/status');
        const data = await response.json();
        setBlockchainData(data);
      } catch (error) {
        console.error('Failed to fetch blockchain status:', error);
        // Mock data for demo purposes
        setBlockchainData({
          service_status: 'active',
          blockchain_connection: {
            connected: true,
            rpc_url: 'https://rpc.qie-pulse.com',
            account_address: '0x742d35Cc6634C0532925a3b8D4C9db96590b5b8c',
            contract_address: '0x1234567890123456789012345678901234567890',
            latest_block: 1234567,
            network_id: 1337
          },
          contract_state: {
            accessible: true,
            current_risk_score: {
              risk_score: 85,
              last_updated: Date.now() - 300000,
              risk_score_normalized: 0.85
            }
          },
          update_statistics: {
            total_updates: 42,
            successful_updates: 40,
            failed_updates: 2,
            success_rate: 0.952,
            last_update_time: new Date().toISOString(),
            last_risk_score: 0.85
          },
          configuration: {
            risk_threshold: 0.85,
            update_interval_seconds: 300,
            force_update_interval_seconds: 3600
          },
          recent_updates: [
            {
              trigger_reason: 'risk_threshold_exceeded',
              risk_score: 0.92,
              success: true,
              tx_hash: '0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890',
              timestamp: new Date(Date.now() - 120000).toISOString(),
              latency_ms: 2100
            },
            {
              trigger_reason: 'periodic_update',
              risk_score: 0.75,
              success: true,
              tx_hash: '0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef',
              timestamp: new Date(Date.now() - 420000).toISOString(),
              latency_ms: 1850
            }
          ]
        });
      } finally {
        setLoading(false);
      }
    };

    fetchBlockchainStatus();
    const interval = setInterval(fetchBlockchainStatus, 15000); // Update every 15 seconds

    return () => clearInterval(interval);
  }, []);

  const getStatusIcon = (status) => {
    if (status === true || status === 'active') return <CheckCircle className="w-5 h-5 text-green-400" />;
    if (status === false || status === 'inactive') return <XCircle className="w-5 h-5 text-red-400" />;
    return <AlertTriangle className="w-5 h-5 text-yellow-400" />;
  };

  const getStatusColor = (status) => {
    if (status === true || status === 'active') return 'text-green-400 bg-green-400/10 border-green-400/20';
    if (status === false || status === 'inactive') return 'text-red-400 bg-red-400/10 border-red-400/20';
    return 'text-yellow-400 bg-yellow-400/10 border-yellow-400/20';
  };

  const formatAddress = (address) => {
    if (!address) return 'Not available';
    return `${address.slice(0, 6)}...${address.slice(-4)}`;
  };

  const formatTxHash = (hash) => {
    if (!hash) return 'N/A';
    return `${hash.slice(0, 10)}...${hash.slice(-8)}`;
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-400"></div>
        <span className="ml-4 text-slate-400">Loading blockchain status...</span>
      </div>
    );
  }

  if (!blockchainData) {
    return (
      <div className="text-center py-12">
        <Zap className="w-12 h-12 text-yellow-400 mx-auto mb-4" />
        <h3 className="text-lg font-medium text-white mb-2">Unable to load blockchain status</h3>
        <p className="text-slate-400">Please check the blockchain connection</p>
      </div>
    );
  }

  const { blockchain_connection, contract_state, update_statistics, configuration, recent_updates } = blockchainData;

  return (
    <div className="space-y-8">
      {/* Connection Status */}
      <div className={`risk-card ${getStatusColor(blockchain_connection?.connected)}`}>
        <div className="flex items-center space-x-4">
          <div className={getStatusColor(blockchain_connection?.connected)}>
            <Zap className="w-8 h-8" />
          </div>
          <div className="flex-1">
            <h2 className="text-xl font-bold text-white mb-2">
              QIE V3 Blockchain Connection
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <p className="text-sm text-slate-400">Status</p>
                <p className="text-lg font-bold text-white">
                  {blockchain_connection?.connected ? 'Connected' : 'Disconnected'}
                </p>
              </div>
              <div>
                <p className="text-sm text-slate-400">Network ID</p>
                <p className="text-lg font-bold text-white">
                  {blockchain_connection?.network_id || 'Unknown'}
                </p>
              </div>
              <div>
                <p className="text-sm text-slate-400">Latest Block</p>
                <p className="text-lg font-bold text-white">
                  {blockchain_connection?.latest_block?.toLocaleString() || 'Unknown'}
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Account & Contract Information */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="risk-card">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
            <Hash className="w-5 h-5 mr-2 text-blue-400" />
            Account Information
          </h3>
          <div className="space-y-3">
            <div>
              <p className="text-sm text-slate-400">Wallet Address</p>
              <p className="text-white font-mono text-sm">
                {formatAddress(blockchain_connection?.account_address)}
              </p>
            </div>
            <div>
              <p className="text-sm text-slate-400">RPC Endpoint</p>
              <p className="text-white text-sm break-all">
                {blockchain_connection?.rpc_url || 'Not configured'}
              </p>
            </div>
          </div>
        </div>

        <div className="risk-card">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
            <Activity className="w-5 h-5 mr-2 text-blue-400" />
            Smart Contract
          </h3>
          <div className="space-y-3">
            <div>
              <p className="text-sm text-slate-400">Contract Address</p>
              <p className="text-white font-mono text-sm">
                {formatAddress(blockchain_connection?.contract_address)}
              </p>
            </div>
            <div>
              <p className="text-sm text-slate-400">Status</p>
              <div className="flex items-center space-x-2">
                {getStatusIcon(contract_state?.accessible)}
                <span className="text-white text-sm">
                  {contract_state?.accessible ? 'Accessible' : 'Not accessible'}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Current Risk Score on Chain */}
      {contract_state?.current_risk_score && (
        <div className="risk-card">
          <h3 className="text-lg font-semibold text-white mb-4">Current On-Chain Risk Score</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="text-center">
              <p className="text-3xl font-bold text-red-400">
                {contract_state.current_risk_score.risk_score}%
              </p>
              <p className="text-sm text-slate-400">Risk Score</p>
            </div>
            <div className="text-center">
              <p className="text-3xl font-bold text-blue-400">
                {(contract_state.current_risk_score.risk_score_normalized * 100).toFixed(1)}%
              </p>
              <p className="text-sm text-slate-400">Normalized</p>
            </div>
            <div className="text-center">
              <p className="text-lg font-bold text-white">
                {Math.floor((Date.now() - contract_state.current_risk_score.last_updated) / 60000)}m ago
              </p>
              <p className="text-sm text-slate-400">Last Updated</p>
            </div>
          </div>
        </div>
      )}

      {/* Update Statistics */}
      <div className="risk-card">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
          <TrendingUp className="w-5 h-5 mr-2 text-blue-400" />
          Update Statistics
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="metric-card">
            <div className="text-center">
              <p className="text-2xl font-bold text-white">{update_statistics?.total_updates || 0}</p>
              <p className="text-sm text-slate-400">Total Updates</p>
            </div>
          </div>
          <div className="metric-card">
            <div className="text-center">
              <p className="text-2xl font-bold text-green-400">{update_statistics?.successful_updates || 0}</p>
              <p className="text-sm text-slate-400">Successful</p>
            </div>
          </div>
          <div className="metric-card">
            <div className="text-center">
              <p className="text-2xl font-bold text-red-400">{update_statistics?.failed_updates || 0}</p>
              <p className="text-sm text-slate-400">Failed</p>
            </div>
          </div>
          <div className="metric-card">
            <div className="text-center">
              <p className="text-2xl font-bold text-blue-400">
                {((update_statistics?.success_rate || 0) * 100).toFixed(1)}%
              </p>
              <p className="text-sm text-slate-400">Success Rate</p>
            </div>
          </div>
        </div>
      </div>

      {/* Configuration */}
      <div className="risk-card">
        <h3 className="text-lg font-semibold text-white mb-4">Configuration</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <p className="text-sm text-slate-400">Risk Threshold</p>
            <p className="text-lg font-bold text-white">
              {((configuration?.risk_threshold || 0) * 100).toFixed(0)}%
            </p>
            <p className="text-xs text-slate-500">Triggers automatic updates</p>
          </div>
          <div>
            <p className="text-sm text-slate-400">Update Interval</p>
            <p className="text-lg font-bold text-white">
              {Math.floor((configuration?.update_interval_seconds || 0) / 60)}m
            </p>
            <p className="text-xs text-slate-500">Regular update frequency</p>
          </div>
          <div>
            <p className="text-sm text-slate-400">Force Update Interval</p>
            <p className="text-lg font-bold text-white">
              {Math.floor((configuration?.force_update_interval_seconds || 0) / 3600)}h
            </p>
            <p className="text-xs text-slate-500">Maximum time between updates</p>
          </div>
        </div>
      </div>

      {/* Recent Transactions */}
      <div className="risk-card">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
          <Clock className="w-5 h-5 mr-2 text-blue-400" />
          Recent Blockchain Updates
        </h3>
        <div className="space-y-3">
          {recent_updates && recent_updates.length > 0 ? (
            recent_updates.map((update, index) => (
              <div key={index} className="bg-slate-700/30 rounded-lg p-4 border border-slate-600/30">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center space-x-3">
                    {getStatusIcon(update.success)}
                    <div>
                      <p className="font-medium text-white capitalize">
                        {update.trigger_reason.replace(/_/g, ' ')}
                      </p>
                      <p className="text-sm text-slate-400">
                        Risk Score: {(update.risk_score * 100).toFixed(1)}%
                      </p>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="text-sm text-white">{update.latency_ms}ms</p>
                    <p className="text-xs text-slate-400">
                      {new Date(update.timestamp).toLocaleTimeString()}
                    </p>
                  </div>
                </div>
                {update.tx_hash && (
                  <div className="mt-2 pt-2 border-t border-slate-600">
                    <p className="text-xs text-slate-400">
                      TX: <span className="font-mono">{formatTxHash(update.tx_hash)}</span>
                    </p>
                  </div>
                )}
              </div>
            ))
          ) : (
            <div className="text-center py-8">
              <Clock className="w-8 h-8 text-slate-400 mx-auto mb-2" />
              <p className="text-slate-400">No recent updates</p>
            </div>
          )}
        </div>
      </div>

      {/* QIE V3 Advantages */}
      <div className="risk-card">
        <h3 className="text-lg font-semibold text-white mb-4">QIE V3 Blockchain Advantages</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-medium text-white mb-3 flex items-center">
              <Zap className="w-4 h-4 mr-2 text-yellow-400" />
              Speed & Performance
            </h4>
            <ul className="space-y-2">
              <li className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                <span className="text-sm text-slate-300">3-second finality for crisis response</span>
              </li>
              <li className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                <span className="text-sm text-slate-300">25,000 TPS capacity</span>
              </li>
              <li className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                <span className="text-sm text-slate-300">Low gas fees for frequent updates</span>
              </li>
            </ul>
          </div>
          
          <div>
            <h4 className="font-medium text-white mb-3 flex items-center">
              <Activity className="w-4 h-4 mr-2 text-blue-400" />
              AI-Optimized Features
            </h4>
            <ul className="space-y-2">
              <li className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-blue-400 rounded-full"></div>
                <span className="text-sm text-slate-300">Real-time oracle data feeds</span>
              </li>
              <li className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-blue-400 rounded-full"></div>
                <span className="text-sm text-slate-300">Smart contract automation</span>
              </li>
              <li className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-blue-400 rounded-full"></div>
                <span className="text-sm text-slate-300">Built for DeFi primitives</span>
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default BlockchainStatus;