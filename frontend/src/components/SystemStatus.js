import React, { useState, useEffect } from 'react';
import { CheckCircle, XCircle, AlertTriangle, Activity, Clock, Server, Database, Zap } from 'lucide-react';

const SystemStatus = () => {
  const [healthData, setHealthData] = useState(null);
  const [statusData, setStatusData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchSystemStatus = async () => {
      try {
        const [healthResponse, statusResponse] = await Promise.all([
          fetch('/health'),
          fetch('/status')
        ]);
        
        const health = await healthResponse.json();
        const status = await statusResponse.json();
        
        setHealthData(health);
        setStatusData(status);
      } catch (error) {
        console.error('Failed to fetch system status:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchSystemStatus();
    const interval = setInterval(fetchSystemStatus, 10000); // Update every 10 seconds

    return () => clearInterval(interval);
  }, []);

  const getStatusIcon = (status) => {
    switch (status) {
      case 'healthy':
      case 'active':
      case 'available':
        return <CheckCircle className="w-5 h-5 text-green-400" />;
      case 'degraded':
      case 'warning':
        return <AlertTriangle className="w-5 h-5 text-yellow-400" />;
      case 'unhealthy':
      case 'error':
      case 'not_available':
        return <XCircle className="w-5 h-5 text-red-400" />;
      default:
        return <Clock className="w-5 h-5 text-gray-400" />;
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'healthy':
      case 'active':
      case 'available':
        return 'text-green-400 bg-green-400/10 border-green-400/20';
      case 'degraded':
      case 'warning':
        return 'text-yellow-400 bg-yellow-400/10 border-yellow-400/20';
      case 'unhealthy':
      case 'error':
      case 'not_available':
        return 'text-red-400 bg-red-400/10 border-red-400/20';
      default:
        return 'text-gray-400 bg-gray-400/10 border-gray-400/20';
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-400"></div>
        <span className="ml-4 text-slate-400">Loading system status...</span>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Overall System Health */}
      {healthData && (
        <div className={`risk-card ${getStatusColor(healthData.status)}`}>
          <div className="flex items-center space-x-4">
            <div className={getStatusColor(healthData.status)}>
              {getStatusIcon(healthData.status)}
            </div>
            <div className="flex-1">
              <h2 className="text-xl font-bold text-white mb-2">
                System Status: {healthData.status.toUpperCase()}
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div>
                  <p className="text-sm text-slate-400">Uptime</p>
                  <p className="text-lg font-bold text-white">
                    {Math.floor(healthData.uptime_seconds / 3600)}h {Math.floor((healthData.uptime_seconds % 3600) / 60)}m
                  </p>
                </div>
                <div>
                  <p className="text-sm text-slate-400">Version</p>
                  <p className="text-lg font-bold text-white">{healthData.version}</p>
                </div>
                <div>
                  <p className="text-sm text-slate-400">Last Check</p>
                  <p className="text-lg font-bold text-white">
                    {new Date(healthData.timestamp).toLocaleTimeString()}
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Component Status */}
      {healthData?.components && (
        <div className="risk-card">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
            <Server className="w-5 h-5 mr-2 text-blue-400" />
            Component Status
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {Object.entries(healthData.components).map(([component, status]) => (
              <div key={component} className="bg-slate-700/30 rounded-lg p-4 border border-slate-600/30">
                <div className="flex items-center justify-between mb-2">
                  <span className="font-medium text-white capitalize">
                    {component.replace(/_/g, ' ')}
                  </span>
                  {getStatusIcon(status)}
                </div>
                <span className={`px-2 py-1 rounded text-xs font-medium capitalize ${getStatusColor(status)}`}>
                  {status.replace(/_/g, ' ')}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Model Information */}
      {healthData?.model_info && (
        <div className="risk-card">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
            <Database className="w-5 h-5 mr-2 text-blue-400" />
            AI Model Information
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <p className="text-sm text-slate-400">Model Version</p>
              <p className="text-lg font-bold text-white">
                {healthData.model_info.version || 'Unknown'}
              </p>
            </div>
            <div>
              <p className="text-sm text-slate-400">Status</p>
              <p className="text-lg font-bold text-white capitalize">
                {healthData.model_info.status || 'Unknown'}
              </p>
            </div>
            <div>
              <p className="text-sm text-slate-400">Loaded At</p>
              <p className="text-lg font-bold text-white">
                {healthData.model_info.loaded_at 
                  ? new Date(healthData.model_info.loaded_at).toLocaleString()
                  : 'Not loaded'
                }
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Performance Metrics */}
      {statusData?.performance_metrics && (
        <div className="risk-card">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
            <Activity className="w-5 h-5 mr-2 text-blue-400" />
            Performance Metrics
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="metric-card">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-slate-400">Cache Size</p>
                  <p className="text-2xl font-bold text-white">{statusData.cache_size}</p>
                </div>
                <Database className="w-8 h-8 text-blue-400" />
              </div>
            </div>

            <div className="metric-card">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-slate-400">Uptime</p>
                  <p className="text-2xl font-bold text-white">
                    {Math.floor(statusData.performance_metrics.uptime_seconds / 3600)}h
                  </p>
                </div>
                <Clock className="w-8 h-8 text-green-400" />
              </div>
            </div>

            <div className="metric-card">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-slate-400">Cache Hit Ratio</p>
                  <p className="text-2xl font-bold text-white">
                    {(statusData.performance_metrics.cache_hit_ratio * 100).toFixed(1)}%
                  </p>
                </div>
                <Zap className="w-8 h-8 text-yellow-400" />
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Service Information */}
      {statusData && (
        <div className="risk-card">
          <h3 className="text-lg font-semibold text-white mb-4">Service Information</h3>
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <span className="text-slate-400">Service Name</span>
              <span className="text-white font-medium">{statusData.service}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-slate-400">Current Status</span>
              <span className={`px-2 py-1 rounded text-xs font-medium capitalize ${getStatusColor(statusData.status)}`}>
                {statusData.status}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-slate-400">Current Time</span>
              <span className="text-white font-medium">
                {new Date(statusData.current_time).toLocaleString()}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-slate-400">Model Version</span>
              <span className="text-white font-medium">{statusData.model_version}</span>
            </div>
          </div>
        </div>
      )}

      {/* System Resources */}
      <div className="risk-card">
        <h3 className="text-lg font-semibold text-white mb-4">System Resources</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-medium text-white mb-3">Memory Usage</h4>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-slate-400">Used</span>
                <span className="text-white">2.1 GB</span>
              </div>
              <div className="w-full bg-slate-700 rounded-full h-2">
                <div className="bg-blue-400 h-2 rounded-full" style={{ width: '65%' }}></div>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-slate-400">Available</span>
                <span className="text-white">3.2 GB</span>
              </div>
            </div>
          </div>

          <div>
            <h4 className="font-medium text-white mb-3">CPU Usage</h4>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-slate-400">Current</span>
                <span className="text-white">23%</span>
              </div>
              <div className="w-full bg-slate-700 rounded-full h-2">
                <div className="bg-green-400 h-2 rounded-full" style={{ width: '23%' }}></div>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-slate-400">Average (1h)</span>
                <span className="text-white">18%</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SystemStatus;