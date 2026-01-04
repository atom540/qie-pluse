import React, { useState, useEffect } from 'react';
import { Activity, Clock, Zap, TrendingUp, Target, BarChart3 } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';

const PerformanceMetrics = () => {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        // Simulate performance metrics (in real implementation, this would come from your API)
        const mockMetrics = {
          current_performance: {
            inference_latency_ms: 145,
            blockchain_latency_ms: 2100,
            api_response_time_ms: 262,
            model_accuracy: 87.3,
            uptime_percentage: 99.95,
            requests_per_second: 3.8
          },
          targets: {
            inference_latency_ms: 500,
            blockchain_latency_ms: 3000,
            api_response_time_ms: 500,
            model_accuracy: 85.0,
            uptime_percentage: 99.9,
            requests_per_second: 10.0
          },
          historical_data: Array.from({ length: 24 }, (_, i) => ({
            time: `${23 - i}:00`,
            inference_latency: 120 + Math.random() * 60,
            api_response: 200 + Math.random() * 100,
            accuracy: 85 + Math.random() * 5,
            requests: 2 + Math.random() * 4
          }))
        };
        
        setMetrics(mockMetrics);
      } catch (error) {
        console.error('Failed to fetch performance metrics:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchMetrics();
    const interval = setInterval(fetchMetrics, 30000); // Update every 30 seconds

    return () => clearInterval(interval);
  }, []);

  const getPerformanceStatus = (current, target, higherIsBetter = false) => {
    const ratio = higherIsBetter ? current / target : target / current;
    if (ratio >= 1) return 'excellent';
    if (ratio >= 0.8) return 'good';
    if (ratio >= 0.6) return 'warning';
    return 'poor';
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'excellent': return 'text-green-400 bg-green-400/10 border-green-400/20';
      case 'good': return 'text-blue-400 bg-blue-400/10 border-blue-400/20';
      case 'warning': return 'text-yellow-400 bg-yellow-400/10 border-yellow-400/20';
      case 'poor': return 'text-red-400 bg-red-400/10 border-red-400/20';
      default: return 'text-gray-400 bg-gray-400/10 border-gray-400/20';
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'excellent': return <Target className="w-5 h-5 text-green-400" />;
      case 'good': return <TrendingUp className="w-5 h-5 text-blue-400" />;
      case 'warning': return <Clock className="w-5 h-5 text-yellow-400" />;
      case 'poor': return <Activity className="w-5 h-5 text-red-400" />;
      default: return <BarChart3 className="w-5 h-5 text-gray-400" />;
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-400"></div>
        <span className="ml-4 text-slate-400">Loading performance metrics...</span>
      </div>
    );
  }

  if (!metrics) {
    return (
      <div className="text-center py-12">
        <Activity className="w-12 h-12 text-yellow-400 mx-auto mb-4" />
        <h3 className="text-lg font-medium text-white mb-2">Unable to load performance metrics</h3>
        <p className="text-slate-400">Please check the API connection</p>
      </div>
    );
  }

  const { current_performance, targets, historical_data } = metrics;

  return (
    <div className="space-y-8">
      {/* Performance Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <div className={`metric-card ${getStatusColor(getPerformanceStatus(current_performance.inference_latency_ms, targets.inference_latency_ms))}`}>
          <div className="flex items-center justify-between mb-4">
            <div>
              <p className="text-sm font-medium text-slate-400">Inference Latency</p>
              <p className="text-2xl font-bold text-white">{current_performance.inference_latency_ms}ms</p>
              <p className="text-xs text-slate-400">Target: {targets.inference_latency_ms}ms</p>
            </div>
            <Zap className="w-8 h-8 text-blue-400" />
          </div>
          <div className="flex items-center space-x-2">
            {getStatusIcon(getPerformanceStatus(current_performance.inference_latency_ms, targets.inference_latency_ms))}
            <span className="text-sm capitalize">
              {getPerformanceStatus(current_performance.inference_latency_ms, targets.inference_latency_ms)}
            </span>
          </div>
        </div>

        <div className={`metric-card ${getStatusColor(getPerformanceStatus(current_performance.blockchain_latency_ms, targets.blockchain_latency_ms))}`}>
          <div className="flex items-center justify-between mb-4">
            <div>
              <p className="text-sm font-medium text-slate-400">Blockchain Latency</p>
              <p className="text-2xl font-bold text-white">{current_performance.blockchain_latency_ms}ms</p>
              <p className="text-xs text-slate-400">Target: {targets.blockchain_latency_ms}ms</p>
            </div>
            <Activity className="w-8 h-8 text-green-400" />
          </div>
          <div className="flex items-center space-x-2">
            {getStatusIcon(getPerformanceStatus(current_performance.blockchain_latency_ms, targets.blockchain_latency_ms))}
            <span className="text-sm capitalize">
              {getPerformanceStatus(current_performance.blockchain_latency_ms, targets.blockchain_latency_ms)}
            </span>
          </div>
        </div>

        <div className={`metric-card ${getStatusColor(getPerformanceStatus(current_performance.model_accuracy, targets.model_accuracy, true))}`}>
          <div className="flex items-center justify-between mb-4">
            <div>
              <p className="text-sm font-medium text-slate-400">Model Accuracy</p>
              <p className="text-2xl font-bold text-white">{current_performance.model_accuracy}%</p>
              <p className="text-xs text-slate-400">Target: {targets.model_accuracy}%</p>
            </div>
            <Target className="w-8 h-8 text-purple-400" />
          </div>
          <div className="flex items-center space-x-2">
            {getStatusIcon(getPerformanceStatus(current_performance.model_accuracy, targets.model_accuracy, true))}
            <span className="text-sm capitalize">
              {getPerformanceStatus(current_performance.model_accuracy, targets.model_accuracy, true)}
            </span>
          </div>
        </div>
      </div>

      {/* Additional Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="metric-card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-slate-400">API Response Time</p>
              <p className="text-2xl font-bold text-white">{current_performance.api_response_time_ms}ms</p>
            </div>
            <Clock className="w-8 h-8 text-blue-400" />
          </div>
        </div>

        <div className="metric-card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-slate-400">System Uptime</p>
              <p className="text-2xl font-bold text-white">{current_performance.uptime_percentage}%</p>
            </div>
            <TrendingUp className="w-8 h-8 text-green-400" />
          </div>
        </div>

        <div className="metric-card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-slate-400">Requests/Second</p>
              <p className="text-2xl font-bold text-white">{current_performance.requests_per_second}</p>
            </div>
            <BarChart3 className="w-8 h-8 text-orange-400" />
          </div>
        </div>
      </div>

      {/* Performance Targets vs Actual */}
      <div className="risk-card">
        <h3 className="text-lg font-semibold text-white mb-4">Performance Targets vs Actual</h3>
        <div className="space-y-4">
          {Object.entries(targets).map(([metric, target]) => {
            const current = current_performance[metric];
            const isHigherBetter = metric === 'model_accuracy' || metric === 'uptime_percentage' || metric === 'requests_per_second';
            const percentage = isHigherBetter ? (current / target) * 100 : (target / current) * 100;
            const status = getPerformanceStatus(current, target, isHigherBetter);
            
            return (
              <div key={metric} className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-slate-400 capitalize">
                    {metric.replace(/_/g, ' ')}
                  </span>
                  <span className={`text-sm font-medium ${getStatusColor(status).split(' ')[0]}`}>
                    {current}{metric.includes('percentage') ? '%' : metric.includes('ms') ? 'ms' : ''}
                  </span>
                </div>
                <div className="w-full bg-slate-700 rounded-full h-2">
                  <div 
                    className={`h-2 rounded-full ${
                      status === 'excellent' ? 'bg-green-400' :
                      status === 'good' ? 'bg-blue-400' :
                      status === 'warning' ? 'bg-yellow-400' : 'bg-red-400'
                    }`}
                    style={{ width: `${Math.min(100, Math.max(0, percentage))}%` }}
                  ></div>
                </div>
                <div className="flex justify-between text-xs text-slate-500">
                  <span>Target: {target}{metric.includes('percentage') ? '%' : metric.includes('ms') ? 'ms' : ''}</span>
                  <span>{percentage.toFixed(1)}% of target</span>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Historical Performance Chart */}
      <div className="risk-card">
        <h3 className="text-lg font-semibold text-white mb-4">24-Hour Performance Trend</h3>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={historical_data}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="time" stroke="#9CA3AF" />
              <YAxis stroke="#9CA3AF" />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#1F2937', 
                  border: '1px solid #374151',
                  borderRadius: '8px'
                }}
              />
              <Line 
                type="monotone" 
                dataKey="inference_latency" 
                stroke="#3B82F6" 
                strokeWidth={2}
                name="Inference Latency (ms)"
              />
              <Line 
                type="monotone" 
                dataKey="api_response" 
                stroke="#10B981" 
                strokeWidth={2}
                name="API Response (ms)"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Model Accuracy Trend */}
      <div className="risk-card">
        <h3 className="text-lg font-semibold text-white mb-4">Model Accuracy Trend</h3>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={historical_data}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="time" stroke="#9CA3AF" />
              <YAxis stroke="#9CA3AF" domain={[80, 95]} />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#1F2937', 
                  border: '1px solid #374151',
                  borderRadius: '8px'
                }}
              />
              <Area 
                type="monotone" 
                dataKey="accuracy" 
                stroke="#8B5CF6" 
                fill="#8B5CF6" 
                fillOpacity={0.3}
                name="Accuracy (%)"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Performance Summary */}
      <div className="risk-card">
        <h3 className="text-lg font-semibold text-white mb-4">Performance Summary</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-medium text-white mb-3">Achievements</h4>
            <ul className="space-y-2">
              <li className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                <span className="text-sm text-slate-300">Model accuracy exceeds target by 2.3%</span>
              </li>
              <li className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                <span className="text-sm text-slate-300">Blockchain latency under 3-second target</span>
              </li>
              <li className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                <span className="text-sm text-slate-300">99.95% uptime exceeds 99.9% target</span>
              </li>
            </ul>
          </div>
          
          <div>
            <h4 className="font-medium text-white mb-3">Areas for Improvement</h4>
            <ul className="space-y-2">
              <li className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-yellow-400 rounded-full"></div>
                <span className="text-sm text-slate-300">Inference latency slightly above target</span>
              </li>
              <li className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-blue-400 rounded-full"></div>
                <span className="text-sm text-slate-300">Request throughput can be optimized</span>
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PerformanceMetrics;