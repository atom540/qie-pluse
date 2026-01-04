import React, { useState, useEffect } from 'react';
import { AlertTriangle, TrendingUp, TrendingDown, Activity, Clock, Brain, MessageSquare } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';

const RiskDashboard = () => {
  const [riskData, setRiskData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState(new Date());

  useEffect(() => {
    const fetchRiskData = async () => {
      try {
        const response = await fetch('/risk');
        const data = await response.json();
        setRiskData(data);
        setLastUpdate(new Date());
      } catch (error) {
        console.error('Failed to fetch risk data:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchRiskData();
    const interval = setInterval(fetchRiskData, 30000); // Update every 30 seconds

    return () => clearInterval(interval);
  }, []);

  const getRiskColor = (score) => {
    if (score >= 0.8) return 'text-red-400';
    if (score >= 0.6) return 'text-orange-400';
    if (score >= 0.4) return 'text-yellow-400';
    return 'text-green-400';
  };

  const getRiskLevel = (score) => {
    if (score >= 0.8) return 'CRITICAL';
    if (score >= 0.6) return 'HIGH';
    if (score >= 0.4) return 'MEDIUM';
    return 'LOW';
  };

  const getRiskCardClass = (score) => {
    if (score >= 0.8) return 'risk-card risk-critical';
    if (score >= 0.6) return 'risk-card risk-high';
    if (score >= 0.4) return 'risk-card risk-medium';
    return 'risk-card risk-low';
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-400"></div>
        <span className="ml-4 text-slate-400">Loading risk assessment...</span>
      </div>
    );
  }

  if (!riskData) {
    return (
      <div className="text-center py-12">
        <AlertTriangle className="w-12 h-12 text-yellow-400 mx-auto mb-4" />
        <h3 className="text-lg font-medium text-white mb-2">Unable to load risk data</h3>
        <p className="text-slate-400">Please check the API connection</p>
      </div>
    );
  }

  const assets = Object.entries(riskData.assessments || {});
  const summary = riskData.summary || {};

  // Prepare chart data
  const chartData = assets.map(([asset, data]) => ({
    asset,
    risk: data.combined_risk_score,
    ml: data.ml_risk_score,
    sentiment: data.sentiment_risk_score,
    confidence: data.confidence
  }));

  return (
    <div className="space-y-8">
      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="metric-card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-slate-400">Average Risk</p>
              <p className={`text-2xl font-bold ${getRiskColor(summary.average_risk_score || 0)}`}>
                {((summary.average_risk_score || 0) * 100).toFixed(1)}%
              </p>
            </div>
            <Activity className="w-8 h-8 text-blue-400" />
          </div>
        </div>

        <div className="metric-card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-slate-400">Highest Risk</p>
              <p className={`text-2xl font-bold ${getRiskColor(summary.highest_risk_score || 0)}`}>
                {((summary.highest_risk_score || 0) * 100).toFixed(1)}%
              </p>
            </div>
            <TrendingUp className="w-8 h-8 text-red-400" />
          </div>
        </div>

        <div className="metric-card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-slate-400">Assets Monitored</p>
              <p className="text-2xl font-bold text-white">{summary.total_assets || 0}</p>
            </div>
            <TrendingDown className="w-8 h-8 text-green-400" />
          </div>
        </div>

        <div className="metric-card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-slate-400">Response Time</p>
              <p className="text-2xl font-bold text-white">{(riskData.total_latency_ms || 0).toFixed(0)}ms</p>
            </div>
            <Clock className="w-8 h-8 text-blue-400" />
          </div>
        </div>
      </div>

      {/* Risk Level Distribution */}
      <div className="risk-card">
        <h3 className="text-lg font-semibold text-white mb-4">Risk Level Distribution</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {Object.entries(summary.risk_level_distribution || {}).map(([level, count]) => (
            <div key={level} className="text-center">
              <div className={`text-2xl font-bold ${
                level === 'CRITICAL' ? 'text-red-400' :
                level === 'HIGH' ? 'text-orange-400' :
                level === 'MEDIUM' ? 'text-yellow-400' : 'text-green-400'
              }`}>
                {count}
              </div>
              <div className="text-sm text-slate-400">{level}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Asset Risk Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {assets.map(([asset, data]) => (
          <div key={asset} className={getRiskCardClass(data.combined_risk_score)}>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-white">{asset}</h3>
              <span className={`px-2 py-1 rounded text-xs font-medium ${
                data.risk_level === 'CRITICAL' ? 'bg-red-500/20 text-red-400' :
                data.risk_level === 'HIGH' ? 'bg-orange-500/20 text-orange-400' :
                data.risk_level === 'MEDIUM' ? 'bg-yellow-500/20 text-yellow-400' :
                'bg-green-500/20 text-green-400'
              }`}>
                {data.risk_level}
              </span>
            </div>

            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-sm text-slate-400">Combined Risk</span>
                <span className={`font-bold ${getRiskColor(data.combined_risk_score)}`}>
                  {(data.combined_risk_score * 100).toFixed(1)}%
                </span>
              </div>

              <div className="flex justify-between items-center">
                <span className="text-sm text-slate-400 flex items-center">
                  <Brain className="w-3 h-3 mr-1" />
                  ML Score
                </span>
                <span className="text-sm text-white">
                  {(data.ml_risk_score * 100).toFixed(1)}%
                </span>
              </div>

              <div className="flex justify-between items-center">
                <span className="text-sm text-slate-400 flex items-center">
                  <MessageSquare className="w-3 h-3 mr-1" />
                  Sentiment
                </span>
                <span className="text-sm text-white">
                  {(data.sentiment_risk_score * 100).toFixed(1)}%
                </span>
              </div>

              <div className="flex justify-between items-center">
                <span className="text-sm text-slate-400">Confidence</span>
                <span className="text-sm text-white">
                  {(data.confidence * 100).toFixed(1)}%
                </span>
              </div>

              <div className="flex justify-between items-center">
                <span className="text-sm text-slate-400">Latency</span>
                <span className="text-sm text-white">
                  {data.inference_latency_ms.toFixed(0)}ms
                </span>
              </div>

              {data.reasoning && data.reasoning.length > 0 && (
                <div className="mt-3 pt-3 border-t border-slate-600">
                  <p className="text-xs text-slate-400 mb-1">AI Reasoning:</p>
                  <p className="text-xs text-slate-300">{data.reasoning[0]}</p>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Risk Comparison Chart */}
      <div className="risk-card">
        <h3 className="text-lg font-semibold text-white mb-4">Risk Score Comparison</h3>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="asset" stroke="#9CA3AF" />
              <YAxis stroke="#9CA3AF" />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#1F2937', 
                  border: '1px solid #374151',
                  borderRadius: '8px'
                }}
              />
              <Bar dataKey="risk" fill="#3B82F6" name="Combined Risk" />
              <Bar dataKey="ml" fill="#10B981" name="ML Score" />
              <Bar dataKey="sentiment" fill="#F59E0B" name="Sentiment" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Last Update */}
      <div className="text-center text-sm text-slate-400">
        Last updated: {lastUpdate.toLocaleTimeString()}
      </div>
    </div>
  );
};

export default RiskDashboard;