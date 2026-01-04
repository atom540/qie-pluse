import React, { useState, useEffect } from 'react';
import { Shield, Activity, TrendingUp, AlertTriangle, CheckCircle, XCircle, Clock, Zap } from 'lucide-react';
import RiskDashboard from './components/RiskDashboard';
import SystemStatus from './components/SystemStatus';
import AlertsPanel from './components/AlertsPanel';
import PerformanceMetrics from './components/PerformanceMetrics';
import BlockchainStatus from './components/BlockchainStatus';

function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [systemHealth, setSystemHealth] = useState('loading');

  useEffect(() => {
    // Check system health on load
    fetch('/health')
      .then(res => res.json())
      .then(data => setSystemHealth(data.status))
      .catch(() => setSystemHealth('error'));
  }, []);

  const tabs = [
    { id: 'dashboard', name: 'Risk Dashboard', icon: Shield },
    { id: 'alerts', name: 'Alerts', icon: AlertTriangle },
    { id: 'performance', name: 'Performance', icon: TrendingUp },
    { id: 'blockchain', name: 'Blockchain', icon: Zap },
    { id: 'status', name: 'System Status', icon: Activity },
  ];

  const getHealthIcon = () => {
    switch (systemHealth) {
      case 'healthy': return <CheckCircle className="w-5 h-5 text-green-400" />;
      case 'degraded': return <AlertTriangle className="w-5 h-5 text-yellow-400" />;
      case 'error': return <XCircle className="w-5 h-5 text-red-400" />;
      default: return <Clock className="w-5 h-5 text-gray-400 animate-spin" />;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      {/* Header */}
      <header className="bg-slate-800/50 backdrop-blur-sm border-b border-slate-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <Shield className="w-8 h-8 text-blue-400" />
                <div>
                  <h1 className="text-xl font-bold text-white">QIE V3 AI Risk Oracle</h1>
                  <p className="text-sm text-slate-400">Predictive Security System for Money</p>
                </div>
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                {getHealthIcon()}
                <span className="text-sm text-slate-300 capitalize">{systemHealth}</span>
              </div>
              <div className="text-sm text-slate-400">
                {new Date().toLocaleTimeString()}
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Navigation */}
      <nav className="bg-slate-800/30 border-b border-slate-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex space-x-8">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center space-x-2 py-4 px-2 border-b-2 font-medium text-sm transition-colors ${
                    activeTab === tab.id
                      ? 'border-blue-400 text-blue-400'
                      : 'border-transparent text-slate-400 hover:text-slate-300 hover:border-slate-300'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span>{tab.name}</span>
                </button>
              );
            })}
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {activeTab === 'dashboard' && <RiskDashboard />}
        {activeTab === 'alerts' && <AlertsPanel />}
        {activeTab === 'performance' && <PerformanceMetrics />}
        {activeTab === 'blockchain' && <BlockchainStatus />}
        {activeTab === 'status' && <SystemStatus />}
      </main>

      {/* Footer */}
      <footer className="bg-slate-800/30 border-t border-slate-700 mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex justify-between items-center">
            <div className="text-sm text-slate-400">
              © 2024 QIE V3 AI Risk Oracle - Hackathon Demo
            </div>
            <div className="flex items-center space-x-4 text-sm text-slate-400">
              <span>Powered by QIE V3 Blockchain</span>
              <span>•</span>
              <span>AI/ML Risk Assessment</span>
              <span>•</span>
              <span>Real-time Protection</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;