import React, { useState, useEffect } from 'react';
import './App.css';
import io from 'socket.io-client';

function App() {
  const [botStatus, setBotStatus] = useState({});
  const [tradingSignals, setTradingSignals] = useState([]);
  const [performanceMetrics, setPerformanceMetrics] = useState({});

  useEffect(() => {
    const backendUrl = ''; // Use relative URLs for production

    // Fetch initial data from REST APIs
    const fetchInitialData = async () => {
      try {
        const statusResponse = await fetch(`${backendUrl}/api/status`);
        const statusData = await statusResponse.json();
        setBotStatus(statusData);

        const performanceResponse = await fetch(`${backendUrl}/api/performance`);
        const performanceData = await performanceResponse.json();
        setPerformanceMetrics(performanceData);

        const signalsResponse = await fetch(`${backendUrl}/api/signals`);
        const signalsData = await signalsResponse.json();
        setTradingSignals(signalsData);

      } catch (error) {
        console.error('Error fetching initial data:', error);
      }
    };

    fetchInitialData();

    // Set up WebSocket connection for real-time updates
    const socket = io(backendUrl);

    socket.on('connect', () => {
      console.log('Connected to WebSocket server');
    });

    socket.on('status_update', (data) => {
      setBotStatus(data);
    });

    socket.on('performance_update', (data) => {
      setPerformanceMetrics(data);
    });

    socket.on('signal_update', (data) => {
      setTradingSignals(prev => [data, ...prev.slice(0, 9)]); // Keep last 10 signals
    });

    socket.on('disconnect', () => {
      console.log('Disconnected from WebSocket server');
    });

    socket.on('error', (error) => {
      console.error('WebSocket error:', error);
    });

    return () => {
      socket.disconnect();
    };
  }, []);

  return (
    <div className="App min-h-screen bg-gray-900 text-white p-8">
      <header className="App-header text-center mb-12">
        <h1 className="text-5xl font-bold text-blue-400">Self-Learning Trading Bot Dashboard</h1>
        <p className="text-lg text-gray-400 mt-2">Real-time insights into your automated trading operations</p>
      </header>

      <section className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-12">
        <div className="bg-gray-800 p-6 rounded-lg shadow-lg">
          <h2 className="text-2xl font-semibold text-blue-300 mb-4">Bot Status</h2>
          <p>Status: <span className={`font-bold ${botStatus.status === 'Running' ? 'text-green-500' : 'text-red-500'}`}>{botStatus.status || 'Loading...'}</span></p>
          <p>Last Update: {botStatus.lastUpdate || 'N/A'}</p>
        </div>

        <div className="bg-gray-800 p-6 rounded-lg shadow-lg">
          <h2 className="text-2xl font-semibold text-blue-300 mb-4">Performance Metrics</h2>
          <p>Total Trades: {performanceMetrics.total_trades || 0}</p>
          <p>Winning Trades: {performanceMetrics.winning_trades || 0}</p>
          <p>Win Rate: {(performanceMetrics.win_rate * 100 || 0).toFixed(2)}%</p>
          <p>Total PnL: ${ (performanceMetrics.total_pnl || 0).toFixed(2)}</p>
          <p>Sharpe Ratio: {(performanceMetrics.sharpe_ratio || 0).toFixed(2)}</p>
          <p>Max Drawdown: {(performanceMetrics.max_drawdown * 100 || 0).toFixed(2)}%</p>
        </div>

        <div className="bg-gray-800 p-6 rounded-lg shadow-lg">
          <h2 className="text-2xl font-semibold text-blue-300 mb-4">Latest Signals</h2>
          <ul className="space-y-2">
            {tradingSignals.map(signal => (
              <li key={signal.id} className="flex justify-between items-center text-sm">
                <span>{signal.symbol}: <span className={`font-bold ${signal.signal === 'BUY' ? 'text-green-400' : 'text-red-400'}`}>{signal.signal}</span> @ {signal.price}</span>
                <span className="text-gray-500">Conf: {(signal.confidence * 100).toFixed(0)}%</span>
              </li>
            ))}
          </ul>
        </div>
      </section>

      <section className="bg-gray-800 p-6 rounded-lg shadow-lg">
        <h2 className="text-2xl font-semibold text-blue-300 mb-4">Trade Log (Coming Soon)</h2>
        <p className="text-gray-400">Detailed trade history and analysis will be available here.</p>
      </section>
    </div>
  );
}

export default App;


