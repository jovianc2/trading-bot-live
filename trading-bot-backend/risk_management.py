import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime, timedelta

class RiskManager:
    def __init__(self, initial_capital=100000, max_drawdown=0.10, max_risk_per_trade=0.02):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_drawdown = max_drawdown
        self.max_risk_per_trade = max_risk_per_trade
        self.peak_capital = initial_capital
        self.current_drawdown = 0.0
        self.daily_trades = 0
        self.max_daily_trades = 10
        self.last_trade_date = None
        
    def calculate_position_size(self, entry_price, stop_loss_price, confidence_score):
        """
        Calculate optimal position size based on risk management rules
        """
        try:
            # Risk per trade based on confidence (higher confidence = higher risk)
            risk_multiplier = min(confidence_score * 1.5, 1.0)  # Cap at 1.0
            adjusted_risk = self.max_risk_per_trade * risk_multiplier
            
            # Calculate risk amount in dollars
            risk_amount = self.current_capital * adjusted_risk
            
            # Calculate price difference (risk per unit)
            price_diff = abs(entry_price - stop_loss_price)
            if price_diff == 0:
                return 0
            
            # Calculate position size
            position_size = risk_amount / price_diff
            
            # Ensure position size doesn't exceed available capital
            max_position_value = self.current_capital * 0.95  # Use 95% of capital max
            max_position_size = max_position_value / entry_price
            
            return min(position_size, max_position_size)
            
        except Exception as e:
            print(f"Error calculating position size: {e}")
            return 0
    
    def calculate_stop_loss(self, entry_price, signal_type, volatility=0.02):
        """
        Calculate dynamic stop loss based on volatility and signal type
        """
        try:
            # Base stop loss percentage
            base_stop_pct = 0.015  # 1.5%
            
            # Adjust based on volatility
            volatility_adjusted_stop = base_stop_pct + (volatility * 0.5)
            
            if signal_type.lower() == 'buy':
                stop_loss = entry_price * (1 - volatility_adjusted_stop)
            else:  # sell/short
                stop_loss = entry_price * (1 + volatility_adjusted_stop)
                
            return stop_loss
            
        except Exception as e:
            print(f"Error calculating stop loss: {e}")
            return entry_price * 0.98 if signal_type.lower() == 'buy' else entry_price * 1.02
    
    def calculate_take_profit(self, entry_price, stop_loss_price, signal_type, risk_reward_ratio=2.0):
        """
        Calculate take profit based on risk-reward ratio
        """
        try:
            risk_amount = abs(entry_price - stop_loss_price)
            reward_amount = risk_amount * risk_reward_ratio
            
            if signal_type.lower() == 'buy':
                take_profit = entry_price + reward_amount
            else:  # sell/short
                take_profit = entry_price - reward_amount
                
            return take_profit
            
        except Exception as e:
            print(f"Error calculating take profit: {e}")
            return entry_price * 1.04 if signal_type.lower() == 'buy' else entry_price * 0.96
    
    def check_drawdown_limit(self):
        """
        Check if current drawdown exceeds the maximum allowed
        """
        self.current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        return self.current_drawdown >= self.max_drawdown
    
    def update_capital(self, new_capital):
        """
        Update current capital and peak capital
        """
        self.current_capital = new_capital
        self.peak_capital = max(self.peak_capital, new_capital)
    
    def can_trade(self, confidence_threshold=0.75):
        """
        Check if trading is allowed based on various risk factors
        """
        # Check drawdown limit
        if self.check_drawdown_limit():
            return False, "Drawdown limit exceeded"
        
        # Check daily trade limit
        today = datetime.now().date()
        if self.last_trade_date == today and self.daily_trades >= self.max_daily_trades:
            return False, "Daily trade limit exceeded"
        
        # Reset daily counter if new day
        if self.last_trade_date != today:
            self.daily_trades = 0
            self.last_trade_date = today
        
        return True, "Trading allowed"
    
    def log_trade(self, trade_data):
        """
        Log trade for performance tracking
        """
        self.daily_trades += 1
        # Additional logging logic can be added here

class PerformanceTracker:
    def __init__(self, db_path="trade_logs.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """
        Initialize SQLite database for trade logging
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create trades table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                entry_time TEXT NOT NULL,
                exit_time TEXT,
                trade_type TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                quantity REAL NOT NULL,
                profit_loss REAL,
                duration_seconds INTEGER,
                max_drawdown REAL,
                max_profit REAL,
                model_confidence REAL NOT NULL,
                signal_id TEXT
            )
        ''')
        
        # Create daily summary table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_summary (
                date TEXT PRIMARY KEY,
                total_profit REAL NOT NULL,
                num_trades INTEGER NOT NULL,
                winning_trades INTEGER NOT NULL,
                losing_trades INTEGER NOT NULL,
                daily_drawdown REAL,
                sharpe_ratio REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def log_trade_entry(self, symbol, trade_type, entry_price, quantity, model_confidence, signal_id):
        """
        Log trade entry
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO trades (symbol, entry_time, trade_type, entry_price, quantity, model_confidence, signal_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (symbol, datetime.now().isoformat(), trade_type, entry_price, quantity, model_confidence, signal_id))
        
        trade_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return trade_id
    
    def log_trade_exit(self, trade_id, exit_price, profit_loss):
        """
        Log trade exit
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get entry time to calculate duration
        cursor.execute('SELECT entry_time FROM trades WHERE trade_id = ?', (trade_id,))
        entry_time_str = cursor.fetchone()[0]
        entry_time = datetime.fromisoformat(entry_time_str)
        exit_time = datetime.now()
        duration = (exit_time - entry_time).total_seconds()
        
        cursor.execute('''
            UPDATE trades 
            SET exit_time = ?, exit_price = ?, profit_loss = ?, duration_seconds = ?
            WHERE trade_id = ?
        ''', (exit_time.isoformat(), exit_price, profit_loss, duration, trade_id))
        
        conn.commit()
        conn.close()
    
    def calculate_performance_metrics(self, days=30):
        """
        Calculate performance metrics for the last N days
        """
        conn = sqlite3.connect(self.db_path)
        
        # Get trades from last N days
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        df = pd.read_sql_query('''
            SELECT * FROM trades 
            WHERE entry_time >= ? AND exit_time IS NOT NULL
        ''', conn, params=(cutoff_date,))
        
        conn.close()
        
        if df.empty:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0,
                "total_pnl": 0,
                "avg_profit_per_trade": 0,
                "sharpe_ratio": 0,
                "max_drawdown": 0
            }
        
        # Calculate metrics
        total_trades = len(df)
        winning_trades = len(df[df['profit_loss'] > 0])
        losing_trades = len(df[df['profit_loss'] <= 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        total_pnl = df['profit_loss'].sum()
        avg_profit_per_trade = total_pnl / total_trades if total_trades > 0 else 0
        
        # Calculate Sharpe ratio (simplified)
        if len(df) > 1:
            returns = df['profit_loss'] / df['entry_price'] / df['quantity']
            sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Calculate max drawdown
        cumulative_pnl = df['profit_loss'].cumsum()
        peak = cumulative_pnl.expanding().max()
        drawdown = (peak - cumulative_pnl) / peak.abs()
        max_drawdown = drawdown.max() if not drawdown.empty else 0
        
        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "avg_profit_per_trade": avg_profit_per_trade,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown
        }

if __name__ == "__main__":
    # Test the risk management system
    risk_manager = RiskManager(initial_capital=100000)
    performance_tracker = PerformanceTracker()
    
    # Test position sizing
    entry_price = 1000
    stop_loss = risk_manager.calculate_stop_loss(entry_price, 'buy')
    take_profit = risk_manager.calculate_take_profit(entry_price, stop_loss, 'buy')
    position_size = risk_manager.calculate_position_size(entry_price, stop_loss, 0.85)
    
    print(f"Entry Price: ${entry_price}")
    print(f"Stop Loss: ${stop_loss:.2f}")
    print(f"Take Profit: ${take_profit:.2f}")
    print(f"Position Size: {position_size:.2f}")
    
    # Test trade logging
    trade_id = performance_tracker.log_trade_entry(
        symbol="ES", 
        trade_type="Long", 
        entry_price=entry_price, 
        quantity=position_size, 
        model_confidence=0.85, 
        signal_id="signal_001"
    )
    
    print(f"Logged trade with ID: {trade_id}")
    
    # Simulate trade exit
    exit_price = 1020
    profit_loss = (exit_price - entry_price) * position_size
    performance_tracker.log_trade_exit(trade_id, exit_price, profit_loss)
    
    # Calculate performance metrics
    metrics = performance_tracker.calculate_performance_metrics()
    print("Performance Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

