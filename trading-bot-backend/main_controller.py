#!/usr/bin/env python3
"""
Main Controller for Self-Learning Trading Bot
Integrates all components: AI model, risk management, communication, and retraining
"""

import os
import sys
import time
import threading
import logging
import signal
from datetime import datetime
import pickle

# Import our custom modules
from python_server import TradingSignalServer
from risk_management import RiskManager, PerformanceTracker
from automated_retraining import AutomatedRetrainingPipeline
from ai_model import preprocess_data, train_model

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)

class TradingBotController:
    def __init__(self, config=None):
        """
        Initialize the trading bot controller with all components
        """
        self.config = config or self.get_default_config()
        self.running = False
        
        # Initialize components
        self.signal_server = None
        self.risk_manager = None
        self.performance_tracker = None
        self.retraining_pipeline = None
        
        # Threading
        self.server_thread = None
        self.retraining_thread = None
        
        # Model state
        self.current_model = None
        self.current_scaler = None
        self.current_features = None
        
        logging.info("Trading Bot Controller initialized")
    
    def get_default_config(self):
        """
        Get default configuration for the trading bot
        """
        return {
            'server': {
                'host': '0.0.0.0',
                'port': 8888
            },
            'risk_management': {
                'initial_capital': 100000,
                'max_drawdown': 0.10,
                'max_risk_per_trade': 0.02,
                'confidence_threshold': 0.75
            },
            'retraining': {
                'interval_days': 3,
                'performance_threshold': 0.65,
                'min_data_points': 100
            },
            'trading': {
                'symbols': ['ES', 'ETH/USDT'],
                'max_daily_trades': 10,
                'risk_reward_ratio': 2.0
            }
        }
    
    def initialize_components(self):
        """
        Initialize all trading bot components
        """
        try:
            logging.info("Initializing trading bot components...")
            
            # Initialize Risk Manager
            self.risk_manager = RiskManager(
                initial_capital=self.config['risk_management']['initial_capital'],
                max_drawdown=self.config['risk_management']['max_drawdown'],
                max_risk_per_trade=self.config['risk_management']['max_risk_per_trade']
            )
            logging.info("Risk Manager initialized")
            
            # Initialize Performance Tracker
            self.performance_tracker = PerformanceTracker()
            logging.info("Performance Tracker initialized")
            
            # Initialize Retraining Pipeline
            self.retraining_pipeline = AutomatedRetrainingPipeline(
                retraining_interval_days=self.config['retraining']['interval_days'],
                performance_threshold=self.config['retraining']['performance_threshold']
            )
            logging.info("Retraining Pipeline initialized")
            
            # Initialize Signal Server
            self.signal_server = TradingSignalServer(
                host=self.config['server']['host'],
                port=self.config['server']['port']
            )
            logging.info("Signal Server initialized")
            
            # Load or train initial model
            self.load_or_train_initial_model()
            
            return True
            
        except Exception as e:
            logging.error(f"Error initializing components: {e}")
            return False
    
    def load_or_train_initial_model(self):
        """
        Load existing model or train a new one if none exists
        """
        try:
            # Try to load existing model
            model_path = "models/current_model.pkl"
            scaler_path = "models/current_scaler.pkl"
            features_path = "models/current_features.pkl"
            
            if all(os.path.exists(path) for path in [model_path, scaler_path, features_path]):
                logging.info("Loading existing model...")
                with open(model_path, 'rb') as f:
                    self.current_model = pickle.load(f)
                with open(scaler_path, 'rb') as f:
                    self.current_scaler = pickle.load(f)
                with open(features_path, 'rb') as f:
                    self.current_features = pickle.load(f)
                
                # Set model in signal server
                self.signal_server.model = self.current_model
                self.signal_server.data_scaler = self.current_scaler
                self.signal_server.features = self.current_features
                
                logging.info("Existing model loaded successfully")
            else:
                logging.info("No existing model found. Training initial model...")
                if self.retraining_pipeline.retrain_model():
                    # Load the newly trained model
                    self.load_or_train_initial_model()
                else:
                    logging.error("Failed to train initial model")
                    return False
            
            return True
            
        except Exception as e:
            logging.error(f"Error loading/training initial model: {e}")
            return False
    
    def start_signal_server(self):
        """
        Start the signal server in a separate thread
        """
        try:
            logging.info("Starting signal server...")
            self.server_thread = threading.Thread(
                target=self.signal_server.start_server,
                daemon=True
            )
            self.server_thread.start()
            logging.info("Signal server started")
            return True
            
        except Exception as e:
            logging.error(f"Error starting signal server: {e}")
            return False
    
    def start_retraining_pipeline(self):
        """
        Start the automated retraining pipeline in a separate thread
        """
        try:
            logging.info("Starting retraining pipeline...")
            self.retraining_thread = threading.Thread(
                target=self.retraining_pipeline.schedule_retraining,
                daemon=True
            )
            self.retraining_thread.start()
            logging.info("Retraining pipeline started")
            return True
            
        except Exception as e:
            logging.error(f"Error starting retraining pipeline: {e}")
            return False
    
    def monitor_system_health(self):
        """
        Monitor the health of all system components
        """
        while self.running:
            try:
                # Check if server thread is alive
                if self.server_thread and not self.server_thread.is_alive():
                    logging.warning("Signal server thread died. Restarting...")
                    self.start_signal_server()
                
                # Check if retraining thread is alive
                if self.retraining_thread and not self.retraining_thread.is_alive():
                    logging.warning("Retraining thread died. Restarting...")
                    self.start_retraining_pipeline()
                
                # Check risk management status
                can_trade, reason = self.risk_manager.can_trade()
                if not can_trade:
                    logging.warning(f"Trading disabled: {reason}")
                
                # Log performance metrics periodically
                metrics = self.performance_tracker.calculate_performance_metrics(days=1)
                if metrics['total_trades'] > 0:
                    logging.info(f"Daily Performance - Trades: {metrics['total_trades']}, "
                               f"Win Rate: {metrics['win_rate']:.2%}, "
                               f"PnL: ${metrics['total_pnl']:.2f}")
                
                # Sleep for 5 minutes before next health check
                time.sleep(300)
                
            except Exception as e:
                logging.error(f"Error in system health monitoring: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
    
    def start(self):
        """
        Start the complete trading bot system
        """
        try:
            logging.info("Starting Self-Learning Trading Bot...")
            
            # Initialize all components
            if not self.initialize_components():
                logging.error("Failed to initialize components. Exiting.")
                return False
            
            # Start signal server
            if not self.start_signal_server():
                logging.error("Failed to start signal server. Exiting.")
                return False
            
            # Start retraining pipeline
            if not self.start_retraining_pipeline():
                logging.error("Failed to start retraining pipeline. Exiting.")
                return False
            
            # Set running flag
            self.running = True
            
            # Start system health monitoring
            logging.info("Trading Bot started successfully!")
            logging.info("=" * 50)
            logging.info("TRADING BOT STATUS:")
            logging.info(f"Signal Server: Running on {self.config['server']['host']}:{self.config['server']['port']}")
            logging.info(f"Risk Management: Active (Max Drawdown: {self.config['risk_management']['max_drawdown']:.1%})")
            logging.info(f"Auto Retraining: Every {self.config['retraining']['interval_days']} days")
            logging.info(f"Supported Symbols: {', '.join(self.config['trading']['symbols'])}")
            logging.info("=" * 50)
            
            # Start monitoring
            self.monitor_system_health()
            
            return True
            
        except Exception as e:
            logging.error(f"Error starting trading bot: {e}")
            return False
    
    def stop(self):
        """
        Stop the trading bot system gracefully
        """
        try:
            logging.info("Stopping Trading Bot...")
            self.running = False
            
            # Stop signal server
            if self.signal_server:
                self.signal_server.stop_server()
            
            # Stop retraining pipeline
            if self.retraining_pipeline:
                # The retraining pipeline will stop when the main thread exits
                pass
            
            logging.info("Trading Bot stopped successfully")
            
        except Exception as e:
            logging.error(f"Error stopping trading bot: {e}")

def signal_handler(signum, frame):
    """
    Handle system signals for graceful shutdown
    """
    logging.info(f"Received signal {signum}. Shutting down...")
    if 'bot_controller' in globals():
        bot_controller.stop()
    sys.exit(0)

if __name__ == "__main__":
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and start the trading bot
    bot_controller = TradingBotController()
    
    try:
        bot_controller.start()
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received. Shutting down...")
        bot_controller.stop()
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        bot_controller.stop()
        sys.exit(1)

