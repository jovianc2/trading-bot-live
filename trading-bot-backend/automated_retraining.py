import os
import time
import schedule
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import pickle
import shutil
from ai_model import preprocess_data, train_model, predict_signal
from risk_management import PerformanceTracker
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('retraining.log'),
        logging.StreamHandler()
    ]
)

class AutomatedRetrainingPipeline:
    def __init__(self, 
                 data_source_path="paper_trading_data.csv",
                 model_save_path="models/",
                 backup_path="models/backup/",
                 performance_threshold=0.6,
                 retraining_interval_days=3):
        
        self.data_source_path = data_source_path
        self.model_save_path = model_save_path
        self.backup_path = backup_path
        self.performance_threshold = performance_threshold
        self.retraining_interval_days = retraining_interval_days
        
        # Create directories if they don't exist
        os.makedirs(self.model_save_path, exist_ok=True)
        os.makedirs(self.backup_path, exist_ok=True)
        
        self.performance_tracker = PerformanceTracker()
        
    def collect_new_data(self):
        """
        Collect new paper trading data from the last retraining period
        """
        try:
            logging.info("Collecting new paper trading data...")
            
            # Get the last retraining date
            last_retrain_file = os.path.join(self.model_save_path, "last_retrain.txt")
            if os.path.exists(last_retrain_file):
                with open(last_retrain_file, 'r') as f:
                    last_retrain_date = datetime.fromisoformat(f.read().strip())
            else:
                # If no previous retraining, use data from the last 30 days
                last_retrain_date = datetime.now() - timedelta(days=30)
            
            # In a real implementation, this would pull data from NinjaTrader
            # For demonstration, we'll simulate collecting new data
            new_data = self.simulate_paper_trading_data(last_retrain_date)
            
            # Append new data to existing dataset
            if os.path.exists(self.data_source_path):
                existing_data = pd.read_csv(self.data_source_path)
                combined_data = pd.concat([existing_data, new_data], ignore_index=True)
            else:
                combined_data = new_data
            
            # Remove duplicates and sort by date
            combined_data = combined_data.drop_duplicates(subset=['Date']).sort_values('Date')
            combined_data.to_csv(self.data_source_path, index=False)
            
            logging.info(f"Collected {len(new_data)} new data points")
            return True
            
        except Exception as e:
            logging.error(f"Error collecting new data: {e}")
            return False
    
    def simulate_paper_trading_data(self, start_date):
        """
        Simulate paper trading data for demonstration
        In a real implementation, this would interface with NinjaTrader's paper trading data
        """
        end_date = datetime.now()
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate realistic-looking market data
        np.random.seed(int(time.time()))
        base_price = 4500  # ES futures base price
        
        data = []
        current_price = base_price
        
        for date in date_range:
            # Simulate daily price movement
            daily_change = np.random.normal(0, 50)  # Mean 0, std 50 points
            current_price += daily_change
            
            # Generate OHLCV data
            open_price = current_price
            high_price = open_price + abs(np.random.normal(0, 25))
            low_price = open_price - abs(np.random.normal(0, 25))
            close_price = open_price + np.random.normal(0, 20)
            volume = np.random.randint(100000, 1000000)
            
            data.append({
                'Date': date.strftime('%Y-%m-%d'),
                'Open': open_price,
                'High': high_price,
                'Low': low_price,
                'Close': close_price,
                'Volume': volume
            })
            
            current_price = close_price
        
        return pd.DataFrame(data)
    
    def validate_new_model(self, new_model, validation_data, features, scaler):
        """
        Validate the newly trained model against recent performance
        """
        try:
            logging.info("Validating new model...")
            
            # Split validation data
            val_size = int(len(validation_data) * 0.2)
            val_data = validation_data.tail(val_size)
            
            # Generate predictions
            correct_predictions = 0
            total_predictions = 0
            
            for i in range(len(val_data) - 1):
                current_data = val_data.iloc[[i]]
                actual_next_close = val_data.iloc[i + 1]['Close']
                current_close = val_data.iloc[i]['Close']
                
                # Get model prediction
                signal, confidence = predict_signal(new_model, current_data[features], scaler, features)
                
                # Check if prediction was correct
                if signal == 'buy' and actual_next_close > current_close:
                    correct_predictions += 1
                elif signal == 'sell' and actual_next_close < current_close:
                    correct_predictions += 1
                
                total_predictions += 1
            
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            logging.info(f"Model validation accuracy: {accuracy:.2%}")
            
            return accuracy >= self.performance_threshold
            
        except Exception as e:
            logging.error(f"Error validating model: {e}")
            return False
    
    def backup_current_model(self):
        """
        Backup the current model before replacing it
        """
        try:
            model_files = ['current_model.pkl', 'current_scaler.pkl', 'current_features.pkl']
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            for file_name in model_files:
                current_path = os.path.join(self.model_save_path, file_name)
                if os.path.exists(current_path):
                    backup_name = f"{timestamp}_{file_name}"
                    backup_path = os.path.join(self.backup_path, backup_name)
                    shutil.copy2(current_path, backup_path)
            
            logging.info(f"Backed up current model with timestamp {timestamp}")
            return True
            
        except Exception as e:
            logging.error(f"Error backing up model: {e}")
            return False
    
    def save_model(self, model, scaler, features):
        """
        Save the trained model, scaler, and features
        """
        try:
            # Save model
            with open(os.path.join(self.model_save_path, 'current_model.pkl'), 'wb') as f:
                pickle.dump(model, f)
            
            # Save scaler
            with open(os.path.join(self.model_save_path, 'current_scaler.pkl'), 'wb') as f:
                pickle.dump(scaler, f)
            
            # Save features
            with open(os.path.join(self.model_save_path, 'current_features.pkl'), 'wb') as f:
                pickle.dump(features, f)
            
            # Update last retrain timestamp
            with open(os.path.join(self.model_save_path, 'last_retrain.txt'), 'w') as f:
                f.write(datetime.now().isoformat())
            
            logging.info("Model saved successfully")
            return True
            
        except Exception as e:
            logging.error(f"Error saving model: {e}")
            return False
    
    def load_current_model(self):
        """
        Load the current model for comparison
        """
        try:
            model_path = os.path.join(self.model_save_path, 'current_model.pkl')
            scaler_path = os.path.join(self.model_save_path, 'current_scaler.pkl')
            features_path = os.path.join(self.model_save_path, 'current_features.pkl')
            
            if all(os.path.exists(path) for path in [model_path, scaler_path, features_path]):
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                with open(features_path, 'rb') as f:
                    features = pickle.load(f)
                
                return model, scaler, features
            else:
                return None, None, None
                
        except Exception as e:
            logging.error(f"Error loading current model: {e}")
            return None, None, None
    
    def retrain_model(self):
        """
        Main retraining function
        """
        try:
            logging.info("Starting automated model retraining...")
            
            # Step 1: Collect new data
            if not self.collect_new_data():
                logging.error("Failed to collect new data. Aborting retraining.")
                return False
            
            # Step 2: Check if we have enough data
            if not os.path.exists(self.data_source_path):
                logging.error("No data source found. Aborting retraining.")
                return False
            
            # Step 3: Preprocess data
            logging.info("Preprocessing data...")
            preprocessed_data, new_scaler, new_features = preprocess_data(self.data_source_path)
            
            if len(preprocessed_data) < 100:  # Minimum data requirement
                logging.warning("Insufficient data for retraining. Need at least 100 samples.")
                return False
            
            # Step 4: Train new model
            logging.info("Training new model...")
            new_model = train_model(preprocessed_data.copy(), new_features)
            
            # Step 5: Validate new model
            if not self.validate_new_model(new_model, preprocessed_data, new_features, new_scaler):
                logging.warning("New model failed validation. Keeping current model.")
                return False
            
            # Step 6: Compare with current model performance (if exists)
            current_model, current_scaler, current_features = self.load_current_model()
            if current_model is not None:
                # Compare performance metrics
                current_metrics = self.performance_tracker.calculate_performance_metrics(days=7)
                if current_metrics['win_rate'] > 0.7:  # If current model is performing well
                    logging.info("Current model is performing well. Higher threshold for replacement.")
                    # You could implement more sophisticated comparison logic here
            
            # Step 7: Backup current model
            if current_model is not None:
                self.backup_current_model()
            
            # Step 8: Save new model
            if self.save_model(new_model, new_scaler, new_features):
                logging.info("Model retraining completed successfully!")
                return True
            else:
                logging.error("Failed to save new model.")
                return False
                
        except Exception as e:
            logging.error(f"Error during model retraining: {e}")
            return False
    
    def schedule_retraining(self):
        """
        Schedule automatic retraining every N days
        """
        schedule.every(self.retraining_interval_days).days.do(self.retrain_model)
        logging.info(f"Scheduled retraining every {self.retraining_interval_days} days")
        
        while True:
            schedule.run_pending()
            time.sleep(3600)  # Check every hour

if __name__ == "__main__":
    # Initialize the retraining pipeline
    pipeline = AutomatedRetrainingPipeline(
        retraining_interval_days=3,
        performance_threshold=0.65
    )
    
    # Run initial retraining
    logging.info("Running initial model training...")
    pipeline.retrain_model()
    
    # Start scheduled retraining
    logging.info("Starting scheduled retraining...")
    pipeline.schedule_retraining()

