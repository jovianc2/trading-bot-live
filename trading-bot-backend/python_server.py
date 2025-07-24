import socket
import json
import threading
import time
import pandas as pd
import numpy as np
from ai_model import preprocess_data, train_model, predict_signal, get_confidence_score

class TradingSignalServer:
    def __init__(self, host='localhost', port=8888):
        self.host = host
        self.port = port
        self.socket = None
        self.running = False
        self.model = None
        self.data_scaler = None
        self.features = None
        
    def start_server(self):
        """Start the TCP server to listen for connections from NinjaTrader"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            self.socket.bind((self.host, self.port))
            self.socket.listen(5)
            self.running = True
            print(f"Trading Signal Server started on {self.host}:{self.port}")
            
            while self.running:
                try:
                    client_socket, address = self.socket.accept()
                    print(f"Connection from {address}")
                    
                    # Handle client in a separate thread
                    client_thread = threading.Thread(
                        target=self.handle_client, 
                        args=(client_socket,)
                    )
                    client_thread.daemon = True
                    client_thread.start()
                    
                except socket.error as e:
                    if self.running:
                        print(f"Socket error: {e}")
                    break
                    
        except Exception as e:
            print(f"Server error: {e}")
        finally:
            if self.socket:
                self.socket.close()
    
    def handle_client(self, client_socket):
        """Handle individual client connections"""
        try:
            while self.running:
                # Receive data from NinjaTrader
                data = client_socket.recv(1024).decode('utf-8')
                if not data:
                    break
                
                try:
                    # Parse JSON data from NinjaTrader
                    market_data = json.loads(data)
                    print(f"Received market data: {market_data}")
                    
                    # Process the data and generate signal
                    signal_response = self.process_market_data(market_data)
                    
                    # Send response back to NinjaTrader
                    response_json = json.dumps(signal_response)
                    client_socket.send(response_json.encode('utf-8'))
                    
                except json.JSONDecodeError:
                    print("Invalid JSON received")
                    error_response = {"error": "Invalid JSON format"}
                    client_socket.send(json.dumps(error_response).encode('utf-8'))
                except Exception as e:
                    print(f"Error processing data: {e}")
                    error_response = {"error": str(e)}
                    client_socket.send(json.dumps(error_response).encode('utf-8'))
                    
        except Exception as e:
            print(f"Client handling error: {e}")
        finally:
            client_socket.close()
    
    def process_market_data(self, market_data):
        """Process market data and generate trading signal"""
        try:
            # Extract OHLCV data from the market_data
            # Expected format: {"open": 1000, "high": 1010, "low": 990, "close": 1005, "volume": 1000000}
            
            if self.model is None:
                return {
                    "signal": "hold",
                    "confidence": 0.0,
                    "error": "Model not loaded"
                }
            
            # Create a DataFrame from the current market data
            # In a real scenario, you'd maintain a rolling window of recent data
            current_data = pd.DataFrame([{
                'Open': market_data.get('open', 0),
                'High': market_data.get('high', 0),
                'Low': market_data.get('low', 0),
                'Close': market_data.get('close', 0),
                'Volume': market_data.get('volume', 0)
            }])
            
            # For demonstration, we'll use dummy technical indicators
            # In practice, you'd calculate these from a rolling window of historical data
            current_data['RSI'] = 50  # Dummy RSI
            current_data['MACD'] = 0  # Dummy MACD
            current_data['Signal_Line'] = 0  # Dummy Signal Line
            current_data['SMA_20'] = current_data['Close']  # Dummy SMA
            current_data['Upper_Band'] = current_data['Close'] * 1.02  # Dummy Upper Band
            current_data['Lower_Band'] = current_data['Close'] * 0.98  # Dummy Lower Band
            current_data['SMA_50'] = current_data['Close']  # Dummy SMA 50
            current_data['SMA_200'] = current_data['Close']  # Dummy SMA 200
            
            # Generate signal using the AI model
            signal, confidence = predict_signal(
                self.model, 
                current_data[self.features], 
                self.data_scaler, 
                self.features
            )
            
            return {
                "signal": signal,
                "confidence": float(confidence),
                "timestamp": time.time()
            }
            
        except Exception as e:
            return {
                "signal": "hold",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def load_model(self, model_path=None):
        """Load the trained AI model"""
        try:
            # For demonstration, we'll train a simple model
            # In practice, you'd load a pre-trained model from disk
            print("Loading/Training AI model...")
            
            # Create dummy training data
            dummy_data = {
                'Date': pd.to_datetime(pd.date_range(start='2023-01-01', periods=300, freq='D')),
                'Open': np.random.rand(300) * 100 + 1000,
                'High': np.random.rand(300) * 100 + 1050,
                'Low': np.random.rand(300) * 100 + 950,
                'Close': np.random.rand(300) * 100 + 1000,
                'Volume': np.random.rand(300) * 1000000
            }
            dummy_df = pd.DataFrame(dummy_data)
            dummy_df.to_csv('training_data.csv', index=False)
            
            # Preprocess and train model
            preprocessed_df, self.data_scaler, self.features = preprocess_data('training_data.csv')
            self.model = train_model(preprocessed_df.copy(), self.features)
            
            print("AI model loaded successfully")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def stop_server(self):
        """Stop the server"""
        self.running = False
        if self.socket:
            self.socket.close()

if __name__ == "__main__":
    # Create and start the trading signal server
    server = TradingSignalServer(host='0.0.0.0', port=8888)
    
    # Load the AI model
    if server.load_model():
        print("Starting trading signal server...")
        try:
            server.start_server()
        except KeyboardInterrupt:
            print("\nShutting down server...")
            server.stop_server()
    else:
        print("Failed to load AI model. Exiting.")

