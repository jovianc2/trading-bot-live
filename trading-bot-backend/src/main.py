from flask import Flask, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import threading
import time
import os
import logging

# Set up logging for Flask app
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("flask_app.log"),
        logging.StreamHandler()
    ]
)

app = Flask(__name__, static_folder="../static", template_folder="../templates")
CORS(app)  # Enable CORS for all origins
socketio = SocketIO(app, cors_allowed_origins="*")
@app.route("/")
def index():
    return "✅ Trading Bot Backend is Running!"

# Dummy data for testing
dummy_performance_metrics = {
    "total_trades": 120,
    "winning_trades": 85,
    "losing_trades": 35,
    "win_rate": 0.7083,
    "total_pnl": 15230.50,
    "avg_profit_per_trade": 126.92,
    "sharpe_ratio": 1.25,
    "max_drawdown": 0.05
}

dummy_signals = [
    {"id": 1, "symbol": "ES", "signal": "BUY", "confidence": 0.85, "price": 4500, "timestamp": time.ctime()},
    {"id": 2, "symbol": "ETH/USDT", "signal": "SELL", "confidence": 0.78, "price": 1800, "timestamp": time.ctime()},
]

@app.route("/api/status")
def get_status():
    return jsonify({"status": "Running", "lastUpdate": time.ctime()})

@app.route("/api/performance")
def get_performance():
    return jsonify(dummy_performance_metrics)

@app.route("/api/signals")
def get_signals():
    return jsonify(dummy_signals)

@app.route("/")
def serve_index():
    return send_from_directory(app.static_folder, "index.html")

@socketio.on("connect")
def test_connect():
    logging.info("Client connected")
    emit("response", {"data": "Connected to Flask-SocketIO server"}) 

@socketio.on("disconnect")
def test_disconnect():
    logging.info("Client disconnected")

def emit_realtime_data():
    while True:
        # Emit bot status
        socketio.emit("status_update", {"status": "Running", "lastUpdate": time.ctime()})

        # Emit performance metrics
        socketio.emit("performance_update", dummy_performance_metrics)

        # Emit dummy signals
        socketio.emit("signal_update", {
            "id": time.time(),
            "symbol": "TEST",
            "signal": "HOLD",
            "confidence": 0.5,
            "price": 4500,
            "timestamp": time.ctime()
        })
        
        time.sleep(5) # Emit every 5 seconds

if __name__ == "__main__":
    # Start emitting real-time data in a separate thread
    realtime_thread = threading.Thread(target=emit_realtime_data)
    realtime_thread.daemon = True
    realtime_thread.start()

    socketio.run(app, host="0.0.0.0", port=5000, allow_unsafe_werkzeug=True)
# Print all active routes
for rule in app.url_map.iter_rules():
    print(f"{rule.endpoint}: {rule.rule}")

# Start the app
if __name__ == '__main__':
    socketio.run(app, host="0.0.0.0", port=5000, allow_unsafe_werkzeug=True)
for rule in app.url_map.iter_rules():
    print(f"✅ Registered route: {rule}")
