import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def preprocess_data(data_path):
    """
    Preprocesses OHLCV data for AI model training.
    Calculates technical indicators and scales data.
    """
    df = pd.read_csv(data_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)

    # Calculate Technical Indicators
    # RSI
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["StdDev"] = df["Close"].rolling(window=20).std()
    df["Upper_Band"] = df["SMA_20"] + (df["StdDev"] * 2)
    df["Lower_Band"] = df["SMA_20"] - (df["StdDev"] * 2)

    # Price Action Patterns (simplified example)
    df["Candle_Range"] = df["High"] - df["Low"]
    df["Body_Range"] = abs(df["Open"] - df["Close"])

    # Additional SMAs
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["SMA_200"] = df["Close"].rolling(window=200).mean()

    # Drop NaN values created by indicators
    df.dropna(inplace=True)

    # Select features for the model
    features = [
        "Open", "High", "Low", "Close", "Volume",
        "RSI", "MACD", "Signal_Line",
        "SMA_20", "Upper_Band", "Lower_Band",
        "Candle_Range", "Body_Range",
        "SMA_50", "SMA_200"
    ]
    
    # Handle cases where features might not be present due to insufficient data
    available_features = [f for f in features if f in df.columns]
    if not available_features:
        raise ValueError("No valid features found after preprocessing. Check data and indicator calculations.")

    # Scale features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[available_features])

    return scaled_data, scaler, available_features

def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length), :])
        y.append(data[i + sequence_length, 3])  # Predict 'Close' price of next step
    return np.array(X), np.array(y)

def train_model(data_path, features, sequence_length=60):
    """
    Trains an LSTM model for trading signal generation.
    """
    scaled_data, scaler, available_features = preprocess_data(data_path)
    
    if len(scaled_data) < sequence_length + 1:
        raise ValueError(f"Insufficient data for sequence creation. Need at least {sequence_length + 1} data points.")

    X, y = create_sequences(scaled_data, sequence_length)

    # Reshape y for scaling (if needed, for inverse transform later)
    y = y.reshape(-1, 1)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build LSTM model
    model = Sequential([
        LSTM(50, activation="relu", input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")

    # Train model
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=0)

    return model

def predict_signal(model, current_data, scaler, features, sequence_length=60):
    """
    Generates a trading signal (buy/sell/hold) and confidence score.
    """
    # Ensure current_data is a DataFrame and has the correct features
    if not isinstance(current_data, pd.DataFrame):
        current_data = pd.DataFrame([current_data])

    # Reindex current_data to ensure all expected features are present
    # Fill missing features with 0 or a sensible default if they weren't in the original training data
    for f in features:
        if f not in current_data.columns:
            current_data[f] = 0  # Or a more appropriate default

    # Ensure the order of columns matches the training features
    current_data = current_data[features]

    # Scale the current data
    scaled_current_data = scaler.transform(current_data)

    # Create sequence for prediction
    # This assumes `current_data` is the latest data point and we need `sequence_length-1` previous points
    # For a real-time system, you'd maintain a rolling window of data.
    # For this simulation, we'll just use the single point for simplicity, which might not be ideal for LSTM.
    # A more robust solution would involve passing a sequence of recent data.
    
    # For now, let's create a dummy sequence if current_data is just one point
    if scaled_current_data.shape[0] < sequence_length:
        # Pad with zeros or replicate the single data point to meet sequence_length
        padding_needed = sequence_length - scaled_current_data.shape[0]
        padded_data = np.vstack([np.zeros((padding_needed, scaled_current_data.shape[1])), scaled_current_data])
        prediction_input = padded_data.reshape(1, sequence_length, scaled_current_data.shape[1])
    else:
        prediction_input = scaled_current_data[-sequence_length:].reshape(1, sequence_length, scaled_current_data.shape[1])

    # Predict next close price
    predicted_scaled_price = model.predict(prediction_input, verbose=0)[0][0]
    
    # Inverse transform the predicted price to original scale
    # Create a dummy array with the same number of features as original scaled_data
    dummy_row = np.zeros(len(features))
    dummy_row[features.index("Close")] = predicted_scaled_price # Place predicted price in 'Close' column position
    predicted_price = scaler.inverse_transform(dummy_row.reshape(1, -1))[0][features.index("Close")]

    # Determine signal based on predicted price vs current close price
    current_close_price = current_data["Close"].iloc[-1] # Get the last close price from the original data

    signal = "hold"
    if predicted_price > current_close_price * 1.001:  # If predicted price is significantly higher
        signal = "buy"
    elif predicted_price < current_close_price * 0.999: # If predicted price is significantly lower
        signal = "sell"

    # Confidence score (simplified: based on distance from current price)
    # A more sophisticated confidence would come from model's internal uncertainty or a separate classifier
    price_diff_percent = abs(predicted_price - current_close_price) / current_close_price
    confidence = min(0.5 + (price_diff_percent * 10), 0.99) # Scale difference to confidence, max 0.99
    if signal == "hold":
        confidence = 0.5 # Low confidence for hold signals

    return signal, confidence

def get_confidence_score(prediction_probability):
    """
    Placeholder for a more sophisticated confidence scoring mechanism.
    For now, it's a direct mapping or simple calculation.
    """
    # In a real scenario, this might involve:
    # - Analyzing the variance of ensemble predictions
    # - Using a separate confidence model (e.g., a calibrated classifier)
    # - Bayesian neural networks for uncertainty estimation
    return prediction_probability # Assuming prediction_probability is already a confidence score

if __name__ == "__main__":
    # Create a dummy CSV file for testing
    data = {
        "Date": pd.to_datetime(pd.date_range(start="2023-01-01", periods=300, freq="D")),
        "Open": np.random.rand(300) * 100 + 1000,
        "High": np.random.rand(300) * 100 + 1050,
        "Low": np.random.rand(300) * 100 + 950,
        "Close": np.random.rand(300) * 100 + 1000,
        "Volume": np.random.rand(300) * 1000000
    }
    dummy_df = pd.DataFrame(data)
    dummy_df.to_csv("training_data.csv", index=False)

    # Test preprocessing
    try:
        scaled_data, scaler, features = preprocess_data("training_data.csv")
        print("Preprocessing successful.")
        print(f"Scaled data shape: {scaled_data.shape}")
        print(f"Features used: {features}")

        # Test training
        model = train_model("training_data.csv", features)
        print("Model training successful.")

        # Test prediction
        # Use the last 'sequence_length' data points for prediction
        last_data_point = dummy_df.iloc[-1]
        signal, confidence = predict_signal(model, last_data_point, scaler, features)
        print(f"Predicted Signal: {signal}, Confidence: {confidence:.2f}")

    except Exception as e:
        print(f"An error occurred during testing: {e}")



