import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

def fit_lstm(series, window_size=30, epochs=20, batch_size=16):
    """
    Fit an LSTM model on a univariate time series.
    """
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled = scaler.fit_transform(series.values.reshape(-1,1))

    X, y = [], []
    for i in range(len(scaled) - window_size):
        X.append(scaled[i:i+window_size])
        y.append(scaled[i+window_size])
    X, y = np.array(X), np.array(y)

    # Safety check
    if len(X) == 0:
        raise ValueError(
            f"Dataset too short ({len(scaled)} points) for window size {window_size}. "
            "Upload more data or reduce the window size."
        )

    # Reshape for LSTM
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(window_size,1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

    return model, scaler


def forecast_lstm(model, scaler, series, window_size=30, steps=7):
    """
    Forecast future values using a trained LSTM model.
    """
    scaled = scaler.transform(series.values.reshape(-1,1))
    last_seq = scaled[-window_size:]
    preds = []

    for _ in range(steps):
        pred = model.predict(last_seq.reshape(1, window_size, 1), verbose=0)
        preds.append(pred[0][0])
        last_seq = np.vstack([last_seq[1:], pred])

    forecast = scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()
    forecast_index = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=steps, freq="D")

    return forecast, forecast_index