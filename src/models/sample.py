from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential()
model.add(LSTM(10, input_shape=(30,1)))
model.add(Dense(1))
print("✅ LSTM model built successfully")