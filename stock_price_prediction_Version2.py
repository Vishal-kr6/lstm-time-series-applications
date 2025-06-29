import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

# Example: Generate synthetic stock price data
def load_data():
    np.random.seed(0)
    data = np.cumsum(np.random.randn(1000)) + 100
    df = pd.DataFrame(data, columns=['Close'])
    return df

df = load_data()
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['Close']])

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:i+seq_length])
        ys.append(data[i+seq_length])
    return np.array(xs), np.array(ys)

SEQ_LEN = 60
X, y = create_sequences(scaled_data, SEQ_LEN)

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(SEQ_LEN, 1)),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=5, batch_size=32, verbose=1)

preds = model.predict(X)
plt.plot(scaler.inverse_transform(y.reshape(-1,1)), label='Actual')
plt.plot(scaler.inverse_transform(preds), label='Predicted')
plt.legend()
plt.title('Stock Price Prediction')
plt.show()