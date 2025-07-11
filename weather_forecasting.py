import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Example: Generate synthetic daily temperature data
def load_data():
    np.random.seed(1)
    data = 20 + 10 * np.sin(np.linspace(0, 20, 1000)) + np.random.randn(1000)
    df = pd.DataFrame(data, columns=['Temp'])
    return df

df = load_data()
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['Temp']])

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:i+seq_length])
        ys.append(data[i+seq_length])
    return np.array(xs), np.array(ys)

SEQ_LEN = 30
X, y = create_sequences(scaled_data, SEQ_LEN)

model = Sequential([
    LSTM(32, input_shape=(SEQ_LEN, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=5, batch_size=16, verbose=1)

preds = model.predict(X)
plt.plot(scaler.inverse_transform(y.reshape(-1,1)), label='Actual')
plt.plot(scaler.inverse_transform(preds), label='Predicted')
plt.legend()
plt.title('Weather Forecasting')
plt.show()
