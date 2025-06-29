import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.utils import to_categorical

# Example: Simple character-level language model
text = "hello world. this is a simple language modeling with lstm."

chars = sorted(list(set(text)))
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}

seq_length = 10
X, y = [], []
for i in range(len(text) - seq_length):
    X.append([char_to_idx[c] for c in text[i:i+seq_length]])
    y.append(char_to_idx[text[i+seq_length]])
X = np.array(X)
y = to_categorical(y, num_classes=len(chars))

model = Sequential([
    Embedding(input_dim=len(chars), output_dim=10, input_length=seq_length),
    LSTM(32),
    Dense(len(chars), activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit(X, y, epochs=10, batch_size=8, verbose=1)

# Generate text
seed = "hello wor"
for _ in range(50):
    x_pred = np.array([[char_to_idx[c] for c in seed[-seq_length:]]])
    preds = model.predict(x_pred, verbose=0)[0]
    next_char = idx_to_char[np.argmax(preds)]
    seed += next_char
print("Generated text:")
print(seed)
