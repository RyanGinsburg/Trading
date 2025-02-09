import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Check GPU availability
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Set memory growth to avoid memory allocation issues
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth set on GPU")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU detected. TensorFlow will use CPU.")

# Explicitly set the device to GPU if available
device = "/GPU:0" if gpus else "/CPU:0"

with tf.device(device):
    # Define a simple LSTM model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(10, 1)),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Generate random training data
    X_train = np.random.rand(1000, 10, 1)
    y_train = np.random.rand(1000)

    print("Training model on:", device)
    
    # Train the model
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=1)

print("Test completed.")
