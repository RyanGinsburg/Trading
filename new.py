import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import LSTM, Dense, Dropout #type: ignore
from keras.callbacks import EarlyStopping #type: ignore
from keras.regularizers import l2 #type: ignore
from datetime import datetime, timedelta
import requests
import pandas as pd

def create_dataset(data, time_step):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

def data(stock, start_date='2019-10-14 00:00:00', end_date=datetime.now(), technical_indicators=['rsi']):
    df_list = []
    for i, indicator in enumerate(technical_indicators):
        url = f'https://financialmodelingprep.com/api/v3/technical_indicator/1day/{stock}?type={indicator}&period=1300&apikey={api_token}'
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        if i > 0:
            df = df[[indicator]].sort_index()
        else:
            df = df.sort_index()
        df_list.append(df)
    combined_df = pd.concat(df_list, axis=1)
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    df_filtered = combined_df.loc[start_date:end_date]
    return df_filtered

api_token = 'lFVm52EqS8EuypuH9FqhzhMAbo7zbeNb'

# Parameters
stock = 'msft'
time_step = 15
future_days = 20
simualted_days = 30

# Download stock data
end_date = datetime.now() - timedelta(days=270)
future_date = datetime.now()
start_date = end_date - timedelta(days=3000)
stock_data = data(stock, start_date, end_date, technical_indicators=['ema', 'sma', 'williams', 'rsi', 'adx'])

# Prepare data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(stock_data['close'].values.reshape(-1, 1))

# Create dataset
X, y = create_dataset(scaled_data, time_step)

# Split data into train and test
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape input for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build LSTM model with Dropout and Regularization
model = Sequential()
model.add(LSTM(units=30, return_sequences=True, input_shape=(time_step, 1), kernel_regularizer=l2(0.01)))
model.add(Dropout(0.3))  # Dropout rate of 30%
model.add(LSTM(units=30, return_sequences=False, kernel_regularizer=l2(0.01)))
model.add(Dropout(0.3))  # Dropout rate of 30%
model.add(Dense(units=15))
model.add(Dense(units=1))

# Compile and train the model with Early Stopping
model.compile(optimizer='adam', loss='mean_squared_error')
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), 
                    epochs=50, batch_size=32, verbose=1, callbacks=[early_stop])

# Make predictions on test data
test_predictions = model.predict(X_test)
test_predictions = scaler.inverse_transform(test_predictions)
y_test_transformed = scaler.inverse_transform([y_test]).T

# Calculate RMSE
rmse = np.sqrt(np.mean((test_predictions - y_test_transformed) ** 2))
print(f'\nModel RMSE on test data: ${rmse:.2f}')

# Future Prediction Methods

# Method 1: Future predictions using test data
future_predictions_method1 = model.predict(X_test[-future_days:])
future_prices_method1 = scaler.inverse_transform(future_predictions_method1)

# Method 2: Iterative future predictions (based on the last sequence of historical data)
last_sequence = scaled_data[-time_step:]
future_predictions_method2 = []
current_batch = last_sequence.reshape((1, time_step, 1))

for i in range(future_days):
    future_pred = model.predict(current_batch, verbose=0)[0]
    future_predictions_method2.append(future_pred)
    current_batch = np.append(current_batch[:, 1:, :], [[future_pred]], axis=1)

future_predictions_method2 = np.array(future_predictions_method2)
future_prices_method2 = scaler.inverse_transform(future_predictions_method2)

# Method 3: Combination of Method 1 and Method 2 (Average)
future_prices_method3 = (future_prices_method1 + future_prices_method2) / 2

# Method 4: Iterative predictions based on Method 1's output
last_sequence_method1 = scaled_data[-(time_step - future_days):]
combined_method1 = np.concatenate((last_sequence_method1, scaler.transform(future_prices_method1).reshape(-1, 1)))

future_predictions_method4 = []
current_batch = combined_method1[:time_step].reshape((1, time_step, 1))

for i in range(future_days):
    future_pred = model.predict(current_batch, verbose=0)[0]
    future_predictions_method4.append(future_pred)
    current_batch = np.append(current_batch[:, 1:, :], [[future_pred]], axis=1)

future_predictions_method4 = np.array(future_predictions_method4)
future_prices_method4 = scaler.inverse_transform(future_predictions_method4)

# Method 5: Iterative predictions based on Method 3's output
last_sequence_method3 = scaled_data[-(time_step - future_days):]
combined_method3 = np.concatenate((last_sequence_method3, scaler.transform(future_prices_method3).reshape(-1, 1)))

future_predictions_method5 = []
current_batch = combined_method3[:time_step].reshape((1, time_step, 1))

for i in range(future_days):
    future_pred = model.predict(current_batch, verbose=0)[0]
    future_predictions_method5.append(future_pred)
    current_batch = np.append(current_batch[:, 1:, :], [[future_pred]], axis=1)

future_predictions_method5 = np.array(future_predictions_method5)
future_prices_method5 = scaler.inverse_transform(future_predictions_method5)

# Method 6: Combination of Method 1 and Method 2 (Weighted Average)
future_prices_method6 = (future_prices_method1 * 3 + future_prices_method2 * 2) / 5

# Adjust all predictions to start from the last known price
last_price = stock_data['close'].values[-1]
methods = [
    future_prices_method1, future_prices_method2, future_prices_method3,
    future_prices_method4, future_prices_method5, future_prices_method6
]

for idx, method in enumerate(methods):
    start_diff = method[0] - last_price
    methods[idx] = method - start_diff

# Smoothing the predictions using a simple moving average
smoothed_methods = [pd.Series(m.flatten()).rolling(window=3, min_periods=1).mean().values for m in methods]

# Generate future dates
last_date = stock_data.index[-1]
if not isinstance(last_date, pd.Timestamp):
    last_date = pd.to_datetime(last_date)
future_dates = [last_date + timedelta(days=x + 1) for x in range(future_days)]

# Plotting
plt.figure(figsize=(15, 7))
plt.plot(stock_data.index, stock_data['close'].values, label='Historical Data')

# Plot actual future data if available
future_data = data(stock, last_date, future_date)
if not future_data.empty:
    plt.plot(future_data.index, future_data['close'].values, label='Actual Future Data', color='green')

# Convert future_dates to datetime
future_dates = pd.to_datetime(future_dates)

# Plot future prediction methods
labels = [
    'Future Predictions (Method 1)', 'Future Predictions (Method 2)', 
    'Future Predictions (Method 3)', 'Future Predictions (Method 4)', 
    'Future Predictions (Method 5)', 'Future Predictions (Method 6)'
]

for smoothed, label in zip(methods, labels):
    print("First 5 future prices for Method 1:", smoothed[:10])
    plt.plot(future_dates, smoothed, label=label, linestyle='--', alpha=0.7)

plt.title(f'{stock} Stock Price Prediction - Comparison of Methods (Starting from Last Known Price)')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()

# Add vertical line to separate historical data and future predictions
plt.axvline(x=last_date, color='red', linestyle='--', alpha=0.5)
plt.text(last_date, plt.ylim()[0], 'Prediction Start', rotation=90, verticalalignment='bottom')

plt.show()
