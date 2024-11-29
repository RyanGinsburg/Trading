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
        X.append(data[i:(i + time_step), :])
        y.append(data[i + time_step, 0])  # Predicting future price (first column assumed to be 'close')
    return np.array(X), np.array(y)

def fetch_stock_data(stock, start_date='2019-10-14 00:00:00', end_date=datetime.now(), technical_indicators=['rsi']):
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
future_days = 100

# Download stock data
end_date = datetime.now() - timedelta(days=270)
future_date = datetime.now()
start_date = end_date - timedelta(days=3000)
technical_indicators = ['ema', 'sma', 'williams', 'rsi', 'adx']
stock_data = fetch_stock_data(stock, start_date, end_date, technical_indicators=technical_indicators)

# Prepare data - Separate technical indicators and 'close' price
stock_data['close'] = yf.download(stock, start=start_date, end=end_date)['Close']

# Create multiple datasets based on technical indicators alone, and combined with 'close' price
datasets = {}
scalers = {}
for name in ['close'] + technical_indicators + ['combined']:
    if name == 'combined':
        data = stock_data[['close'] + technical_indicators].dropna().values
    else:
        data = stock_data[[name]].dropna().values
    scalers[name] = MinMaxScaler(feature_range=(0, 1))
    datasets[name] = scalers[name].fit_transform(data)

# Print shapes of datasets
for name, dataset in datasets.items():
    print(f"Shape of {name} dataset: {dataset.shape}")

# Define function to build and train LSTM models
def build_and_train_model(data, time_step):
    X, y = create_dataset(data, time_step)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], data.shape[1])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], data.shape[1])
    
    model = Sequential()
    model.add(LSTM(units=30, return_sequences=True, input_shape=(time_step, data.shape[1]), kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.3))
    model.add(LSTM(units=30, return_sequences=False, kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.3))
    model.add(Dense(units=15))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32, verbose=1, callbacks=[early_stop])
    
    return model, X_test, y_test

# Train models for each dataset
models = {}
for name, dataset in datasets.items():
    print(f"Training model for: {name}")
    models[name] = build_and_train_model(dataset, time_step)

# Define prediction functions for each model
def predict_future(model, data, scaler, future_days):
    last_sequence = data[-time_step:]
    future_predictions = []
    current_batch = last_sequence.reshape((1, time_step, data.shape[1]))
    
    for i in range(future_days):
        future_pred = model.predict(current_batch, verbose=0)[0]
        future_predictions.append(future_pred[0])  # Extract the scalar value
        
        # Reshape future_pred to match the number of features
        future_pred_reshaped = np.zeros((1, 1, data.shape[1]))
        future_pred_reshaped[0, 0, 0] = future_pred[0]  # Use the scalar value
        
        current_batch = np.append(current_batch[:, 1:, :], future_pred_reshaped, axis=1)
    
    future_predictions = np.array(future_predictions)

    # Adjust for inverse transform: handle multi-feature scenarios
    if data.shape[1] > 1:
        # If using multiple features, expand future_predictions to match input shape expected by scaler
        future_predictions_expanded = np.zeros((future_predictions.shape[0], data.shape[1]))
        future_predictions_expanded[:, 0] = future_predictions  # Place predicted values in the first column
        inverse_transformed = scaler.inverse_transform(future_predictions_expanded)
        return inverse_transformed[:, 0]  # Return only the 'close' values
    else:
        # For single-feature models, expand to 2D for consistency and inverse transform
        return scaler.inverse_transform(future_predictions.reshape(-1, 1)).flatten()

# Evaluate each model and make predictions
future_predictions = {}
for name, (model, _, _) in models.items():
    future_predictions[name] = predict_future(model, datasets[name], scalers[name], future_days)

# Adjust all predictions to start from the last known price for consistency
last_price = stock_data['close'].values[-1]
for name, preds in future_predictions.items():
    start_diff = preds[0] - last_price
    future_predictions[name] -= start_diff

# Generate future dates
last_date = stock_data.index[-1]
if not isinstance(last_date, pd.Timestamp):
    last_date = pd.to_datetime(last_date)
future_dates = [last_date + timedelta(days=x + 1) for x in range(future_days)]

# Plotting the results
plt.figure(figsize=(15, 7))
plt.plot(stock_data.index, stock_data['close'].values, label='Historical Data')
future_data = fetch_stock_data(stock, last_date, future_date)

if not future_data.empty:
    plt.plot(future_data.index, future_data['close'].values, label='Actual Future Data', color='green')

# Convert future_dates to datetime
future_dates = pd.to_datetime(future_dates)

# Plot predictions for each model
for name, preds in future_predictions.items():
    plt.plot(future_dates, preds, label=f'Future Predictions ({name})', linestyle='--', alpha=0.7)

plt.title(f'{stock} Stock Price Prediction - Comparison of Technical Indicators')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()

plt.axvline(x=last_date, color='red', linestyle='--', alpha=0.5)
plt.text(last_date, plt.ylim()[0], 'Prediction Start', rotation=90, verticalalignment='bottom')

plt.show()