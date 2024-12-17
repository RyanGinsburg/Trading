import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import LSTM, Dense, Dropout  # type: ignore
from keras.callbacks import EarlyStopping  # type: ignore
from keras.optimizers import Adam # type: ignore
from datetime import datetime, timedelta
import requests
import pandas as pd
import optuna

api_token = 'lFVm52EqS8EuypuH9FqhzhMAbo7zbeNb'

# Utility Functions
def create_dataset(data, time_step):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

def build_lstm(input_shape, units=50, dropout_rate=0.2, learning_rate=0.001):
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout_rate),
        LSTM(units),
        Dropout(dropout_rate),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
    return model

def fetch_technical_data(stock, start_date, end_date, technical_indicators):
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
    return combined_df.loc[pd.to_datetime(start_date):pd.to_datetime(end_date)]

def optimize_model(data, train_split=0.7):
    def objective(trial):
        time_step = trial.suggest_int('time_step', 1, 30)
        units = trial.suggest_int('units', 30, 100)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)

        X, y = create_dataset(data, time_step)
        train_size = int(len(X) * train_split)
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]

        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)

        model = build_lstm(X_train.shape[1:], units, dropout_rate, learning_rate)
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32, verbose=0, callbacks=[early_stop])

        val_loss = model.evaluate(X_val, y_val, verbose=0)
        return val_loss

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=30)
    return study.best_params

# Prediction Functions
def future_predictions_method_1(model, X_test, scaler, future_days):
    future_predictions = model.predict(X_test[-future_days:])
    return scaler.inverse_transform(future_predictions)

def future_predictions_method_2(model, last_sequence, future_days):
    predictions = []
    current_batch = last_sequence.reshape((1, last_sequence.shape[0], 1))
    for _ in range(future_days):
        pred = model.predict(current_batch, verbose=0)[0]
        predictions.append(pred)
        current_batch = np.append(current_batch[:, 1:, :], [[pred]], axis=1)
    return np.array(predictions)

def plot_predictions(stock_data, future_dates, methods, labels, last_date, future_data, stock):
    plt.figure(figsize=(15, 7))
    plt.plot(stock_data.index, stock_data['close'].values, label='Historical Data')

    if not future_data.empty:
        plt.plot(future_data.index, future_data['close'].values, label='Actual Future Data', color='green')

    for smoothed, label in zip(methods, labels):
        plt.plot(future_dates, smoothed, label=label, linestyle='--', alpha=0.7)

    plt.axvline(x=last_date, color='red', linestyle='--', alpha=0.5)
    plt.text(last_date, plt.ylim()[0], 'Prediction Start', rotation=90, verticalalignment='bottom')
    plt.title(f'{stock} Stock Price Prediction - Comparison of Methods')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.show()

# Main Workflow
def main():
    stock = 'tsla'
    future_days = 20
    end_date = datetime.now() - timedelta(days=270)
    start_date = end_date - timedelta(days=3000)
    technical_indicators = ['ema', 'sma', 'williams', 'rsi', 'adx']

    stock_data = fetch_technical_data(stock, start_date, end_date, technical_indicators)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(stock_data['close'].values.reshape(-1, 1))

    best_params = optimize_model(scaled_data)
    time_step = best_params['time_step']

    X, y = create_dataset(scaled_data, time_step)
    train_size = int(len(X) * 0.7)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    model = build_lstm(X_train.shape[1:], units=best_params['units'], dropout_rate=best_params['dropout_rate'], learning_rate=best_params['learning_rate'])
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32, callbacks=[early_stop])

    # Method 1
    future_prices_method_1 = future_predictions_method_1(model, X_test, scaler, future_days)

    # Method 2
    last_sequence = scaled_data[-time_step:]
    future_predictions = future_predictions_method_2(model, last_sequence, future_days)
    future_prices_method_2 = scaler.inverse_transform(future_predictions)

    last_price = stock_data['close'].values[-1]
    future_prices_method_2 -= future_prices_method_2[0] - last_price
    future_prices_method_1 -= future_prices_method_1[0] - last_price

    future_dates = [stock_data.index[-1] + timedelta(days=i + 1) for i in range(future_days)]
    plot_predictions(stock_data, future_dates, [future_prices_method_1, future_prices_method_2], ['Method 1', 'Method 2'], stock_data.index[-1], pd.DataFrame(), stock)

if __name__ == "__main__":
    main()
