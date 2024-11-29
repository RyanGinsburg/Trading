import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from datetime import datetime, timedelta
import requests
import pandas as pd
import pandas_market_calendars as mcal

def create_dataset(data, time_step):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

def get_market_days(start_year, end_year, exchange='NYSE'):
    market_calendar = mcal.get_calendar(exchange)
    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-12-31"
    schedule = market_calendar.schedule(start_date=start_date, end_date=end_date)
    market_days = schedule.index.to_list()
    return market_days

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

def adjusted_simple_average(scores):
    return np.mean(scores)

def adjusted_exponential_weighted_average(scores, alpha=0.5):
    n = len(scores)
    weights = [alpha * (1 - alpha) ** (n - 1 - i) for i in range(n)]
    total_weights = sum(weights)
    weighted_sum = sum(score * weight for score, weight in zip(scores, weights))
    return weighted_sum / total_weights

def evaluate_predictions(historical_prices, future_prices, predicted_averages, labels):
    evaluation_results = []
    profits = {label: 0 for label in labels}

    # Ideal scenario profit calculation (with shorting)
    ideal_profit = 0
    for i in range(1, len(future_prices)):
        if future_prices[i] > future_prices[i - 1]:  # Buy if price is increasing
            ideal_profit += future_prices[i] - future_prices[i - 1]
        elif future_prices[i] < future_prices[i - 1]:  # Short if price is decreasing
            ideal_profit += future_prices[i - 1] - future_prices[i]

    # Buy-and-hold profit calculation
    buy_and_hold_profit = future_prices[-1] - future_prices[0]  # Last price - first price

    for i, predictions in enumerate(predicted_averages):
        label = labels[i]

        # Align lengths
        min_length = min(len(future_prices), len(predictions))
        aligned_future_prices = future_prices[:min_length]
        aligned_predictions = predictions[:min_length]

        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(aligned_future_prices, aligned_predictions))
        print(f"RMSE for {label}: {rmse}")

        # Calculate profits/losses (with shorting)
        current_price = historical_prices[-1]  # Last known price
        balance = 0  # Total profit/loss
        for predicted_price, actual_price in zip(aligned_predictions, aligned_future_prices):
            if predicted_price > current_price:  # Long position (Buy low, sell high)
                balance += actual_price - current_price
            elif predicted_price < current_price:  # Short position (Sell high, buy low)
                balance += current_price - actual_price
            current_price = actual_price  # Update current price for next prediction

        profits[label] = balance
        evaluation_results.append((label, rmse, balance))
        print(f"Profit/Loss for {label} (with shorting): ${balance:.2f}")

    print(f"Ideal scenario profit (with shorting): ${ideal_profit:.2f}")
    print(f"Buy-and-hold profit: ${buy_and_hold_profit:.2f}")

    return evaluation_results, ideal_profit, buy_and_hold_profit


api_token = 'lFVm52EqS8EuypuH9FqhzhMAbo7zbeNb'
market_days = get_market_days(2015, 2025)

# Parameters
stock = 'aapl'
time_step = 15
future_days = 5
simulated_days = 30
count = 5

# 3D arrays for predictions
predictions_method1 = {}
predictions_method2 = {}
predictions_method3 = {}

prediction_arrays = [predictions_method1, predictions_method2, predictions_method3]

for predictions in prediction_arrays:
    predictions[stock] = {}

p1simple, p1exponential, p1, p2simple, p2exponential, p2, p3simple, p3exponential, p3 = [], [], [], [], [], [], [], [], []
averages_array  = [p1simple, p1exponential, p1, p2simple, p2exponential, p2, p3simple, p3exponential, p3]
labels  = ['p1simple', 'p1exponential', 'p1', 'p2simple', 'p2exponential', 'p2', 'p3simple', 'p3exponential', 'p3']

end_date = datetime.now() - timedelta(days=270)
last_date = end_date
future_date = datetime.now()
start_date = end_date - timedelta(days=3000)

full_data = data(stock, start_date, future_date, technical_indicators=['ema', 'sma', 'williams', 'rsi', 'adx'])
position = full_data.index.get_indexer([end_date], method="pad")[0]
prev_dates = full_data.index[max(0, position + 1 - count):position]
end_date = prev_dates[0]

for day in range(simulated_days + count - 1):
    while pd.Timestamp(end_date.replace(hour=0, minute=0, second=0, microsecond=0)) not in market_days:
        end_date += timedelta(days=1)
    
    stock_data = full_data.loc[start_date:end_date]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(stock_data['close'].values.reshape(-1, 1))

    X, y = create_dataset(scaled_data, time_step)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    model = Sequential([
        LSTM(30, return_sequences=True, input_shape=(time_step, 1), kernel_regularizer=l2(0.01)),
        Dropout(0.3),
        LSTM(30, kernel_regularizer=l2(0.01)),
        Dropout(0.3),
        Dense(15),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32, callbacks=[early_stop])

    future_prices_method1 = scaler.inverse_transform(model.predict(X_test[-future_days:]))
    future_prices_method2 = []
    last_sequence = scaled_data[-time_step:]
    current_batch = last_sequence.reshape((1, time_step, 1))

    for _ in range(count):
        future_pred = model.predict(current_batch, batch_size=1, verbose=0)[0]
        future_prices_method2.append(future_pred)
        current_batch = np.concatenate((current_batch[:, 1:, :], future_pred.reshape(1, 1, 1)), axis=1)

    future_prices_method2 = scaler.inverse_transform(np.array(future_prices_method2))
    future_prices_method3 = (future_prices_method1[:count] + future_prices_method2) / 2

    methods = [future_prices_method1[:count], future_prices_method2, future_prices_method3]
    dates = []
    future_date = end_date + timedelta(days=1)
    for i in range(count):
        while pd.Timestamp(future_date.replace(hour=0, minute=0, second=0, microsecond=0)) not in market_days:
            future_date += timedelta(days=1)
        dates.append(future_date.strftime("%Y-%m-%d"))
        future_date += timedelta(days=1)

    for method, prediction in zip(methods, prediction_arrays):
        for i in range(count):
            if dates[i] not in prediction[stock]:
                prediction[stock][dates[i]] = []
            prediction[stock][dates[i]].append(method[i].item())

    end_date += timedelta(days=1)

i = 0
for prediction in prediction_arrays:
    for date, values in prediction[stock].items():
        if len(values) == count:
            j = 0
            simple = adjusted_simple_average(values)
            averages_array[i+j].append(simple)
            j += 1
            exponential = adjusted_exponential_weighted_average(values, alpha=0.5)
            averages_array[i+j].append(exponential)
            j += 1
            last = values[count-1]
            averages_array[i+j].append(last)
            j += 1
    i += 3

# Future actual prices (if available in future_data)
future_data = data(stock, last_date, future_date)
actual_future_prices = future_data['close'].values if not future_data.empty else np.array([])

# Evaluate and visualize
if actual_future_prices.size > 0:
    evaluation_results, ideal_profit, buy_and_hold_profit = evaluate_predictions(
        stock_data['close'].values,
        actual_future_prices,
        averages_array,
        labels
    )

    # Determine the best model
    best_model_index = np.argmin([result[1] for result in evaluation_results])
    best_model_label = evaluation_results[best_model_index][0]
    best_model_predictions = averages_array[best_model_index]

    # Plot only the best prediction
    plt.figure(figsize=(12, 6))
    plt.plot(actual_future_prices, label='Actual Prices', color='blue')
    plt.plot(best_model_predictions[:len(actual_future_prices)], label=f'Best Model: {best_model_label}', color='orange')
    plt.title(f'Best Model Prediction vs Actual Prices')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    # Plot profits
    model_labels = [result[0] for result in evaluation_results]
    profits_values = [result[2] for result in evaluation_results]

    # Add ideal and buy-and-hold
    model_labels.extend(['Ideal Scenario (with Shorting)', 'Buy-and-Hold'])
    profits_values.extend([ideal_profit, buy_and_hold_profit])

    plt.figure(figsize=(12, 6))
    plt.bar(model_labels, profits_values, color=['green' if p > 0 else 'red' for p in profits_values])
    plt.title('Profit/Loss Comparison')
    plt.xlabel('Models')
    plt.ylabel('Profit/Loss ($)')
    plt.xticks(rotation=45)
    plt.show()
else:
    print("Skipping evaluation and visualization as future actual prices are not available.")
