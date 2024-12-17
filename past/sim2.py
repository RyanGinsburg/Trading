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
    sum = 0
    for i in range(len(scores)):
        sum += scores[i]
    return sum/len(scores)

def adjusted_exponential_weighted_average(scores, alpha=0.5):
    n = len(scores)
    weights = [alpha * (1 - alpha) ** (n - 1 - i) for i in range(n)]
    total_weights = sum(weights)
    weighted_sum = sum(score * weight for score, weight in zip(scores, weights))
    return weighted_sum / total_weights

api_token = 'lFVm52EqS8EuypuH9FqhzhMAbo7zbeNb'
market_days = get_market_days(2015, 2025)

# Parameters
stock = 'msft'
time_step = 15
future_days = 5
simulated_days = 2
count = 5

# 3D arrays for predictions
predictions_method1 = {}
predictions_method2 = {}
predictions_method3 = {}

# List of prediction dictionaries
prediction_arrays = [predictions_method1, predictions_method2, predictions_method3]

# Initialize each dictionary for the stock symbol
for predictions in prediction_arrays:
    predictions[stock] = {}
    
#2d arrays for averages
p1simple, p1exponential, p1, p2simple, p2exponential, p2, p3simple, p3exponential, p3 = [], [], [], [], [], [], [], [], []


averages_array  = [p1simple, p1exponential, p1, p2simple, p2exponential, p2, p3simple, p3exponential, p3]
labels  = ['p1simple', 'p1exponential', 'p1', 'p2simple', 'p2exponential', 'p2', 'p3simple', 'p3exponential', 'p3']

'''''
#3d arrays for averages
p1simple = {}
p1exponential = {}
p2simple = {}
p2exponential = {}
p3simple = {}
p3exponential = {}

averages_array  = [p1simple, p1exponential, p2simple, p2exponential, p3simple, p3exponential]

# Initialize each dictionary for the stock symbol
for average in averages_array:
    average[stock] = {}
'''''

# Download stock data and prepare dates
end_date = datetime.now() - timedelta(days=270)
last_date = end_date
future_date = datetime.now()
start_date = end_date - timedelta(days=3000)
formatted_end_date = end_date.strftime("%Y-%m-%d")
formatted_future_date = future_date.strftime("%Y-%m-%d")
formatted_start_date = start_date.strftime("%Y-%m-%d")
print(f'End date: {formatted_end_date}')
print(f'Future date: {formatted_future_date}')
print(f'Start date: {formatted_start_date}')

# Get full data
full_data = data(stock, start_date, future_date, technical_indicators=['ema', 'sma', 'williams', 'rsi', 'adx'])
# Simulate predictions for multiple days
position = full_data.index.get_indexer([end_date], method="pad")[0]
# Now get the last preceding dates
prev_dates = full_data.index[max(0, position + 1 - count):position]
end_date = prev_dates[0]

for day in range(simulated_days+count - 1):
    print("_________________________________________________________________________________________")
    while pd.Timestamp(end_date.replace(hour=0, minute=0, second=0, microsecond=0)) not in market_days:
        end_date += timedelta(days=1)
    
    formatted_end_date = end_date.strftime("%Y-%m-%d")
    print(f'Simulation day {day - count + 2}. Last used data: {formatted_end_date}. Creating predicitons for {(end_date+timedelta(days=1)).strftime("%Y-%m-%d")}')

    stock_data = full_data.loc[start_date:end_date]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(stock_data['close'].values.reshape(-1, 1))

    # Create dataset
    X, y = create_dataset(scaled_data, time_step)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Reshape for LSTM input
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Build LSTM model
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

    # Predictions
    future_prices = []
    methods = [[], [], []]

    # Method 1: Direct prediction
    future_predictions_method1 = model.predict(X_test[-future_days:])
    future_prices_method1 = scaler.inverse_transform(future_predictions_method1)
    methods[0] = future_prices_method1[:count]

    # Method 2: Iterative predictions
    last_sequence = scaled_data[-time_step:]
    current_batch = last_sequence.reshape((1, time_step, 1))
    future_predictions_method2 = []

    for _ in range(count):
        try:
            # Make prediction with model and batch size of 1
            future_pred = model.predict(current_batch, batch_size=1, verbose=0)[0]
            future_predictions_method2.append(future_pred)
            
            # Update current_batch by shifting and appending the new prediction
            current_batch = np.concatenate((current_batch[:, 1:, :], future_pred.reshape(1, 1, 1)), axis=1)
        except Exception as e:
            print(f"Error predicting future step: {e}")
            # Optional: Append a fallback value if needed, e.g., 0 or previous prediction
            future_predictions_method2.append(np.array([0]))

    # Inverse transform the predictions to get future prices
    future_prices_method2 = scaler.inverse_transform(np.array(future_predictions_method2))
    methods[1] = future_prices_method2

    # Method 3: Average of Method 1 and Method 2
    future_prices_method3 = (future_prices_method1[:count] + future_prices_method2) / 2
    methods[2] = future_prices_method3

    index = None
    dates = []
    future_date = end_date + timedelta(days=1)
    for i in range(count):
        while pd.Timestamp(future_date.replace(hour=0, minute=0, second=0, microsecond=0)) not in market_days:
            future_date += timedelta(days=1)
        dates.append(future_date.strftime("%Y-%m-%d"))
        future_date += timedelta(days=1)
    
    # Store predictions in dictionaries
    for method, prediction in zip(methods, prediction_arrays):
        for i in range(count):
            if dates[i] not in prediction[stock]:
                prediction[stock][dates[i]] = []
            if prediction[stock][dates[i]] is None:
                prediction[stock][dates[i]] = [method[i].item()]  # Initialize as a list
            else:
                prediction[stock][dates[i]].append(method[i].item())  # Extend with a list

    end_date += timedelta(days=1)
    
i = 0
for prediction in prediction_arrays:
    for date, values in prediction[stock].items():  # Iterate through each date and its associated list of values
        if len(values) == count:  # Check if there are exactly 5 numerical values for this date
            j = 0
            simple = adjusted_simple_average(values)  # Calculate simple average
            averages_array[i+j].append(simple)
            j += 1
            exponential = adjusted_exponential_weighted_average(values, alpha=0.5)  # Calculate exponential weighted average
            averages_array[i+j].append(exponential)
            j += 1
            last = values[count-1]
            averages_array[i+j].append(last)
            j += 1
    i += 3
        
print("_________________________________________________________________________________________")
for average, label in zip(averages_array, labels):
    print(f'{label}: {average}')

# Plotting

# Generate future dates
future_dates = [last_date + timedelta(days=x + 1) for x in range(simulated_days)]
#future_dates = full_data.loc[last_date:last_date + timedelta(days = simulated_days)]

plt.figure(figsize=(15, 7))
plt.plot(stock_data.index, stock_data['close'].values, label='Historical Data')

# Plot actual future data if available
future_data = data(stock, last_date, future_date)
if not future_data.empty:
    plt.plot(future_data.index, future_data['close'].values, label='Actual Future Data', color='green')

# Convert future_dates to datetime
future_dates = pd.to_datetime(future_dates)

# Plot future prediction methods

for average, label in zip(averages_array, labels):
    plt.plot(future_dates, average, label=label, linestyle='--', alpha=0.7)

plt.title(f'{stock} Stock Price Prediction - Comparison of Methods (Starting from Last Known Price)')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()

# Add vertical line to separate historical data and future predictions
plt.axvline(x=last_date, color='red', linestyle='--', alpha=0.5)
plt.text(last_date, plt.ylim()[0], 'Prediction Start', rotation=90, verticalalignment='bottom')

plt.show()

from sklearn.metrics import mean_squared_error

# Evaluation function
def evaluate_predictions(historical_prices, future_prices, predicted_averages, labels):
    evaluation_results = []
    profits = {label: 0 for label in labels}
    
    for i, predictions in enumerate(predicted_averages):
        label = labels[i]
        
        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(future_prices, predictions))
        print(f"RMSE for {label}: {rmse}")
        
        # Calculate profits/losses
        current_price = historical_prices[-1]  # Last known price
        balance = 0  # Total profit/loss
        for predicted_price, actual_price in zip(predictions, future_prices):
            if predicted_price > current_price:  # Buy
                balance += actual_price - current_price
            elif predicted_price < current_price:  # Sell
                balance += current_price - actual_price
            current_price = actual_price  # Update current price for next prediction
        
        profits[label] = balance
        evaluation_results.append((label, rmse, balance))
        print(f"Profit/Loss for {label}: ${balance:.2f}")
    
    return evaluation_results

# Future actual prices (if available in future_data)
actual_future_prices = future_data['close'].values if not future_data.empty else []

# Evaluate only if future actual prices are available
if actual_future_prices:
    evaluation_results = evaluate_predictions(
        stock_data['close'].values,  # Historical prices for reference
        actual_future_prices,        # Actual future prices
        averages_array,              # Predicted averages
        labels                       # Labels for the averages
    )
else:
    print("Future actual prices are not available. Skipping evaluation.")

# Visualization of RMSE and Profits
if actual_future_prices:
    # Extract labels, RMSE values, and profits for plotting
    model_labels = [result[0] for result in evaluation_results]
    rmse_values = [result[1] for result in evaluation_results]
    profits_values = [result[2] for result in evaluation_results]

    # Plot RMSE
    plt.figure(figsize=(10, 5))
    plt.bar(model_labels, rmse_values)
    plt.title('RMSE of Prediction Models')
    plt.xlabel('Prediction Models')
    plt.ylabel('RMSE')
    plt.show()

    # Plot Profits/Losses
    plt.figure(figsize=(10, 5))
    plt.bar(model_labels, profits_values, color='green' if profits_values[0] > 0 else 'red')
    plt.title('Profit/Loss from Prediction Models')
    plt.xlabel('Prediction Models')
    plt.ylabel('Profit/Loss ($)')
    plt.show()