import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import LSTM, Dense # type: ignore
from datetime import datetime, timedelta
import requests
import pandas as pd

#test

def create_dataset(data, time_step):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)



def data(stock, start_date='2019-10-14 00:00:00', end_date=datetime.now(), technical_indicators=['rsi']):
    
    # Initialize an empty list to store the DataFrames
    df_list = []

    # Loop through the technical indicators array
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

        # Add the DataFrame to the list
        df_list.append(df)

    # Merge all DataFrames in the list on the date index
    combined_df = pd.concat(df_list, axis=1)
    
    # Filter the DataFrame to include only the rows between start_date and end_date
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    df_filtered = combined_df.loc[start_date:end_date]
    
    print(df_filtered)
    
    return df_filtered

api_token = 'lFVm52EqS8EuypuH9FqhzhMAbo7zbeNb'  # Defines api token

# Parameters
stock = 'msft'
time_step = 15
future_days = 100

# Download stock data
end_date = datetime.now() - timedelta(days=270)
future_date = datetime.now()
start_date = end_date - timedelta(days=3000)  # 10 years of data
stock_data = data(stock, start_date, end_date, technical_indicators=['ema', 'sma', 'williams', 'rsi', 'adx' ])

print(stock_data)

print(f'start date: {start_date}')
print(f'end date: {end_date}')

# Prepare data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(stock_data['close'].values.reshape(-1, 1))

# Create dataset
X, y = create_dataset(scaled_data, time_step)

# Split data into train and test
train_size = int(len(X) * 0.8)  # Using 80% for training
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape input for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

# Compile and train the model
model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), 
                    epochs=10, batch_size=32, verbose=1)

# Make predictions on test data
test_predictions = model.predict(X_test)
test_predictions = scaler.inverse_transform(test_predictions)
y_test_transformed = scaler.inverse_transform([y_test]).T

# Calculate RMSE
rmse = np.sqrt(np.mean((test_predictions - y_test_transformed) ** 2))
print(f'\nModel RMSE on test data: ${rmse:.2f}')

# Method 1: Future predictions using test data
future_predictions_method1 = model.predict(X_test[-future_days:])
future_prices_method1 = scaler.inverse_transform(future_predictions_method1)

# Method 2: Iterative future predictions (based on the last sequence of historical data)
last_sequence = scaled_data[-time_step:]  # Use the last part of the entire historical data
future_predictions_method2 = []

# Reshape the last sequence for prediction
current_batch = last_sequence.reshape((1, time_step, 1))

# Predict future values based on historical data
for i in range(future_days):
    # Get the prediction
    future_pred = model.predict(current_batch, verbose=0)[0]
    
    # Append the prediction
    future_predictions_method2.append(future_pred)
    
    # Update the sequence by shifting values and adding the new prediction
    current_batch = np.append(current_batch[:, 1:, :], [[future_pred]], axis=1)

# Convert predictions back to original scale
future_predictions_method2 = np.array(future_predictions_method2)
future_prices_method2 = scaler.inverse_transform(future_predictions_method2)

# Method 3: Combination of Method 1 and Method 2. Average of method 1 and 2
future_prices_method3 = (future_prices_method1 + future_prices_method2) / 2

# Method 6: Combination of Method 1 and Method 2. Average of method 1 and 2
future_prices_method6 = (future_prices_method1*3 + future_prices_method2*2) / 5

# Method 4: Iterative future predictions based on Method 1's previous output
last_sequence_method1 = scaled_data[-(time_step - future_days):]  # Combine last part of historical data
combined_method1 = np.concatenate((last_sequence_method1, scaler.transform(future_prices_method1).reshape(-1, 1)))

future_predictions_method4 = []
current_batch = combined_method1[:time_step].reshape((1, time_step, 1))  # Use combined sequence

for i in range(future_days):
    future_pred = model.predict(current_batch, verbose=0)[0]
    future_predictions_method4.append(future_pred)
    current_batch = np.append(current_batch[:, 1:, :], [[future_pred]], axis=1)

future_predictions_method4 = np.array(future_predictions_method4)
future_prices_method4 = scaler.inverse_transform(future_predictions_method4)


# Method 5: Iterative future predictions based on Method 3's previous output
last_sequence_method3 = scaled_data[-(time_step - future_days):]  # Combine last part of historical data
combined_method3 = np.concatenate((last_sequence_method3, scaler.transform(future_prices_method3).reshape(-1, 1)))

future_predictions_method5 = []
current_batch = combined_method3[:time_step].reshape((1, time_step, 1))  # Use combined sequence

for i in range(future_days):
    future_pred = model.predict(current_batch, verbose=0)[0]
    future_predictions_method5.append(future_pred)
    current_batch = np.append(current_batch[:, 1:, :], [[future_pred]], axis=1)

future_predictions_method5 = np.array(future_predictions_method5)
future_prices_method5 = scaler.inverse_transform(future_predictions_method5)


# Get the last known price from historical data
last_price = stock_data['close'].values[-1]

# Adjust predictions so they all start from the last historical price
# Method 1 adjustment
method1_start_diff = future_prices_method1[0] - last_price
future_prices_method1 -= method1_start_diff

# Method 2 adjustment
method2_start_diff = future_prices_method2[0] - last_price
future_prices_method2 -= method2_start_diff

# Method 3 adjustment
method3_start_diff = future_prices_method3[0] - last_price
future_prices_method3 -= method3_start_diff

# Method 4 adjustment
method4_start_diff = future_prices_method4[0] - last_price
future_prices_method4 -= method4_start_diff

# Method 5 adjustment
method5_start_diff = future_prices_method5[0] - last_price
future_prices_method5 -= method5_start_diff

# Method 6 adjustment
method6_start_diff = future_prices_method6[0] - last_price
future_prices_method6 -= method6_start_diff

# Generate future dates
last_date = stock_data.index[-1]

# Convert last_date to datetime if necessary
if not isinstance(last_date, pd.Timestamp):
    last_date = pd.to_datetime(last_date)

# Generate a list of future dates, adding a timedelta to the last date
future_dates = [last_date + timedelta(days=x+1) for x in range(future_days)]


# Plotting
plt.figure(figsize=(15, 7))
plt.plot(stock_data.index, stock_data['close'].values, label='Historical Data')

# Plot actual future data if available
future_data = data(stock, last_date, future_date)
if not future_data.empty:
    plt.plot(future_data.index, future_data['close'].values, label='Actual Future Data', color='green')

# Convert future_dates to datetime
future_dates = pd.to_datetime(future_dates)

# Plot all future prediction methods (starting from the same last price)
plt.plot(future_dates, future_prices_method1, label='Future Predictions (Method 1)', linestyle='--', alpha=0.5)
plt.plot(future_dates, future_prices_method2, label='Future Predictions (Method 2)', linestyle=':', alpha=0.5)
plt.plot(future_dates, future_prices_method3, label='Future Predictions (Method 3)', linestyle='-.', alpha=0.7)
plt.plot(future_dates, future_prices_method4, label='Future Predictions (Method 4)', linestyle='--', alpha=0.5)
plt.plot(future_dates, future_prices_method5, label='Future Predictions (Method 5)', linestyle='-.', alpha=0.7)
plt.plot(future_dates, future_prices_method6, label='Future Predictions (Method 6)', linestyle='-.', alpha=0.7)

plt.title(f'{stock} Stock Price Prediction - Comparison of Methods (Starting from Last Known Price)')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()

# Add vertical line to separate historical data and future predictions
plt.axvline(x=last_date, color='red', linestyle='--', alpha=0.5)
plt.text(last_date, plt.ylim()[0], 'Prediction Start', rotation=90, verticalalignment='bottom')

plt.show()