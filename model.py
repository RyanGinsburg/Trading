import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, GRU
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns

# Function to fetch stock data and calculate technical indicators
def fetch_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data['SMA'] = data['Close'].rolling(window=20).mean()
    data['EMA'] = data['Close'].ewm(span=20, adjust=False).mean()
    data['RSI'] = calculate_rsi(data['Close'])
    data.dropna(inplace=True)
    return data

# Function to calculate RSI
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Function to prepare data for sequence models
def prepare_sequence_data(data, lookback):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:(i + lookback)])
        y.append(data[i + lookback, 0])  # Predict only the 'Close' price
    return np.array(X), np.array(y)

# Function to evaluate models
def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
    return mse, mae, r2

# Fetch and prepare data
ticker = "AAPL"
start_date = "2018-01-01"
end_date = "2023-01-01"
data = fetch_stock_data(ticker, start_date, end_date)

# Prepare features and target
features = ['Close', 'SMA', 'EMA', 'RSI']
X = data[features].values
y = data['Close'].values

# Split the data
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Scale the data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

# Prepare sequence data for LSTM and GRU
lookback = 60
X_train_seq, y_train_seq = prepare_sequence_data(X_train_scaled, lookback)
X_test_seq, y_test_seq = prepare_sequence_data(X_test_scaled, lookback)

# Initialize dictionary to store results and predictions
results = {}
predictions = {}

# 1. ARIMA Model
print("Training ARIMA model...")
arima_model = ARIMA(y_train, order=(5,1,0))
arima_results = arima_model.fit()
predictions['ARIMA'] = arima_results.forecast(steps=len(y_test))
results['ARIMA'] = evaluate_model(y_test, predictions['ARIMA'], "ARIMA")

# 2. Random Forest
print("Training Random Forest model...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train_scaled)
predictions['Random Forest'] = scaler_y.inverse_transform(rf_model.predict(X_test_scaled).reshape(-1, 1)).flatten()
results['Random Forest'] = evaluate_model(y_test, predictions['Random Forest'], "Random Forest")

# 3. XGBoost
print("Training XGBoost model...")
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                         colsample_bytree=1, max_depth=7)
xgb_model.fit(X_train_scaled, y_train_scaled)
predictions['XGBoost'] = scaler_y.inverse_transform(xgb_model.predict(X_test_scaled).reshape(-1, 1)).flatten()
results['XGBoost'] = evaluate_model(y_test, predictions['XGBoost'], "XGBoost")

# 4. LSTM Model
print("Training LSTM model...")
lstm_model = Sequential([
    LSTM(50, activation='relu', input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]), return_sequences=True),
    LSTM(50, activation='relu', return_sequences=False),
    Dense(1)
])
lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
lstm_model.fit(X_train_seq, y_train_seq, epochs=50, batch_size=32, validation_split=0.1, verbose=0)
predictions['LSTM'] = scaler_y.inverse_transform(lstm_model.predict(X_test_seq)).flatten()
results['LSTM'] = evaluate_model(y_test[lookback:], predictions['LSTM'], "LSTM")

# 5. GRU Model
print("Training GRU model...")
gru_model = Sequential([
    GRU(50, activation='relu', input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]), return_sequences=True),
    GRU(50, activation='relu', return_sequences=False),
    Dense(1)
])
gru_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
gru_model.fit(X_train_seq, y_train_seq, epochs=50, batch_size=32, validation_split=0.1, verbose=0)
predictions['GRU'] = scaler_y.inverse_transform(gru_model.predict(X_test_seq)).flatten()
results['GRU'] = evaluate_model(y_test[lookback:], predictions['GRU'], "GRU")

# Visualize results
plt.figure(figsize=(12, 6))
models = list(results.keys())
mse_values = [result[0] for result in results.values()]
mae_values = [result[1] for result in results.values()]
r2_values = [result[2] for result in results.values()]

x = np.arange(len(models))
width = 0.25

plt.bar(x - width, mse_values, width, label='MSE')
plt.bar(x, mae_values, width, label='MAE')
plt.bar(x + width, r2_values, width, label='R2')

plt.xlabel('Models')
plt.ylabel('Scores')
plt.title('Model Comparison')
plt.xticks(x, models, rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Plot predictions vs actual for the best model
best_model = min(results, key=lambda k: results[k][0])  # Model with lowest MSE
plt.figure(figsize=(12, 6))
plt.plot(data.index[train_size:], y_test, label='Actual')
if best_model in ['LSTM', 'GRU']:
    plt.plot(data.index[train_size+lookback:], predictions[best_model], label=f'{best_model} Predictions')
else:
    plt.plot(data.index[train_size:], predictions[best_model], label=f'{best_model} Predictions')
plt.title(f'Actual vs Predicted Prices - {best_model} Model')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

print(f"Best performing model: {best_model}")