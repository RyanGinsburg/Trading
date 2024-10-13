import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, LSTM, Dropout #type: ignore
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split  # Importing train_test_split
from sklearn.metrics import mean_squared_error  # Importing mean_squared_error
import requests
import ta
import json

# Step 1: Data Gathering
def get_historical_data(stock_symbol):
    stock_data = yf.download(stock_symbol, period="5y")
    return stock_data

# Step 2: Feature Engineering
def calculate_technical_indicators(df):
    df['SMA_50'] = df['Close'].rolling(window=50).mean()  # Simple Moving Average
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)   # Relative Strength Index
    df['MACD'] = ta.trend.macd(df['Close'])               # MACD Indicator
    df['Volume'] = df['Volume']                           # Trading volume
    df = df.dropna()                                      # Drop NaNs
    return df

# Step 3: Incorporating Sentiment Data
def get_sentiment_data():
    url = "https://api.sentimentanalysisapi.com"  # Example API URL
    response = requests.get(url)
    sentiment_data = json.loads(response.text)
    sentiment_score = sentiment_data['sentiment_score']
    return sentiment_score

# Step 4: Get Economic Indicators (dummy data for simplicity)
def get_economic_indicators():
    return {"interest_rate": 2.5, "gdp_growth": 3.0, "inflation": 1.8}  # Example data

# Step 5: Weighting Features Based on Importance
def apply_feature_weights(df):
    # Weighting based on the importance ranking
    importance_weights = {
        'SMA_50': 0.25,   # Technical indicators like SMA, RSI, MACD
        'RSI': 0.25,
        'MACD': 0.25,
        'Volume': 0.20,   # Trading volume
        'Sentiment': 0.15,  # Sentiment score (market sentiment)
        'Interest_Rate': 0.10,  # Economic indicators
        'GDP_Growth': 0.10,
        'Inflation': 0.10
    }
    
    for feature, weight in importance_weights.items():
        df[feature] = df[feature] * weight  # Scale feature by weight
    
    return df

# Step 6: Feature Combination
def create_features(df):
    economic_indicators = get_economic_indicators()
    df['Interest_Rate'] = economic_indicators['interest_rate']
    df['GDP_Growth'] = economic_indicators['gdp_growth']
    df['Inflation'] = economic_indicators['inflation']

    sentiment_score = get_sentiment_data()
    df['Sentiment'] = sentiment_score

    df = apply_feature_weights(df)  # Apply proportional weighting to features
    return df

# Step 7: Prepare Data for LSTM
def prepare_data(df):
    # Scaling data between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    X = []
    y = []
    
    # Create a sliding window of data for time series forecasting
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i])  # Previous 60 days of data
        y.append(scaled_data[i, 3])  # 'Close' price is the target
    
    X, y = np.array(X), np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Use train_test_split
    return X_train, X_test, y_train, y_test, scaler

# Step 8: Build the LSTM Model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))  # Output layer (prediction of the closing price)
    model.compile(optimizer='adam', loss='mean_squared_error')  # Loss is mean_squared_error
    return model

# Step 9: Train Model
def train_lstm_model(model, X_train, y_train):
    model.fit(X_train, y_train, batch_size=32, epochs=10)
    return model

# Step 10: Make Predictions and Evaluate
def make_predictions(model, X_test, scaler, y_test):
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)  # Rescale predictions back to original values

    mse = mean_squared_error(y_test, predictions)  # Use mean_squared_error
    print(f'Mean Squared Error: {mse}')
    return predictions

# Main pipeline
def main(stock_symbol):
    # Step 1: Get Data
    df = get_historical_data(stock_symbol)
    
    # Step 2: Calculate Technical Indicators
    df = calculate_technical_indicators(df)
    
    # Step 3: Create Features
    df = create_features(df)
    
    # Step 4: Prepare Data for LSTM
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)
    
    # Step 5: Build and Train Model
    lstm_model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    trained_model = train_lstm_model(lstm_model, X_train, y_train)
    
    # Step 6: Make Predictions
    predictions = make_predictions(trained_model, X_test, scaler, y_test)

if __name__ == "__main__":
    stock_symbol = "AAPL"  # Example stock symbol
    main(stock_symbol)
