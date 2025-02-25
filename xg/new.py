import optuna #type: ignore
import xgboost as xgb #type: ignore
import pandas as pd #type: ignore
import numpy as np #type: ignore 
import matplotlib.pyplot as plt#type: ignore 
from sklearn.preprocessing import StandardScaler #type: ignore
from sklearn.model_selection import train_test_split #type: ignore
from sklearn.ensemble import RandomForestRegressor #type: ignore
from sklearn.metrics import mean_squared_error #type: ignore
import requests #type: ignore
from datetime import datetime, timedelta

api_token = 'lFVm52EqS8EuypuH9FqhzhMAbo7zbeNb'

# Fetch Stock Data
def fetch_stock_data(stock, start_date, end_date, technical_indicators=['ema', 'sma', 'williams', 'rsi', 'adx']):
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

# Parameters
stock = 'msft'
trial_num = 150
end_date = datetime.now()
start_date = end_date - timedelta(days=300)

df = fetch_stock_data(stock, start_date, end_date)
features = ['close', 'volume', 'ema', 'sma', 'williams', 'rsi', 'adx']
forecast_horizon = 8  # Predict next 7 days

# Create shifted columns for multi-step forecasting
for i in range(1, forecast_horizon + 1):
    df[f'Close_T+{i}'] = df['close'].shift(-i)

df.dropna(inplace=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df[features], df[[f'Close_T+{i}' for i in range(1, forecast_horizon + 1)]],
    test_size=0.2, shuffle=False
)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Directional Accuracy Score (DAS)
def directional_accuracy(y_true, y_pred, current_price):
    signs_true = np.sign(y_true - current_price)
    signs_pred = np.sign(y_pred - current_price)
    return np.mean(signs_true == signs_pred)

# Define Optuna objective functions
def objective(trial, model_type, loss_type):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
    }
    
    # Add learning_rate only for XGBoost
    if model_type == "xgb":
        params['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 0.3)
        model = xgb.XGBRegressor(**params)
    
    elif model_type == "rf":
        params['min_samples_split'] = trial.suggest_int('min_samples_split', 2, 10)
        params['min_samples_leaf'] = trial.suggest_int('min_samples_leaf', 1, 10)
        model = RandomForestRegressor(**params, random_state=42)
    
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    mse = mean_squared_error(y_test, y_pred)
    das = directional_accuracy(y_test.iloc[:, 0].values, y_pred[:, 0], df['close'].iloc[-len(y_test):].values)
    
    if loss_type == "mse":
        return mse
    elif loss_type == "das":
        return 1 - das  # Minimize DAS
    else:
        return mse + (5 * (1 - das))  # Minimize MSE + DAS

# Train models with Optuna
study_xgb_mse = optuna.create_study(direction='minimize')
study_xgb_mse.optimize(lambda trial: objective(trial, "xgb", "mse"), n_trials=trial_num)

study_xgb_das = optuna.create_study(direction='minimize')
study_xgb_das.optimize(lambda trial: objective(trial, "xgb", "das"), n_trials=trial_num)

study_xgb_mse_das = optuna.create_study(direction='minimize')
study_xgb_mse_das.optimize(lambda trial: objective(trial, "xgb", "mse_das"), n_trials=trial_num)

study_rf_mse = optuna.create_study(direction='minimize')
study_rf_mse.optimize(lambda trial: objective(trial, "rf", "mse"), n_trials=trial_num)

study_rf_das = optuna.create_study(direction='minimize')
study_rf_das.optimize(lambda trial: objective(trial, "rf", "das"), n_trials=trial_num)

study_rf_mse_das = optuna.create_study(direction='minimize')
study_rf_mse_das.optimize(lambda trial: objective(trial, "rf", "mse_das"), n_trials=trial_num)

# Train final models
models = {
    "xgb_mse": xgb.XGBRegressor(**study_xgb_mse.best_params),
    "xgb_das": xgb.XGBRegressor(**study_xgb_das.best_params),
    "xgb_mse_das": xgb.XGBRegressor(**study_xgb_mse_das.best_params),
    "rf_mse": RandomForestRegressor(**study_rf_mse.best_params),
    "rf_das": RandomForestRegressor(**study_rf_das.best_params),
    "rf_mse_das": RandomForestRegressor(**study_rf_mse_das.best_params)
}

for model in models.values():
    model.fit(X_train_scaled, y_train)

# Rolling Forecast Function
def rolling_forecast(model, initial_data, num_days):
    rolling_features = initial_data.copy()
    predictions = []
    
    for _ in range(num_days):
        pred = np.ravel(model.predict(rolling_features.reshape(1, -1)))[0]  
        predictions.append(pred)
        rolling_features[:-1] = rolling_features[1:]  
        rolling_features[-1] = pred  
    
    return predictions

# Generate predictions
latest_data = X_test_scaled[-1]

predictions = {}
for key, model in models.items():
    predictions[f"{key}_direct"] = model.predict(latest_data.reshape(1, -1))[0]
    predictions[f"{key}_rolling"] = rolling_forecast(model, latest_data, forecast_horizon)

arrays = [np.array(pred) for pred in predictions.values()]

# ------------------------- Print Predictions First ------------------------- #
print("\nFull Predictions Array for Each Method:\n")

# Print entire array for each method
for label, pred in predictions.items():
    print(f"{label.replace('_', ' ').title()}: {np.array(pred)}")

print("\nNext 7 Days of Predictions for Each Method:\n")

# Print next 7 days separately
for label, pred in predictions.items():
    print(f"{label.replace('_', ' ').title()}:")
    for day, price in enumerate(pred[:7], start=1):  # First 7 days
        print(f"  Day {day}: {price:.2f}")
    print()  # Blank line for readability

# ------------------------- Plot Actual vs Predicted ------------------------- #

# Extract last 30 days of actual prices
actual_prices_past = df['close'].iloc[-30:].values
actual_prices_future = df['close'].iloc[-forecast_horizon:].values  # Actual future prices

# Create x-axis labels
days_past = np.arange(-30, 1)  # Include day 0 for alignment
days_future = np.arange(0, forecast_horizon + 1)  # Include day 0

# Define simplified color scheme
color_scheme = {
    "xgb_mse_direct": "darkred", "xgb_mse_iteration": "red",
    "xgb_das_direct": "darkblue", "xgb_das_iteration": "blue",
    "xgb_mse_das_direct": "goldenrod", "xgb_mse_das_iteration": "yellow",
    
    "rf_mse_direct": "darkred", "rf_mse_iteration": "red",
    "rf_das_direct": "darkblue", "rf_das_iteration": "blue",
    "rf_mse_das_direct": "goldenrod", "rf_mse_das_iteration": "yellow"
}

# Define line styles: Solid for XGBoost, Dashed for Random Forest
line_styles = {"xgb": "-", "rf": "--"}

# Last actual price before predictions start
last_actual_price = actual_prices_past[-1]

# Adjust actual future prices to start from the last actual price at Day 0
adjusted_actual_prices_future = np.insert(actual_prices_future, 0, last_actual_price)

# Start the plot
plt.figure(figsize=(12, 6))

# Plot past actual prices
plt.plot(days_past, np.append(actual_prices_past, last_actual_price), linestyle='-', label="Actual Prices (Past)", color="black", linewidth=2)

# Plot adjusted future actual prices (starting from Day 0)
plt.plot(days_future, adjusted_actual_prices_future, linestyle='-', linewidth=2.5, label="Actual Prices (Future)", color="black")

# Add vertical gray lines for each day in the future
for day in days_future:
    plt.axvline(x=day, color="lightgray", linestyle="--", linewidth=0.75)

# Adjust predictions to start from the last actual price at Day 0
for label, pred in predictions.items():
    model_type = "xgb" if "xgb" in label else "rf"  # Determine model type
    eval_type = "mse" if "mse_das" not in label and "das" not in label else "das" if "mse_das" not in label else "mse_das"
    method = "direct" if "direct" in label else "iteration"
    
    color = color_scheme[f"{model_type}_{eval_type}_{method}"]
    
    # Adjust first value to match last actual price at Day 0
    adjusted_pred = np.insert(pred, 0, last_actual_price)  

    plt.plot(days_future, adjusted_pred, linestyle=line_styles[model_type], label=label.replace("_", " ").title(), color=color, linewidth=2)

# Add a vertical line indicating where predictions start
plt.axvline(x=0, color='gray', linestyle='--', linewidth=1.5, label="Prediction Start")

# Formatting
plt.xlabel("Days (Past & Future)")
plt.ylabel("Stock Price")
plt.title("Stock Price Forecasting: Actual vs. Predictions (12 Methods)")
plt.legend()
plt.grid(True)
plt.show()
