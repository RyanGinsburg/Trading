import xgboost as xgb #type: ignore
import numpy as np #type: ignore
import matplotlib.pyplot as plt #type: ignore
from sklearn.preprocessing import StandardScaler #type: ignore
from sklearn.model_selection import train_test_split #type: ignore
from sklearn.ensemble import RandomForestRegressor #type: ignore
from sklearn.metrics import mean_squared_error #type: ignore
import requests #type: ignore
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler #type: ignore
from keras import Sequential   #type: ignore
from keras.layers import LSTM, Dense, Dropout  # type: ignore
from keras.callbacks import EarlyStopping  # type: ignore
from keras.regularizers import l2  # type: ignore
from keras.optimizers import Adam  # type: ignore
from keras import backend as K  # <-- Added for clearing sessions #type: ignore
from datetime import datetime, timedelta
import requests  # type: ignore
import pandas as pd #type: ignore
import optuna #type: ignore
import tensorflow as tf  #type: ignore
import pandas_market_calendars as mcal #type: ignore
import json
import sqlite3
import multiprocessing
from multiprocessing import shared_memory
import gc
import os

# Suppress TF logs and set multiprocessing to use "spawn"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
multiprocessing.set_start_method("spawn", force=True)
pd.options.mode.chained_assignment = None  # Disable the warning globally

# Directional Accuracy Score (DAS)
def directional_accuracy(y_true, y_pred, current_price):
    signs_true = np.sign(y_true - current_price)
    signs_pred = np.sign(y_pred - current_price)
    return np.mean(signs_true == signs_pred)

# Define Optuna objective functions
# -----------------------------

# -----------------------------
# Utility Functions and Classes
# -----------------------------
def get_market_days(start_year, end_year, exchange='NYSE'):
    market_calendar = mcal.get_calendar(exchange)
    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-12-31"
    schedule = market_calendar.schedule(start_date=start_date, end_date=end_date)
    market_days = schedule.index.to_list()
    return market_days

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

class OptimizationHistory:
    def __init__(self, db_name='optimization_history.db'):
        self.db_name = db_name
        self._initialize_db()

    def _initialize_db(self):
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS optimization_history (
                    stock TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    parameters TEXT NOT NULL,
                    PRIMARY KEY (stock, version)
                )
            """)
            conn.commit()

    def get_parameters(self, stock, version):
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT parameters FROM optimization_history
                WHERE stock = ? AND version = ?
            """, (stock, version))
            result = cursor.fetchone()
            return json.loads(result[0]) if result else None

    def store_parameters(self, stock, version, parameters):
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO optimization_history (stock, version, parameters)
                VALUES (?, ?, ?)
            """, (stock, version, json.dumps(parameters)))
            conn.commit()

def create_optimization_trial(previous_params):
    def suggest_with_history(trial, param_name, suggest_type, *args):
        if previous_params and param_name in previous_params:
            prev_value = previous_params[param_name]
            if suggest_type == 'int':
                return trial.suggest_int(param_name, max(args[0], prev_value - 5), min(args[1], prev_value + 5))
            elif suggest_type == 'float':
                return trial.suggest_float(param_name, max(args[0], prev_value * 0.8), min(args[1], prev_value * 1.2))
            elif suggest_type == 'loguniform':
                return trial.suggest_loguniform(param_name, max(args[0], prev_value * 0.8), min(args[1], prev_value * 1.2))
        else:
            if suggest_type == 'int':
                return trial.suggest_int(param_name, *args)
            elif suggest_type == 'float':
                return trial.suggest_float(param_name, *args)
            elif suggest_type == 'loguniform':
                return trial.suggest_loguniform(param_name, *args)
    return suggest_with_history

def directional_accuracy(y_true, y_pred, current_price):
    signs_true = np.sign(y_true - current_price)
    signs_pred = np.sign(y_pred - current_price)
    return np.mean(signs_true == signs_pred)

def dates():
    dates = [
        datetime(2023, 2, 6),
        datetime(2024, 1, 9),
        datetime(2024, 5, 2),
    ]
    result_dates = []
    for start in dates:
        future_dates = []
        current_date = start
        for _ in range(future_days+1):
            while current_date not in market_days:
                current_date += timedelta(days=1)
            future_dates.append(current_date)
            current_date += timedelta(days=1)
        result_dates.append(future_dates)
    return result_dates

def array_to_comma_separated_string(array: np.ndarray) -> str:
    flat_array = array.flatten()
    return ",".join(map(str, flat_array))

def create_table_if_not_exists(stock, table_date, db_name):
    table_name = f"{stock}_{table_date.strftime('%Y_%m_%d')}"
    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        date DATE PRIMARY KEY,
        open REAL,
        close REAL,
        rsi REAL,
        williams REAL,
        adx REAL,
        pred_1_1 TEXT,
        pred_1_2 TEXT,
        pred_2_1 TEXT,
        pred_2_2 TEXT,
        pred_3_1 TEXT,
        pred_3_2 TEXT,
        pred_4_1 TEXT,
        pred_4_2 TEXT,
        pred_5_1 TEXT,
        pred_5_2 TEXT,
        pred_6_1 TEXT,
        pred_6_2 TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """
    with sqlite3.connect(db_name) as conn:
        cursor = conn.cursor()
        cursor.execute(create_table_sql)
        conn.commit()
    return table_name

def insert_predictions(table_name, last_date, last_open, last_close, last_rsi, last_williams, last_adx, predictions, db_name):
    insert_sql = f"""
    INSERT OR REPLACE INTO {table_name} (
        date, open, close, rsi, williams, adx,
        pred_1_1, pred_1_2, pred_2_1, pred_2_2,
        pred_3_1, pred_3_2, pred_4_1, pred_4_2,
        pred_5_1, pred_5_2, pred_6_1, pred_6_2
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with sqlite3.connect(db_name) as conn:
        cursor = conn.cursor()
        cursor.execute(insert_sql, (last_date, last_open, last_close, last_rsi, last_williams, last_adx, *predictions))
        conn.commit()

def process_stocks(stock):
    print("************************************************************************")
    print(f'Starting to process predictions for {stock}')
    print("************************************************************************")

    full_data = fetch_stock_data(stock, start_date, future_date)
    full_data_np = full_data.to_numpy()
    full_data_columns = full_data.columns
    full_data_index = full_data.index

    shm = shared_memory.SharedMemory(create=True, size=full_data_np.nbytes)
    np_full_data = np.ndarray(full_data_np.shape, dtype=full_data_np.dtype, buffer=shm.buf)
    np_full_data[:] = full_data_np[:]

    for dates_index, dates_ in enumerate(full_dates, start=1):
        if stock == 'stock_name' and dates_index<3:
            continue
        
        print("********************************************************")
        print(f'{stock} Processing date group {dates_index} for {stock}')
        print("________________________________________________________")
        table_date = dates_[0]
        table_name = create_table_if_not_exists(stock, table_date, db_name)

        if stock== 'stock_name':
            dates_to_process = dates_[16:]
        else:
            dates_to_process = dates_

        args_list = [(date, table_name, shm.name, full_data_np.shape, full_data_np.dtype, full_data_columns, full_data_index, stock, dates_index) for date in dates_to_process]

        with multiprocessing.Pool(processes=processes_num) as pool:
            pool.starmap(process_dates, args_list)

    shm.close()
    shm.unlink()

    print(f'[11] Finished processing predictions for {stock}')
    gc.collect()

def process_dates(date, table_name, shm_name, shape, dtype, columns, index, stock, day):
    print("________________________________________________________")
    print(f'[1] {stock} Processing day {day} for date {date}')
    print("________________________________________________________")

    RESET = (day-1) % 5 == 0  # Assuming day is defined earlier

    existing_shm = shared_memory.SharedMemory(name=shm_name)
    np_full_data = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
    full_data = pd.DataFrame(np_full_data, columns=columns, index=index)

    features = ['close', 'volume', 'ema', 'sma', 'williams', 'rsi', 'adx']
    df = full_data.loc[start_date:date]
    trial_num = 5 if testing else (150 if RESET else 50)

    last_row = df.iloc[-1]
    last_date = last_row.name.date()
    last_close = last_row['close']
    last_open = last_row['open']
    last_rsi = last_row['rsi']
    last_williams = last_row['williams']
    last_adx = last_row['adx']

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
    def objective(trial, model_type, loss_type, stock, version, opt_history):
        previous_params = opt_history.get_parameters(stock, version)
        suggest_with_history = create_optimization_trial(previous_params)

        params = {
            'n_estimators': suggest_with_history(trial, 'n_estimators', 'int', 50, 300),
            'max_depth': suggest_with_history(trial, 'max_depth', 'int', 3, 10),
        }

        if model_type == "xgb":
            params['learning_rate'] = suggest_with_history(trial, 'learning_rate', 'float', 0.01, 0.3)
            model = xgb.XGBRegressor(**params)
        elif model_type == "rf":
            params['min_samples_split'] = suggest_with_history(trial, 'min_samples_split', 'int', 2, 10)
            params['min_samples_leaf'] = suggest_with_history(trial, 'min_samples_leaf', 'int', 1, 10)
            model = RandomForestRegressor(**params, random_state=42)

        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        mse = mean_squared_error(y_test, y_pred)
        das = directional_accuracy(y_test.iloc[:, 0].values, y_pred[:, 0], df['close'].iloc[-len(y_test):].values)

        loss = mse if loss_type == "mse" else (1 - das if loss_type == "das" else mse + (5 * (1 - das)))

        opt_history.store_parameters(stock, version, params)
        print("**************************************")
        print(version)
        print("**************************************")

        return loss


    # Retrieve previous best parameters for each model (if they exist)
    prev_xgb_mse = opt_history.get_parameters(stock, "xgb_mse") or {}
    prev_xgb_das = opt_history.get_parameters(stock, "xgb_das") or {}
    prev_xgb_mse_das = opt_history.get_parameters(stock, "xgb_mse_das") or {}
    prev_rf_mse = opt_history.get_parameters(stock, "rf_mse") or {}
    prev_rf_das = opt_history.get_parameters(stock, "rf_das") or {}
    prev_rf_mse_das = opt_history.get_parameters(stock, "rf_mse_das") or {}

    # Debugging: Confirm that version is a string and not a dictionary
    print(f"Retrieving parameters for stock={stock}, version=xgb_mse -> {prev_xgb_mse}")
    print(f"Retrieving parameters for stock={stock}, version=rf_mse -> {prev_rf_mse}")

    # Run Optuna trials with correct version names
    study_xgb_mse = optuna.create_study(direction='minimize')
    study_xgb_mse.optimize(lambda trial: objective(trial, "xgb", "mse", stock, "xgb_mse", opt_history), n_trials=trial_num)

    study_xgb_das = optuna.create_study(direction='minimize')
    study_xgb_das.optimize(lambda trial: objective(trial, "xgb", "das", stock, "xgb_das", opt_history), n_trials=trial_num)

    study_xgb_mse_das = optuna.create_study(direction='minimize')
    study_xgb_mse_das.optimize(lambda trial: objective(trial, "xgb", "mse_das", stock, "xgb_mse_das", opt_history), n_trials=trial_num)

    study_rf_mse = optuna.create_study(direction='minimize')
    study_rf_mse.optimize(lambda trial: objective(trial, "rf", "mse", stock, "rf_mse", opt_history), n_trials=trial_num)

    study_rf_das = optuna.create_study(direction='minimize')
    study_rf_das.optimize(lambda trial: objective(trial, "rf", "das", stock, "rf_das", opt_history), n_trials=trial_num)

    study_rf_mse_das = optuna.create_study(direction='minimize')
    study_rf_mse_das.optimize(lambda trial: objective(trial, "rf", "mse_das", stock, "rf_mse_das", opt_history), n_trials=trial_num)

    # Store only if the new parameters are better
    def update_best_params(model_name, study, prev_best):
        new_best = study.best_params
        if not prev_best or study.best_value < prev_best.get("best_value", float("inf")):
            new_best["best_value"] = study.best_value  # Store best loss value for comparison
            opt_history.store_parameters(stock, model_name, new_best)
            print(f"Updated best parameters for {model_name}: {new_best}")  # Debugging statement

    # Update storage with the best parameters so far
    update_best_params("xgb_mse", study_xgb_mse, prev_xgb_mse)
    update_best_params("xgb_das", study_xgb_das, prev_xgb_das)
    update_best_params("xgb_mse_das", study_xgb_mse_das, prev_xgb_mse_das)
    update_best_params("rf_mse", study_rf_mse, prev_rf_mse)
    update_best_params("rf_das", study_rf_das, prev_rf_das)
    update_best_params("rf_mse_das", study_rf_mse_das, prev_rf_mse_das)









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

    day_num = 0
    for i in range(0, 12, 2):
        # Ensure we don't exceed the length of arrays
        if i + 1 < len(arrays):
            future_prices_method_1, future_prices_method_2 = arrays[i:i+2]
        else:
            break  # Prevent index errors if fewer than expected arrays exist

        # Store predictions in the dictionary correctly
        predictions[f"future_prices_{i+1}"] = array_to_comma_separated_string(future_prices_method_1)
        predictions[f"future_prices_{i+2}"] = array_to_comma_separated_string(future_prices_method_2)

        filtered_predictions = [value for key, value in predictions.items() if key.startswith("future_prices_")]

        # Ensure exactly 12 values
        filtered_predictions = filtered_predictions[:12]

        print("________________________________________________________")
        day_num += 1
        print(f'[2] {stock} Generated predictions for version {day_num}')
        print(f"Expected 12 predictions, got {len(filtered_predictions)}")
        print("________________________________________________________")

    # Insert into database
    insert_predictions(table_name, last_date, last_open, last_close, last_rsi, last_williams, last_adx, filtered_predictions, db_name)
    existing_shm.close()  # Explicitly close shared memory in the worker process
    existing_shm.unlink()
    gc.collect()

# -----------------------------
# Global Variables
# -----------------------------
testing = False
processes_num = 16

if testing:
    forecast_horizon = 2  # Predict next 7 days
    future_days = 1
    stocks = ['AAPL']
else:
    forecast_horizon = 8  # Predict next 7 days
    part = 1
    future_days = 30
    stocks = [
        'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META', 'AMD', 'ORCL',  # technology
        'UNH', 'LLY', 'JNJ', 'MRK', 'ABBV', 'ABT', 'AMGN',         # healthcare
        'JPM', 'BRK.B', 'V', 'MA', 'BAC', 'WFC',                   # finance
        'AMZN', 'TSLA', 'MCD', 'HD',                               # consumer discretionary
        'XOM', 'CVX', 'COP',                                       # energy
        'NFLX', 'DIS', 'CMCSA',                                    # communication services
        'KO', 'PEP'                                              # consumer staples
    ]
    
    if part == 1:
        stocks = stocks[:12]
    elif part == 2:
        stocks = stocks[12:24]
    elif part == 3:
        stocks = stocks[24:]
    else:
        stocks = stocks[:1]

db_name = 'trading_algo.db'
api_token = 'lFVm52EqS8EuypuH9FqhzhMAbo7zbeNb'
technical_indicators = ['volume', 'ema', 'sma', 'williams', 'rsi', 'adx']
conn = sqlite3.connect(db_name)
cursor = conn.cursor()
conn.close()

opt_history = OptimizationHistory('optimization_history.db')
market_days = get_market_days(2015, 2026)

future_date = datetime.now()
start_date = future_date - timedelta(days=3000)
full_dates = dates()

if __name__ == "__main__":
    for stock in stocks:
        process_stocks(stock)
