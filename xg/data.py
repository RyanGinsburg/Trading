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
from sklearn.ensemble import GradientBoostingClassifier  #type: ignore
from sklearn.model_selection import TimeSeriesSplit #type: ignore
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

def dates():
    dates = [
        datetime(2025, 2, 11),
        #datetime(2024, 1, 9),
        #datetime(2024, 5, 2),
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
        if stock == stock_start_point and dates_index<group:
            continue
        
        print("********************************************************")
        print(f'{stock} Processing date group {dates_index} for {stock}')
        print("________________________________________________________")
        table_date = dates_[0]
        table_name = create_table_if_not_exists(stock, table_date, db_name)

        if stock == stock_start_point:
            dates_to_process = dates_[date:]
        else:
            dates_to_process = dates_

        args_list = [(date, table_name, shm.name, full_data_np.shape, full_data_np.dtype, full_data_columns, full_data_index, stock, dates_index) for date in dates_to_process]

        with multiprocessing.Pool(processes=processes_num) as pool:
            pool.starmap(process_dates, args_list)

    shm.close()
    shm.unlink()

    print(f'[11] Finished processing predictions for {stock}')
    gc.collect()

# === MODIFIED process_dates() with Walk-Forward Validation, 12 Prediction Methods, and Confidence Filtering ===

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
        class_pred TEXT,
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
        pred_5_1, pred_5_2, pred_6_1, pred_6_2,
        class_pred
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with sqlite3.connect(db_name) as conn:
        cursor = conn.cursor()
        cursor.execute(insert_sql, (last_date, last_open, last_close, last_rsi, last_williams, last_adx, *predictions))
        conn.commit()


def process_dates(date, table_name, shm_name, shape, dtype, columns, index, stock, day):
    print("________________________________________________________")
    print(f'[1] {stock} Processing day {day} for date {date}')
    print("________________________________________________________")

    RESET = (day-1) % 5 == 0
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    np_full_data = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
    full_data = pd.DataFrame(np_full_data, columns=columns, index=index)

    df = full_data.loc[start_date:date].copy()

    # Add lagged features
    df['close_lag1'] = df['close'].shift(1)
    df['volume_lag1'] = df['volume'].shift(1)
    df['rsi_lag1'] = df['rsi'].shift(1)
    df['adx_lag1'] = df['adx'].shift(1)

    features = ['close', 'volume', 'ema', 'sma', 'williams', 'rsi', 'adx',
                'close_lag1', 'volume_lag1', 'rsi_lag1', 'adx_lag1']

    for i in range(1, forecast_horizon + 1):
        df[f'Close_T+{i}'] = df['close'].shift(-i)

    df['Close_T+1'] = df['close'].shift(-1)
    df.dropna(inplace=True)

    y_class = (df['Close_T+1'] > df['close']).astype(int)

    scaler = StandardScaler()
    X = df[features]
    X_scaled = scaler.fit_transform(X)
    y = df[[f'Close_T+{i}' for i in range(1, forecast_horizon + 1)]]
    y_class_full = y_class

    split_index = int(0.8 * len(df))
    X_train, X_test = X_scaled[:split_index], X_scaled[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    y_train_class = y_class_full[:split_index]
    y_test_class = y_class_full[split_index:]

    trial_num = 5 if testing else (150 if RESET else 50)

    # Directional Accuracy Score (DAS)
    def directional_accuracy(y_true, y_pred, current_price):
        signs_true = np.sign(y_true - current_price)
        signs_pred = np.sign(y_pred - current_price)
        return np.mean(signs_true == signs_pred)

    # Profit Score
    def profit_score(y_pred, open_prices):
        return np.sum((y_pred - open_prices)[(y_pred - open_prices) > 0])

    # Create history-aware trial suggestions
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

    # Define a generic Optuna objective function supporting multiple loss types and parameter memory
    def optuna_objective(model_class, loss_type, X_train, y_train, df, suggest_with_history, stock, version, opt_history):
        def objective(trial):
            if model_class.__name__ == 'XGBRegressor':
                params = {
                    'n_estimators': suggest_with_history(trial, 'n_estimators', 'int', 50, 300),
                    'max_depth': suggest_with_history(trial, 'max_depth', 'int', 3, 10),
                    'learning_rate': suggest_with_history(trial, 'learning_rate', 'float', 0.01, 0.3),
                    'subsample': suggest_with_history(trial, 'subsample', 'float', 0.6, 1.0),
                    'colsample_bytree': suggest_with_history(trial, 'colsample_bytree', 'float', 0.6, 1.0),
                }
            else:
                params = {
                    'n_estimators': suggest_with_history(trial, 'n_estimators', 'int', 50, 300),
                    'max_depth': suggest_with_history(trial, 'max_depth', 'int', 3, 10),
                    'min_samples_split': suggest_with_history(trial, 'min_samples_split', 'int', 2, 10),
                    'min_samples_leaf': suggest_with_history(trial, 'min_samples_leaf', 'int', 1, 10),
                }

            model = model_class(**params)
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []

            for train_idx, val_idx in tscv.split(X_train):
                if len(train_idx) == 0 or len(val_idx) == 0:
                    continue

                model.fit(X_train[train_idx], y_train.iloc[train_idx])
                y_pred = model.predict(X_train[val_idx])

                if y_pred.ndim == 1:
                    y_pred = y_pred.reshape(-1, 1)

                if len(y_pred) == 0:
                    continue

                if loss_type == 'das':
                    close_slice = df['close'].iloc[train_idx[-1]:train_idx[-1] + len(val_idx)]
                    if len(close_slice) != len(y_pred):
                        continue
                    score = directional_accuracy(y_train.iloc[val_idx, 0].values, y_pred[:, 0], close_slice.values)
                    scores.append(1 - score)

                elif loss_type == 'profit':
                    open_prices = df['open'].iloc[train_idx[-1]:train_idx[-1] + len(val_idx)].values
                    if len(open_prices) != len(y_pred):
                        continue
                    scores.append(-profit_score(y_pred[:, 0], open_prices))

                elif loss_type == 'hybrid':
                    close_slice = df['close'].iloc[train_idx[-1]:train_idx[-1] + len(val_idx)]
                    open_prices = df['open'].iloc[train_idx[-1]:train_idx[-1] + len(val_idx)].values
                    if len(close_slice) != len(y_pred) or len(open_prices) != len(y_pred):
                        continue
                    das_score = directional_accuracy(y_train.iloc[val_idx, 0].values, y_pred[:, 0], close_slice.values)
                    profit = profit_score(y_pred[:, 0], open_prices)
                    hybrid_score = 0.5 * (1 - das_score) + 0.5 * (-profit)
                    scores.append(hybrid_score)

            if not scores:
                return float("inf")

            # Save the best parameters to optimization history
            opt_history.store_parameters(stock, version, params)

            return np.mean(scores)

        return objective

    # Forecast function that returns a flat list of predictions
    def rolling_forecast(model, latest_data):
        preds = model.predict(latest_data.reshape(1, -1))[0]
        return str(np.round(preds, 4).tolist())

    
    def build_model(model_class, loss_type, stock, version):
        previous_params = opt_history.get_parameters(stock, version)
        suggest_with_history = create_optimization_trial(previous_params)

        study = optuna.create_study(direction='minimize')
        study.optimize(
            optuna_objective(
                model_class, loss_type, X_train, y_train, df,
                suggest_with_history, stock, version, opt_history
            ),
            n_trials=trial_num
        )
        best_model = model_class(**study.best_params)
        best_model.fit(X_train, y_train)
        return best_model


    models = {
        "pred_1_1": build_model(RandomForestRegressor, 'profit', stock, "pred_1_1"),
        "pred_1_2": build_model(RandomForestRegressor, 'profit', stock, "pred_1_2"),
        "pred_2_1": build_model(RandomForestRegressor, 'das', stock, "pred_2_1"),
        "pred_2_2": build_model(RandomForestRegressor, 'das', stock, "pred_2_2"),
        "pred_3_1": build_model(xgb.XGBRegressor, 'profit', stock, "pred_3_1"),
        "pred_3_2": build_model(xgb.XGBRegressor, 'profit', stock, "pred_3_2"),
        "pred_4_1": build_model(xgb.XGBRegressor, 'das', stock, "pred_4_1"),
        "pred_4_2": build_model(xgb.XGBRegressor, 'das', stock, "pred_4_2"),
        "pred_5_1": build_model(RandomForestRegressor, 'hybrid', stock, "pred_5_1"),
        "pred_5_2": build_model(RandomForestRegressor, 'hybrid', stock, "pred_5_2"),
        "pred_6_1": build_model(xgb.XGBRegressor, 'hybrid', stock, "pred_6_1"),
        "pred_6_2": build_model(xgb.XGBRegressor, 'hybrid', stock, "pred_6_2"),
    }


    latest_data = X_test[-1]
    # Forecast function that returns a flat list of predictions
    def rolling_forecast(model, latest_data):
        preds = model.predict(latest_data.reshape(1, -1))[0]
        return str(np.round(preds, 4).tolist())

    predictions = [rolling_forecast(model, latest_data) for model in models.values()]


    clf = GradientBoostingClassifier()
    clf.fit(X_train, y_train_class)
    class_pred = clf.predict_proba(latest_data.reshape(1, -1))[0][1]

    last_row = df.iloc[-1]
    last_date = last_row.name.date()
    last_close = last_row['close']
    last_open = last_row['open']
    last_rsi = last_row['rsi']
    last_williams = last_row['williams']
    last_adx = last_row['adx']

    predictions.append(str(round(class_pred, 4)))
    insert_predictions(table_name, last_date, last_open, last_close, last_rsi, last_williams, last_adx, predictions, db_name)

    existing_shm.close()
    existing_shm.unlink()
    gc.collect()

# -----------------------------
# Global Variables
# -----------------------------
testing = False
processes_num = 16

if testing:
    forecast_horizon = 2 # Predict next 7 days
    future_days = 2
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
    stock_start_point = 'none'
    group = 2
    date = 16
    
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
    if stock_start_point in stocks:
        # Find the index of the target and process from that point on
        start_index = stocks.index(stock_start_point)
        for stock in stocks[start_index:]:
            process_stocks(stock)
    else:
        for stock in stocks:
            process_stocks(stock)

