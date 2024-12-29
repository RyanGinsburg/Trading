import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import LSTM, Dense, Dropout  # type: ignore
from keras.callbacks import EarlyStopping  # type: ignore
from keras.regularizers import l2 # type: ignore
from keras.optimizers import Adam # type: ignore
from datetime import datetime, timedelta
import requests
import pandas as pd
import optuna
import tensorflow as tf
import pandas_market_calendars as mcal
import json
import os
from pathlib import Path
import sqlite3

def get_market_days(start_year, end_year, exchange='NYSE'):
    market_calendar = mcal.get_calendar(exchange)
    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-12-31"
    schedule = market_calendar.schedule(start_date=start_date, end_date=end_date)
    market_days = schedule.index.to_list()
    return market_days

def fetch_technical_data(stock, start_date, end_date, technical_indicators=['rsi']):
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
    def __init__(self, filepath='optimization_history.json'):
        self.filepath = Path(filepath)
        self.history = self._load_history()

    def _load_history(self):
        """Load existing history from file or create new if doesn't exist"""
        if self.filepath.exists():
            try:
                with open(self.filepath, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {}
        return {}

    def _save_history(self):
        """Save history to file"""
        with open(self.filepath, 'w') as f:
            json.dump(self.history, f, indent=4)

    def get_parameters(self, stock, version):
        """Retrieve stored parameters for a specific stock and version"""
        key = f"{stock}_v{version}"
        return self.history.get(key, None)

    def store_parameters(self, stock, version, parameters):
        """Store parameters for a specific stock and version"""
        key = f"{stock}_v{version}"
        self.history[key] = parameters
        self._save_history()

def create_optimization_trial(previous_params):
    """
    Create an optimization trial with previous parameters as starting points
    """
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

def versions(i, RESET=False):
    if i == 1:
        def v1_create_dataset(data, time_step):
            X, y = [], []
            for i in range(len(data) - time_step - 1):
                X.append(data[i:(i + time_step), 0])
                y.append(data[i + time_step, 0])
            return np.array(X), np.array(y)

        def v1_build_lstm(input_shape, units=50, dropout_rate=0.2, learning_rate=0.001):
            model = Sequential([
                LSTM(units, return_sequences=True, input_shape=input_shape),
                Dropout(dropout_rate),
                LSTM(units),
                Dropout(dropout_rate),
                Dense(1)
            ])
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
            return model

        def v1_optimize_model(data, train_split=0.7):
            previous_params = opt_history.get_parameters(stock, 1)
            num_trials = 30
            if not previous_params:
                if RESET:
                    num_trials = 30
                else: 
                    num_trials = 10
            suggest = create_optimization_trial(previous_params)
            
            def v1_objective(trial):
                params = {
                    'time_step': suggest(trial, 'time_step', 'int', 1, 30),
                    'units': suggest(trial, 'units', 'int', 30, 100),
                    'dropout_rate': suggest(trial, 'dropout_rate', 'float', 0.1, 0.5),
                    'learning_rate': suggest(trial, 'learning_rate', 'loguniform', 1e-4, 1e-2)
                }

                X, y = v1_create_dataset(data, params['time_step'])
                train_size = int(len(X) * train_split)
                X_train, X_val = X[:train_size], X[train_size:]
                y_train, y_val = y[:train_size], y[train_size:]

                X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
                X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)

                model = v1_build_lstm(X_train.shape[1:], params['units'], params['dropout_rate'], params['learning_rate'])
                early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32, verbose=0, callbacks=[early_stop])

                val_loss = model.evaluate(X_val, y_val, verbose=0)
                return val_loss

            study = optuna.create_study(direction='minimize')
            study.optimize(v1_objective, n_trials=num_trials)
            best_params = study.best_params
            opt_history.store_parameters(stock, 1, best_params)
            return best_params

        # Prediction Functions
        def v1_future_predictions_method_1(model, X_test, scaler, future_days):
            future_predictions = model.predict(X_test[-(future_days+1):])
            return scaler.inverse_transform(future_predictions)

        def v1_future_predictions_method_2(model, last_sequence, future_days):
            predictions = []
            current_batch = last_sequence.reshape((1, last_sequence.shape[0], 1))
            for _ in range(future_days+1):
                pred = model.predict(current_batch, verbose=0)[0]
                predictions.append(pred)
                current_batch = np.append(current_batch[:, 1:, :], [[pred]], axis=1)
            return np.array(predictions)

        # Main Workflow
        def v1_main():
            last_price = stock_data['close'].values[-1]
            
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(stock_data['close'].values.reshape(-1, 1))

            best_params = v1_optimize_model(scaled_data)
            time_step = best_params['time_step']

            X, y = v1_create_dataset(scaled_data, time_step)
            train_size = int(len(X) * 0.7)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]

            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

            model = v1_build_lstm(X_train.shape[1:], units=best_params['units'], dropout_rate=best_params['dropout_rate'], learning_rate=best_params['learning_rate'])
            early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32, callbacks=[early_stop])

            # Method 1
            future_prices_method_1 = v1_future_predictions_method_1(model, X_test, scaler, future_days)

            # Method 2
            last_sequence = scaler.transform(stock_data['close'].values[-time_step:].reshape(-1, 1)).reshape(-1)
            future_predictions = v1_future_predictions_method_2(model, last_sequence, future_days)
            future_prices_method_2 = scaler.inverse_transform(future_predictions)
            
            future_prices_method_2 -= future_prices_method_2[0] - last_price
            future_prices_method_1 -= future_prices_method_1[0] - last_price
            future_prices_method_2 = future_prices_method_2[1:]
            future_prices_method_1 = future_prices_method_1[1:]
            return future_prices_method_1[:data_num], future_prices_method_2[:data_num]
        
        return(v1_main())
    elif i == 2:
        def v2_create_dataset(data, time_step):
            X, y = [], []
            for i in range(len(data) - time_step - 1):
                X.append(data[i:(i + time_step), 0])
                y.append(data[i + time_step, 0])
            return np.array(X), np.array(y)

        def v2_augment_data(data, noise_level=0.02):
            noise = np.random.normal(0, noise_level, data.shape)
            return data + noise

        def v2_build_lstm(input_shape, units=50, dropout_rate=0.3, learning_rate=0.001):
            model = Sequential([
                LSTM(units, return_sequences=True, input_shape=input_shape, kernel_regularizer=l2(0.01)),  # Added L2 regularization
                Dropout(dropout_rate),
                LSTM(units, kernel_regularizer=l2(0.01)),  # Added L2 regularization
                Dropout(dropout_rate),
                Dense(1)
            ])
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
            return model

        def v2_optimize_model(data, train_split=0.7):
            previous_params = opt_history.get_parameters(stock, 2)
            suggest = create_optimization_trial(previous_params)
            num_trials = 40
            if not previous_params:
                if RESET:
                    num_trials = 40
                else:
                    num_trials = 15
            
            def v2_objective(trial):
                params = {
                    'time_step': suggest(trial, 'time_step', 'int', 1, 30),
                    'units': suggest(trial, 'units', 'int', 30, 100),
                    'dropout_rate': suggest(trial, 'dropout_rate', 'float', 0.1, 0.5),
                    'learning_rate': suggest(trial, 'learning_rate', 'loguniform', 1e-4, 1e-2)
                }

                X, y = v2_create_dataset(data, params['time_step'])
                train_size = int(len(X) * train_split)
                X_train, X_val = X[:train_size], X[train_size:]
                y_train, y_val = y[:train_size], y[train_size:]

                X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
                X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)

                model = v2_build_lstm(X_train.shape[1:], params['units'], params['dropout_rate'], params['learning_rate'])
                early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

                model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32, verbose=0, callbacks=[early_stop])

                val_loss = model.evaluate(X_val, y_val, verbose=0)
                return val_loss
            
            study = optuna.create_study(direction='minimize')
            study.optimize(v2_objective, n_trials=num_trials)
            best_params = study.best_params
            opt_history.store_parameters(stock, 2, best_params)
            return best_params

        def v2_future_predictions_method_1(model, X_test, scaler, future_days):
            future_predictions = model.predict(X_test[-(future_days+1):])
            return scaler.inverse_transform(future_predictions)

        def v2_future_predictions_method_2(model, last_sequence, future_days):
            predictions = []
            current_batch = last_sequence.reshape((1, last_sequence.shape[0], 1))
            for _ in range(future_days+1):
                pred = model.predict(current_batch, verbose=0)[0]
                predictions.append(pred)
                current_batch = np.append(current_batch[:, 1:, :], [[pred]], axis=1)
            return np.array(predictions)

        # Main Workflow
        def v2_main():
            last_price = stock_data['close'].values[-1]
            
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(stock_data['close'].values.reshape(-1, 1))

            best_params = v2_optimize_model(scaled_data)
            time_step = best_params['time_step']

            X, y = v2_create_dataset(scaled_data, time_step)
            train_size = int(len(X) * 0.7)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]

            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_train = v2_augment_data(X_train)  # Augment training data
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

            model = v2_build_lstm(X_train.shape[1:], units=best_params['units'], dropout_rate=best_params['dropout_rate'], learning_rate=best_params['learning_rate'])
            early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

            model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32, callbacks=[early_stop])

            # Method 1
            future_prices_method_1 = v2_future_predictions_method_1(model, X_test, scaler, future_days)

            # Method 2
            last_sequence = scaler.transform(stock_data['close'].values[-time_step:].reshape(-1, 1)).reshape(-1)
            future_predictions = v2_future_predictions_method_2(model, last_sequence, future_days)
            future_prices_method_2 = scaler.inverse_transform(future_predictions)
            
            future_prices_method_2 -= future_prices_method_2[0] - last_price
            future_prices_method_1 -= future_prices_method_1[0] - last_price
            future_prices_method_2 = future_prices_method_2[1:]
            future_prices_method_1 = future_prices_method_1[1:]
            return future_prices_method_1[:data_num], future_prices_method_2[:data_num]
        
        return(v2_main())
    elif i == 3:
        def v3_create_dataset(data, time_step):
            X, y = [], []
            for i in range(len(data) - time_step - 1):
                X.append(data[i:(i + time_step), 0])
                y.append(data[i + time_step, 0])
            return np.array(X), np.array(y)

        def v3_augment_data(data, noise_level=0.02):
            noise = np.random.normal(0, noise_level, data.shape)
            return data + noise

        def v3_build_lstm(input_shape, units=50, dropout_rate=0.3, learning_rate=0.001):
            model = Sequential([
                LSTM(units, return_sequences=True, input_shape=input_shape, kernel_regularizer=l2(0.01)),  # Added L2 regularization
                Dropout(dropout_rate),
                LSTM(units, kernel_regularizer=l2(0.01)),  # Added L2 regularization
                Dropout(dropout_rate),
                Dense(1)
            ])
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
            return model

        def v3_optimize_model(data, train_split=0.7):
            previous_params = opt_history.get_parameters(stock, 3)
            suggest = create_optimization_trial(previous_params)
            num_trials = 40
            if not previous_params:
                if RESET:
                    num_trials = 40
                else:
                    num_trials = 15
            
            def v3_objective(trial):
                params = {
                    'time_step': suggest(trial, 'time_step', 'int', 1, 30),
                    'units': suggest(trial, 'units', 'int', 30, 100),
                    'dropout_rate': suggest(trial, 'dropout_rate', 'float', 0.1, 0.5),
                    'learning_rate': suggest(trial, 'learning_rate', 'loguniform', 1e-4, 1e-2),
                    'epochs': suggest(trial, 'epochs', 'int', 10, 30),
                    'batch_size': suggest(trial, 'batch_size', 'int', 16, 128)
                }

                X, y = v3_create_dataset(data, params['time_step'])
                train_size = int(len(X) * train_split)
                X_train, X_val = X[:train_size], X[train_size:]
                y_train, y_val = y[:train_size], y[train_size:]

                X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
                X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)

                model = v3_build_lstm(X_train.shape[1:], params['units'], params['dropout_rate'], params['learning_rate'])
                early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

                model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=params['epochs'], batch_size=params['batch_size'], verbose=0, callbacks=[early_stop])

                val_loss = model.evaluate(X_val, y_val, verbose=0)
                return val_loss
            study = optuna.create_study(direction='minimize')
            study.optimize(v3_objective, n_trials=num_trials)
            best_params = study.best_params
            opt_history.store_parameters(stock, 3, best_params)
            return best_params

        def v3_future_predictions_method_1(model, X_test, scaler, future_days):
            future_predictions = model.predict(X_test[-(future_days+1):])
            return scaler.inverse_transform(future_predictions)

        def v3_future_predictions_method_2(model, last_sequence, future_days):
            predictions = []
            current_batch = last_sequence.reshape((1, last_sequence.shape[0], 1))
            for _ in range(future_days+1):
                pred = model.predict(current_batch, verbose=0)[0]
                predictions.append(pred)
                current_batch = np.append(current_batch[:, 1:, :], [[pred]], axis=1)
            return np.array(predictions)

        # Main Workflow
        def v3_main():
            last_price = stock_data['close'].values[-1]
            
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(stock_data['close'].values.reshape(-1, 1))

            best_params = v3_optimize_model(scaled_data)
            time_step = best_params['time_step']

            X, y = v3_create_dataset(scaled_data, time_step)
            train_size = int(len(X) * 0.7)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]

            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_train = v3_augment_data(X_train)  # Augment training data
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

            model = v3_build_lstm(X_train.shape[1:], units=best_params['units'], dropout_rate=best_params['dropout_rate'], learning_rate=best_params['learning_rate'])
            early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

            model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=best_params['epochs'], batch_size=best_params['batch_size'], callbacks=[early_stop])

                # Method 1
            future_prices_method_1 = v3_future_predictions_method_1(model, X_test, scaler, future_days)

            # Method 2
            last_sequence = scaler.transform(stock_data['close'].values[-time_step:].reshape(-1, 1)).reshape(-1)
            future_predictions = v3_future_predictions_method_2(model, last_sequence, future_days)
            future_prices_method_2 = scaler.inverse_transform(future_predictions)
            
            future_prices_method_2 -= future_prices_method_2[0] - last_price
            future_prices_method_1 -= future_prices_method_1[0] - last_price
            future_prices_method_2 = future_prices_method_2[1:]
            future_prices_method_1 = future_prices_method_1[1:]
            return future_prices_method_1[:data_num], future_prices_method_2[:data_num]

        return(v3_main())   
    elif i == 4: 
        def v4_create_dataset(data, time_step):
            X, y = [], []
            for i in range(len(data) - time_step - 1):
                X.append(data[i:(i + time_step), 0])
                y.append(data[i + time_step, 0])
            return np.array(X), np.array(y)

        def v4_weighted_loss(y_true, y_pred):
            # Weight earlier errors higher using an exponential decay
            decay_facotr = 0.15
            weights = tf.exp(-tf.range(0, tf.shape(y_true)[0], dtype=tf.float32) * decay_facotr)
            mse = tf.keras.losses.MeanSquaredError()
            loss = mse(y_true, y_pred)
            return tf.reduce_mean(loss * weights)

        def v4_augment_data(data, noise_level=0.02):
            noise = np.random.normal(0, noise_level, data.shape)
            return data + noise

        def v4_build_lstm(input_shape, units=50, dropout_rate=0.3, learning_rate=0.001):
            model = Sequential([
                LSTM(units, return_sequences=True, input_shape=input_shape, kernel_regularizer=l2(0.01)),  # Added L2 regularization
                Dropout(dropout_rate),
                LSTM(units, kernel_regularizer=l2(0.01)),  # Added L2 regularization
                Dropout(dropout_rate),
                Dense(1)
            ])
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss=v4_weighted_loss)
            return model

        def v4_optimize_model(data, train_split=0.7):
            previous_params = opt_history.get_parameters(stock, 4)
            suggest = create_optimization_trial(previous_params)
            num_trials = 40
            if not previous_params:
                if RESET:
                    num_trials = 40
                else: num_trials = 15
            
            def v4_objective(trial):
                params = {
                    'time_step': suggest(trial, 'time_step', 'int', 1, 30),
                    'units': suggest(trial, 'units', 'int', 30, 100),
                    'dropout_rate': suggest(trial, 'dropout_rate', 'float', 0.1, 0.5),
                    'learning_rate': suggest(trial, 'learning_rate', 'loguniform', 1e-4, 1e-2),
                    'epochs': suggest(trial, 'epochs', 'int', 10, 30),
                    'batch_size': suggest(trial, 'batch_size', 'int', 16, 128)
                }

                X, y = v4_create_dataset(data, params['time_step'])
                train_size = int(len(X) * train_split)
                X_train, X_val = X[:train_size], X[train_size:]
                y_train, y_val = y[:train_size], y[train_size:]

                X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
                X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)

                model = v4_build_lstm(X_train.shape[1:], params['units'], params['dropout_rate'], params['learning_rate'])
                early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

                model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=params['epochs'], batch_size=params['batch_size'], verbose=0, callbacks=[early_stop])

                val_loss = model.evaluate(X_val, y_val, verbose=0)
                return val_loss
            study = optuna.create_study(direction='minimize')
            study.optimize(v4_objective, n_trials=num_trials)
            best_params = study.best_params
            opt_history.store_parameters(stock, 4, best_params)
            return best_params

        def v4_future_predictions_method_1(model, X_test, scaler, future_days):
            future_predictions = model.predict(X_test[-(future_days+1):])
            return scaler.inverse_transform(future_predictions)

        def v4_future_predictions_method_2(model, last_sequence, future_days):
            predictions = []
            current_batch = last_sequence.reshape((1, last_sequence.shape[0], 1))
            for _ in range(future_days+1):
                pred = model.predict(current_batch, verbose=0)[0]
                predictions.append(pred)
                current_batch = np.append(current_batch[:, 1:, :], [[pred]], axis=1)
            return np.array(predictions)

        # Main Workflow
        def v4_main():
            last_price = stock_data['close'].values[-1]
            
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(stock_data['close'].values.reshape(-1, 1))

            best_params = v4_optimize_model(scaled_data)
            time_step = best_params['time_step']

            X, y = v4_create_dataset(scaled_data, time_step)
            train_size = int(len(X) * 0.7)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]

            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_train = v4_augment_data(X_train)  # Augment training data
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

            model = v4_build_lstm(X_train.shape[1:], units=best_params['units'], dropout_rate=best_params['dropout_rate'], learning_rate=best_params['learning_rate'])
            early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

            model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=best_params['epochs'], batch_size=best_params['batch_size'], callbacks=[early_stop])

            # Method 1
            future_prices_method_1 = v4_future_predictions_method_1(model, X_test, scaler, future_days)

            # Method 2
            last_sequence = scaler.transform(stock_data['close'].values[-time_step:].reshape(-1, 1)).reshape(-1)
            future_predictions = v4_future_predictions_method_2(model, last_sequence, future_days)
            future_prices_method_2 = scaler.inverse_transform(future_predictions)
            
            future_prices_method_2 -= future_prices_method_2[0] - last_price
            future_prices_method_1 -= future_prices_method_1[0] - last_price
            future_prices_method_2 = future_prices_method_2[1:]
            future_prices_method_1 = future_prices_method_1[1:]
            return future_prices_method_1[:data_num], future_prices_method_2[:data_num]
        
        return(v4_main())   
    elif i == 5:
        def v5_create_dataset(data, time_step):
            X, y = [], []
            for i in range(len(data) - time_step - 1):
                X.append(data[i:(i + time_step), 0])
                y.append(data[i + time_step, 0])
            return np.array(X), np.array(y)

        def v5_weighted_loss(decay_factor):
            def loss_fn(y_true, y_pred):
                # Weight earlier errors higher using an exponential decay
                weights = tf.exp(-tf.range(0, tf.shape(y_true)[0], dtype=tf.float32) * decay_factor)
                mse = tf.keras.losses.MeanSquaredError()
                loss = mse(y_true, y_pred)
                return tf.reduce_mean(loss * weights)
            return loss_fn

        def v5_augment_data(data, noise_level=0.02):
            noise = np.random.normal(0, noise_level, data.shape)
            return data + noise

        def v5_build_lstm(input_shape, units=50, dropout_rate=0.3, learning_rate=0.001, decay_factor=0.2):
            model = Sequential([
                LSTM(units, return_sequences=True, input_shape=input_shape, kernel_regularizer=l2(0.01)),
                Dropout(dropout_rate),
                LSTM(units, kernel_regularizer=l2(0.01)),
                Dropout(dropout_rate),
                Dense(1)
            ])
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss=v5_weighted_loss(decay_factor))
            return model

        def v5_optimize_model(data, train_split=0.7):
            previous_params = opt_history.get_parameters(stock, 5)
            suggest = create_optimization_trial(previous_params)
            num_trials = 50
            if not previous_params:
                if RESET:
                    num_trials = 50
                else: num_trials = 20
            
            
            def v5_objective(trial):
                params = {
                    'time_step': suggest(trial, 'time_step', 'int', 1, 30),
                    'units': suggest(trial, 'units', 'int', 30, 100),
                    'dropout_rate': suggest(trial, 'dropout_rate', 'float', 0.1, 0.5),
                    'learning_rate': suggest(trial, 'learning_rate', 'loguniform', 1e-4, 1e-2),
                    'epochs': suggest(trial, 'epochs', 'int', 10, 30),
                    'batch_size': suggest(trial, 'batch_size', 'int', 16, 128),
                    'decay_factor': suggest(trial, 'decay_factor', 'float', 0.2, 0.5)
                }

                X, y = v5_create_dataset(data, params['time_step'])
                train_size = int(len(X) * train_split)
                X_train, X_val = X[:train_size], X[train_size:]
                y_train, y_val = y[:train_size], y[train_size:]

                X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
                X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)

                model = v5_build_lstm(X_train.shape[1:], params['units'], params['dropout_rate'], params['learning_rate'], params['decay_factor'])
                early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

                model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=params['epochs'], batch_size=params['batch_size'], verbose=0, callbacks=[early_stop])

                val_loss = model.evaluate(X_val, y_val, verbose=0)
                return val_loss

            study = optuna.create_study(direction='minimize')
            study.optimize(v5_objective, n_trials=num_trials)
            best_params = study.best_params
            opt_history.store_parameters(stock, 5, best_params)
            return best_params

        def v5_future_predictions_method_1(model, X_test, scaler, future_days):
            future_predictions = model.predict(X_test[-(future_days+1):])
            return scaler.inverse_transform(future_predictions)

        def v5_future_predictions_method_2(model, last_sequence, future_days):
            predictions = []
            current_batch = last_sequence.reshape((1, last_sequence.shape[0], 1))
            for _ in range(future_days+1):
                pred = model.predict(current_batch, verbose=0)[0]
                predictions.append(pred)
                current_batch = np.append(current_batch[:, 1:, :], [[pred]], axis=1)
            return np.array(predictions)

        # Main Workflow
        def v5_main():
            last_price = stock_data['close'].values[-1]
            
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(stock_data['close'].values.reshape(-1, 1))

            best_params = v5_optimize_model(scaled_data)
            time_step = best_params['time_step']

            X, y = v5_create_dataset(scaled_data, time_step)
            train_size = int(len(X) * 0.7)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]

            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_train = v5_augment_data(X_train)  # Augment training data
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

            model = v5_build_lstm(X_train.shape[1:], units=best_params['units'], dropout_rate=best_params['dropout_rate'], learning_rate=best_params['learning_rate'], decay_factor=best_params['decay_factor'])
            early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

            model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=best_params['epochs'], batch_size=best_params['batch_size'], callbacks=[early_stop])

            # Method 1
            future_prices_method_1 = v5_future_predictions_method_1(model, X_test, scaler, future_days)

            # Method 2
            last_sequence = scaler.transform(stock_data['close'].values[-time_step:].reshape(-1, 1)).reshape(-1)
            future_predictions = v5_future_predictions_method_2(model, last_sequence, future_days)
            future_prices_method_2 = scaler.inverse_transform(future_predictions)
            
            future_prices_method_2 -= future_prices_method_2[0] - last_price
            future_prices_method_1 -= future_prices_method_1[0] - last_price
            future_prices_method_2 = future_prices_method_2[1:]
            future_prices_method_1 = future_prices_method_1[1:]
            return future_prices_method_1[:data_num], future_prices_method_2[:data_num]  
    
        return(v5_main()) 
    elif i == 6:
        def v6_create_dataset(data, time_step):
            X, y = [], []
            for i in range(len(data) - time_step - 1):
                X.append(data[i:(i + time_step), 0])
                y.append(data[i + time_step, 0])
            return np.array(X), np.array(y)

        def v6_weighted_loss(decay_factor):
            def loss_fn(y_true, y_pred):
                # Weight earlier errors higher using an exponential decay
                weights = tf.exp(-tf.range(0, tf.shape(y_true)[0], dtype=tf.float32) * decay_factor)
                mse = tf.keras.losses.MeanSquaredError()
                loss = mse(y_true, y_pred)
                return tf.reduce_mean(loss * weights)
            return loss_fn

        def v6_augment_data(data, noise_level=0.02):
            noise = np.random.normal(0, noise_level, data.shape)
            return data + noise

        def v6_build_lstm(input_shape, units=50, dropout_rate=0.3, learning_rate=0.001, decay_factor=0.2):
            model = Sequential([
                LSTM(units, return_sequences=True, input_shape=input_shape, kernel_regularizer=l2(0.01)),
                Dropout(dropout_rate),
                LSTM(units, kernel_regularizer=l2(0.01)),
                Dropout(dropout_rate),
                Dense(1)
            ])
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss=v6_weighted_loss(decay_factor))
            return model

        def v6_optimize_model(data, train_split=0.7):
            previous_params = opt_history.get_parameters(stock, 6)
            suggest = create_optimization_trial(previous_params)
            num_trials = 50
            if not previous_params:
                if RESET:
                    num_trials = 50
                else: num_trials = 20
            
            
            def v6_objective(trial):
                # Tuneable hyperparameters
                params = {
                    'time_step': suggest(trial, 'time_step', 'int', 1, 30),
                    'units': suggest(trial, 'units', 'int', 30, 100),
                    'dropout_rate': suggest(trial, 'dropout_rate', 'float', 0.1, 0.5),
                    'learning_rate': suggest(trial, 'learning_rate', 'loguniform', 1e-4, 1e-2),
                    'epochs': suggest(trial, 'epochs', 'int', 10, 30),
                    'batch_size': suggest(trial, 'batch_size', 'int', 16, 128),
                    'decay_factor': suggest(trial, 'decay_factor', 'float', 0.2, 0.5),
                    'future_days': suggest(trial, 'future_days', 'int', 1, 50)
                }

                # Create the dataset
                X, y = v6_create_dataset(data, params['time_step'])
                train_size = int(len(X) * train_split)
                X_train, _ = X[:train_size], X[train_size:]
                y_train, _ = y[:train_size], y[train_size:]

                # Reshape training data
                X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

                # Build the LSTM model
                model = v6_build_lstm(X_train.shape[1:], params['units'], params['dropout_rate'], params['learning_rate'], params['decay_factor'])
                early_stop = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)

                # Train the model
                model.fit(X_train, y_train, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0, callbacks=[early_stop])

                # Evaluate on a sequence of `future_days` from training data
                eval_start_idx = -(future_days + params['time_step'])
                X_eval = X_train[eval_start_idx:-future_days]
                y_eval = y_train[-future_days:]

                # Predict future_days directly from the training dataset
                predictions = model.predict(X_eval, verbose=0).flatten()

                # Align lengths of predictions and y_eval
                min_length = min(len(predictions), len(y_eval))
                predictions = predictions[:min_length]
                y_eval = y_eval[:min_length]

                # Calculate loss between predictions and actual future values
                loss = tf.keras.losses.MeanSquaredError()(y_eval, predictions)

                return loss.numpy()

            # Use Optuna to find the best hyperparameters
            study = optuna.create_study(direction='minimize')
            study.optimize(v6_objective, n_trials=num_trials)
            best_params = study.best_params
            opt_history.store_parameters(stock, 6, best_params)
            return best_params

        def v6_future_predictions_method_1(model, X_test, scaler, future_days):
            future_predictions = model.predict(X_test[-(future_days+1):])
            return scaler.inverse_transform(future_predictions)


        def v6_future_predictions_method_2(model, last_sequence, future_days):
            predictions = []
            current_batch = last_sequence.reshape((1, last_sequence.shape[0], 1))
            for _ in range(future_days+1):
                pred = model.predict(current_batch, verbose=0)[0]
                predictions.append(pred)
                current_batch = np.append(current_batch[:, 1:, :], [[pred]], axis=1)
            return np.array(predictions)

        # Main Workflow
        def v6_main():
            last_price = stock_data['close'].values[-1]
            
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(stock_data['close'].values.reshape(-1, 1))

            best_params = v6_optimize_model(scaled_data)
            time_step = best_params['time_step']

            X, y = v6_create_dataset(scaled_data, time_step)
            train_size = int(len(X) * 0.7)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]

            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_train = v6_augment_data(X_train)  # Augment training data
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

            model = v6_build_lstm(X_train.shape[1:], units=best_params['units'], dropout_rate=best_params['dropout_rate'], learning_rate=best_params['learning_rate'], decay_factor=best_params['decay_factor'])
            early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

            model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=best_params['epochs'], batch_size=best_params['batch_size'], callbacks=[early_stop])

            # Method 1
            future_prices_method_1 = v6_future_predictions_method_1(model, X_test, scaler, future_days)

            # Method 2
            last_sequence = scaler.transform(stock_data['close'].values[-time_step:].reshape(-1, 1)).reshape(-1)
            future_predictions = v6_future_predictions_method_2(model, last_sequence, future_days)
            future_prices_method_2 = scaler.inverse_transform(future_predictions)
            
            future_prices_method_2 -= future_prices_method_2[0] - last_price
            future_prices_method_1 -= future_prices_method_1[0] - last_price
            future_prices_method_2 = future_prices_method_2[1:]
            future_prices_method_1 = future_prices_method_1[1:]
            return future_prices_method_1[:data_num], future_prices_method_2[:data_num]

        return(v6_main())

def dates():
    dates = [
        datetime(2022, 1, 1),
        datetime(2023, 2, 6),
        datetime(2024, 1, 9),
        datetime(2024, 5, 2),
        datetime(2024, 11, 1),
    ]

    # Create a new list to store results
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
    return(result_dates)

def create_table_if_not_exists(stock, start_date):
    """
    Create a table for the stock with a specific start date if it doesn't exist.
    """
    # Format the start date
    start_date_str = start_date.strftime("%Y_%m_%d")
    table_name = f"{stock}_{start_date_str}"

    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        last_date TEXT PRIMARY KEY,
        last_open REAL,
        last_close REAL,
        last_rsi REAL,
        last_williams REAL,
        last_adx REAL,
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
        pred_6_2 TEXT
    )
    """
    cursor.execute(create_table_query)
    conn.commit()
    return table_name

def insert_predictions(table_name, last_date, last_open, last_close, last_rsi, last_williams, last_adx, predictions):
    """
    Insert prediction data into the specified table.
    """
    # Ensure last_date is a string (e.g., from a Timestamp or Index)
    last_date = str(last_date)

    # Ensure numeric inputs are of float type
    last_open = float(last_open)
    last_close = float(last_close)
    last_rsi = float(last_rsi)
    last_williams = float(last_williams)
    last_adx = float(last_adx)

    # SQL query
    insert_query = f"""
    INSERT OR REPLACE INTO {table_name} (
        last_date, last_open, last_close, last_rsi, last_williams, last_adx,
        pred_1_1, pred_1_2, pred_2_1, pred_2_2, pred_3_1, pred_3_2,
        pred_4_1, pred_4_2, pred_5_1, pred_5_2, pred_6_1, pred_6_2
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    
    # Execute query with the predictions unpacked into individual values
    cursor.execute(insert_query, (
        last_date, last_open, last_close, last_rsi, last_williams, last_adx,
        *predictions  # Unpack the predictions list into individual columns
    ))
    conn.commit()

def array_to_comma_separated_string(array: np.ndarray) -> str:
    """
    Converts a numpy.ndarray into a comma-separated string.

    Parameters:
        array (np.ndarray): The input numpy array.

    Returns:
        str: A string of the array's elements, separated by commas.
    """
    if not isinstance(array, np.ndarray):
        raise ValueError("Input must be a numpy.ndarray")

    # Flatten the array to handle multidimensional arrays
    flat_array = array.flatten()

    # Convert the elements to strings and join with commas
    return ",".join(map(str, flat_array))

conn = sqlite3.connect("trading_algo.db")
cursor = conn.cursor()
opt_history = OptimizationHistory()
api_token = 'lFVm52EqS8EuypuH9FqhzhMAbo7zbeNb'
technical_indicators = ['williams', 'rsi', 'adx']
market_days = get_market_days(2015, 2026)
data_num = 8
future_days = 30
future_date = datetime.now()
start_date = future_date - timedelta(days=3000)
full_dates = dates()
stocks = [
    'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'GOOG', 'META', 'BRK.B', 'TSLA', 'UNH',
    'LLY', 'JPM', 'XOM', 'JNJ', 'V', 'PG', 'AVGO', 'MA', 'HD', 'CVX',
    'MRK', 'ABBV', 'PEP', 'COST', 'ADBE', 'KO', 'CSCO', 'WMT', 'TMO', 'MCD',
    'PFE', 'CRM', 'BAC', 'ACN', 'CMCSA', 'LIN', 'NFLX', 'ABT', 'ORCL', 'DHR',
    'AMD', 'WFC', 'DIS', 'TXN', 'PM', 'VZ', 'INTU', 'COP', 'CAT', 'AMGN'
]

for stock in stocks:
    print(f'Predicitons for {stock}')
    full_data = fetch_technical_data(stock, start_date, future_date, technical_indicators)
    for dates_ in full_dates:
        table_date = dates_[0]
        print(f'{stock} predicitons starting at {str(table_date)})')
        table_name = create_table_if_not_exists(stock, table_date)
        for day, date in enumerate(dates_):
            print(f'Day {day}')
            stock_data = full_data.loc[start_date:date]
            if day % 5 == 0:
                RESET = True
            else:
                RESET = False
            last_row = stock_data.iloc[-1]
            last_date = last_row.name.date()
            last_close = last_row['close']
            last_open = last_row['open']
            last_rsi = last_row['rsi']
            last_williams = last_row['williams']
            last_adx = last_row['adx']
            
            predictions = []
            for i in range(1, 7):
                future_prices_method_1, future_prices_method_2 = versions(i, RESET)
                # Convert to comma-separated strings
                future_prices_method_1 = array_to_comma_separated_string(future_prices_method_1)
                future_prices_method_2 = array_to_comma_separated_string(future_prices_method_2)

                predictions.append(future_prices_method_1)
                predictions.append(future_prices_method_2)
                
            insert_predictions(
                table_name, last_date, last_open, last_close,
                last_rsi, last_williams, last_adx, predictions
            )
conn.close()