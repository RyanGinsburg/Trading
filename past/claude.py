import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential # type: ignore
from keras.layers import LSTM, Dense, Dropout # type: ignore
from keras.callbacks import EarlyStopping # type: ignore
from keras.regularizers import l2 # type: ignore
from keras.optimizers import Adam # type: ignore
import optuna
from datetime import datetime, timedelta
import pandas_market_calendars as mcal
from typing import Tuple, List, Dict, Any
import logging
import requests

class StockPricePredictor:
    def __init__(self, stock: str, api_token: str, future_days: int = 20):
        """Initialize the stock price predictor with configuration."""
        self.stock = stock
        self.api_token = api_token
        self.future_days = future_days
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.setup_logging()

    def setup_logging(self):
        """Configure logging for the predictor."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def get_market_days(self, start_year: int, end_year: int, exchange: str = 'NYSE') -> List:
        """Get trading days for the specified exchange."""
        market_calendar = mcal.get_calendar(exchange)
        schedule = market_calendar.schedule(
            start_date=f"{start_year}-01-01",
            end_date=f"{end_year}-12-31"
        )
        return schedule.index.to_list()

    def fetch_technical_data(self, start_date: datetime, end_date: datetime,
                           technical_indicators: List[str] = None) -> pd.DataFrame:
        """Fetch technical indicators data for the stock."""
        if technical_indicators is None:
            technical_indicators = ['ema']  # Default indicator

        df_list = []
        for i, indicator in enumerate(technical_indicators):
            url = (f'https://financialmodelingprep.com/api/v3/technical_indicator/1day/'
                  f'{self.stock}?type={indicator}&period=1300&apikey={self.api_token}')
            
            try:
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
                
            except Exception as e:
                self.logger.error(f"Error fetching data for {indicator}: {str(e)}")
                raise
                
        combined_df = pd.concat(df_list, axis=1)
        return combined_df.loc[pd.to_datetime(start_date):pd.to_datetime(end_date)]

    def create_dataset(self, data: np.ndarray, time_step: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create time series dataset for LSTM training."""
        X, y = [], []
        for i in range(len(data) - time_step - 1):
            X.append(data[i:(i + time_step), 0])
            y.append(data[i + time_step, 0])
        return np.array(X), np.array(y)

    def custom_loss(self, decay_factor: float = 0.2):
        """Create a custom loss function with temporal decay."""
        def loss_fn(y_true, y_pred):
            weights = tf.exp(-tf.range(0, tf.shape(y_true)[0], dtype=tf.float32) * decay_factor)
            squared_error = tf.square(y_true - y_pred)
            return tf.reduce_mean(squared_error * weights)
        return loss_fn

    def build_model(self, input_shape: tuple, params: Dict[str, Any]) -> Sequential:
        """Build and compile the LSTM model."""
        model = Sequential([
            LSTM(params['units'], 
                 return_sequences=True, 
                 input_shape=input_shape,
                 kernel_regularizer=l2(0.01)),
            Dropout(params['dropout_rate']),
            LSTM(params['units'],
                 kernel_regularizer=l2(0.01)),
            Dropout(params['dropout_rate']),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=params['learning_rate']),
            loss=self.custom_loss(params.get('decay_factor', 0.2))
        )
        return model

    def optimize_hyperparameters(self, data: np.ndarray, n_trials: int = 50) -> Dict[str, Any]:
        """Optimize model hyperparameters using Optuna."""
        def objective(trial):
            params = {
                'time_step': trial.suggest_int('time_step', 1, 30),
                'units': trial.suggest_int('units', 30, 100),
                'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
                'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-2),
                'decay_factor': trial.suggest_float('decay_factor', 0.1, 0.5),
                'batch_size': trial.suggest_int('batch_size', 16, 128),
                'epochs': trial.suggest_int('epochs', 10, 30)
            }

            X, y = self.create_dataset(data, params['time_step'])
            train_size = int(len(X) * 0.7)
            X_train = X[:train_size].reshape(-1, params['time_step'], 1)
            y_train = y[:train_size]

            model = self.build_model(X_train.shape[1:], params)
            early_stop = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
            
            history = model.fit(
                X_train, y_train,
                epochs=params['epochs'],
                batch_size=params['batch_size'],
                callbacks=[early_stop],
                verbose=0
            )
            
            return min(history.history['loss'])

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        return study.best_params

    def predict_future_prices(self, model: Sequential, X_test: np.ndarray, 
                            last_sequence: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate future price predictions using two different methods."""
        # Method 1: Direct prediction
        future_predictions_1 = model.predict(X_test[-(self.future_days+1):])
        prices_1 = self.scaler.inverse_transform(future_predictions_1)

        # Method 2: Sequential prediction
        predictions = []
        current_batch = last_sequence.reshape((1, last_sequence.shape[0], 1))
        
        for _ in range(self.future_days + 1):
            pred = model.predict(current_batch, verbose=0)[0]
            predictions.append(pred)
            current_batch = np.append(current_batch[:, 1:, :], [[pred]], axis=1)
        
        prices_2 = self.scaler.inverse_transform(np.array(predictions))
        
        return prices_1, prices_2

    def train_and_predict(self, stock_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Main workflow for training the model and generating predictions."""
        last_price = stock_data['close'].values[-1]
        scaled_data = self.scaler.fit_transform(stock_data['close'].values.reshape(-1, 1))
        
        # Optimize hyperparameters
        best_params = self.optimize_hyperparameters(scaled_data)
        self.logger.info(f"Best hyperparameters found: {best_params}")
        
        # Prepare training data
        X, y = self.create_dataset(scaled_data, best_params['time_step'])
        train_size = int(len(X) * 0.7)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        
        # Train model
        model = self.build_model(X_train.shape[1:], best_params)
        early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        
        model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=best_params['epochs'],
            batch_size=best_params['batch_size'],
            callbacks=[early_stop]
        )
        
        # Generate predictions
        last_sequence = scaled_data[-best_params['time_step']:]
        future_prices_1, future_prices_2 = self.predict_future_prices(model, X_test, last_sequence)
        
        # Adjust predictions relative to last known price
        future_prices_1 = self._adjust_predictions(future_prices_1, last_price)
        future_prices_2 = self._adjust_predictions(future_prices_2, last_price)
        
        return future_prices_1[:8], future_prices_2[:8], best_params

    def _adjust_predictions(self, predictions: np.ndarray, last_price: float) -> np.ndarray:
        """Adjust predictions relative to the last known price."""
        adjusted = predictions.copy()
        adjusted -= adjusted[0] - last_price
        return adjusted[1:]

# Example usage
if __name__ == "__main__":
    api_token = 'lFVm52EqS8EuypuH9FqhzhMAbo7zbeNb'
    predictor = StockPricePredictor('AAPL', api_token)
    
    end_date = datetime.now() - timedelta(days=270)
    start_date = end_date - timedelta(days=3000)
    
    # Fetch and prepare data
    technical_data = predictor.fetch_technical_data(start_date, end_date)
    
    # Generate predictions
    future_prices_1, future_prices_2, best_params = predictor.train_and_predict(technical_data)
    
    print("Method 1 Predictions:", future_prices_1)
    print("Method 2 Predictions:", future_prices_2)
    print("Best Parameters:", best_params)