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
    def v1_objective(trial):
        time_step = trial.suggest_int('time_step', 1, 30)
        units = trial.suggest_int('units', 30, 100)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)

        X, y = v1_create_dataset(data, time_step)
        train_size = int(len(X) * train_split)
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]

        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)

        model = v1_build_lstm(X_train.shape[1:], units, dropout_rate, learning_rate)
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32, verbose=0, callbacks=[early_stop])

        val_loss = model.evaluate(X_val, y_val, verbose=0)
        return val_loss

    study = optuna.create_study(direction='minimize')
    study.optimize(v1_objective, n_trials=30)
    return study.best_params

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
    return future_prices_method_1[:8], future_prices_method_2[:8]
