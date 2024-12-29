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
    def v6_objective(trial):
        # Tuneable hyperparameters
        time_step = trial.suggest_int('time_step', 1, 30)
        units = trial.suggest_int('units', 30, 100)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
        epochs = trial.suggest_int('epochs', 10, 30)
        batch_size = trial.suggest_int('batch_size', 16, 128)
        decay_factor = trial.suggest_float('decay_factor', 0.2, 0.5)
        future_days = trial.suggest_int('future_days', 1, 50)  # Tuneable future_days

        # Create the dataset
        X, y = v6_create_dataset(data, time_step)
        train_size = int(len(X) * train_split)
        X_train, _ = X[:train_size], X[train_size:]
        y_train, _ = y[:train_size], y[train_size:]

        # Reshape training data
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

        # Build the LSTM model
        model = v6_build_lstm(X_train.shape[1:], units, dropout_rate, learning_rate, decay_factor)
        early_stop = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)

        # Train the model
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[early_stop])

        # Evaluate on a sequence of `future_days` from training data
        eval_start_idx = -(future_days + time_step)
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
    study.optimize(v6_objective, n_trials=50)

    return study.best_params


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
    return future_prices_method_1[:8], future_prices_method_2[:8]