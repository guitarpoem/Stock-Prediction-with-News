import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def calculate_rsi(data, period=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    exp1 = data.ewm(span=fast_period, adjust=False).mean()
    exp2 = data.ewm(span=slow_period, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    return macd, signal

def calculate_bollinger_bands(data, period=20, num_std=2):
    sma = data.rolling(window=period).mean()
    std = data.rolling(window=period).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return upper_band, lower_band

def calculate_sma(data, period=20):
    return data.rolling(window=period).mean()

def load_data(file_path, sequence_length=5, use_sentiment=True, use_technical_indicators=True):
    # Load the data
    df = pd.read_csv(file_path)
    
    # Convert sentiment to numerical values
    sentiment_map = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
    df['Sentiment'] = df['Sentiment'].map(sentiment_map)
    
    # Select basic features
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    if use_sentiment:
        features.insert(0, 'Sentiment')
    
    # Add technical indicators if enabled
    if use_technical_indicators:
        # Calculate indicators
        df['RSI'] = calculate_rsi(df['Close'])
        macd, signal = calculate_macd(df['Close'])
        df['MACD'] = macd
        df['MACD_Signal'] = signal
        upper_band, lower_band = calculate_bollinger_bands(df['Close'])
        df['BB_Upper'] = upper_band
        df['BB_Lower'] = lower_band
        df['SMA_20'] = calculate_sma(df['Close'])
        
        # Add technical indicators to features list
        technical_features = ['RSI', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower', 'SMA_20']
        features.extend(technical_features)
    
    data = df[features].values
    
    # Normalize the data
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    
    # Create sequences
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        # Create binary target: 1 if next day's close is higher than current day's close
        y.append(1 if data[i + sequence_length][4] > data[i + sequence_length - 1][4] else 0)
    
    return np.array(X), np.array(y), scaler

def build_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.0005),
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    return model

def main():
    # Parameters
    sequence_length = 5
    test_size = 0.2
    random_state = 42
    epochs = 150
    batch_size = 64
    learning_rate = 0.0001
    use_sentiment = False
    use_technical_indicators = True  # New parameter to control technical indicators
    verbose = 1
    early_stopping_patience = 30
    
    stock_name = 'AMZN'

    # Load and prepare data
    X, y, scaler = load_data(f'combined_{stock_name}.csv', sequence_length, use_sentiment, use_technical_indicators)
    
    # Split data chronologically (last 20% for testing)
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Build model
    model = build_model((sequence_length, X.shape[2]))
    
    # Configure early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=early_stopping_patience,
        restore_best_weights=True,
        verbose=0
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        verbose=verbose,
        callbacks=[early_stopping]
    )
    
    # Evaluate model
    train_score = model.evaluate(X_train, y_train, verbose=0)
    test_score = model.evaluate(X_test, y_test, verbose=0)
    
    # Predict test set
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    print(f"Training Accuracy: {train_score[1]:.4f}")
    print(f"Testing Accuracy: {test_score[1]:.4f}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'training_history_{stock_name}.png')
    plt.close()

if __name__ == "__main__":
    main() 