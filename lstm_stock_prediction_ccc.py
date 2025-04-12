import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def load_data(file_path, sequence_length=5, use_sentiment=True):
    # Load the data
    df = pd.read_csv(file_path)
    
    # Convert sentiment to numerical values
    sentiment_map = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
    df['Sentiment'] = df['Sentiment'].map(sentiment_map)
    
    # Select features
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    if use_sentiment:
        features.append('Sentiment')
    data = df[features].values
    
    # Normalize the data
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    
    # Create sequences
    X, y = [], []
    close_idx = features.index('Close')
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        # Target is the next day's close price
        y.append(data[i + sequence_length][close_idx])
    
    return np.array(X), np.array(y), scaler

def build_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1)  # Linear activation for regression
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.0005),
                 loss='mean_squared_error',
                 metrics=['mean_absolute_error'])
    return model

def main():
    # Parameters
    sequence_length = 10
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15
    epochs = 150
    batch_size = 64
    learning_rate = 0.0005
    # use_sentiment = True
    use_sentiment = False

    price_change_threshold = 0.01  # 1% threshold for price changes
    verbose = 1  # Set to 1 to show training progress, 0 to hide it
    
    # Load and prepare data
    # Choose stock name
    stock_name = 'AMZN'  
    X, y, scaler = load_data(f'combined_{stock_name}.csv', sequence_length, use_sentiment)
    
    # Split data in chronological order
    train_size = int(len(X) * train_ratio)
    val_size = int(len(X) * val_ratio)
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]
    
    # Build model
    model = build_model((sequence_length, X.shape[2]))
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        verbose=verbose
    )
    
    # Evaluate model
    train_score = model.evaluate(X_train, y_train, verbose=0)
    val_score = model.evaluate(X_val, y_val, verbose=0)
    test_score = model.evaluate(X_test, y_test, verbose=0)
    
    # Predict test set
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Training MSE: {train_score[0]:.4f}, MAE: {train_score[1]:.4f}")
    print(f"Validation MSE: {val_score[0]:.4f}, MAE: {val_score[1]:.4f}")
    print(f"Testing MSE: {test_score[0]:.4f}, MAE: {test_score[1]:.4f}")
    print(f"R2 Score: {r2:.4f}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['mean_absolute_error'], label='Train')
    plt.plot(history.history['val_mean_absolute_error'], label='Validation')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss (MSE)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    
    # Plot actual vs predicted prices
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label='Actual Price')
    plt.plot(y_pred, label='Predicted Price')
    plt.title('Actual vs Predicted Stock Prices')
    plt.xlabel('Time')
    plt.ylabel('Normalized Price')
    plt.legend()
    plt.savefig('price_predictions.png')
    plt.close()

if __name__ == "__main__":
    main() 