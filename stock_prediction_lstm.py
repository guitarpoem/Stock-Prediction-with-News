import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Parameters
LAG_WINDOW = 5  # Size of the lag window (Δd)
TEST_SIZE = 0.2  # Proportion of data to use for testing

def load_and_prepare_data(file_path):
    """Load and prepare the dataset."""
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Sort by date
    df = df.sort_values('date')
    
    # Convert sentiment to numerical values
    # Treat 'Missing' as a fourth category
    sentiment_map = {'Positive': 1, 'Neutral': 0, 'Negative': -1, 'Missing': 2}
    df['sentiment_numeric'] = df['Sentiment'].map(sentiment_map)
    
    # Select relevant features
    features = ['movement_percent', 'open_price', 'high_price', 'low_price', 
                'close_price', 'volume', 'sentiment_numeric']
    
    return df[['date'] + features]

def create_sequences(data, lag_window):
    """Create sequences for LSTM model with multiple prediction targets."""
    X, y = [], []
    
    for i in range(len(data) - lag_window):
        # Input sequence: features from days i to i+lag_window-1
        seq = data[i:i+lag_window].values
        
        # Target: movement direction for each day in the window
        # 1 if movement_percent > 0 else 0
        targets = (data['movement_percent'].iloc[i:i+lag_window].values > 0).astype(int)
        
        X.append(seq)
        y.append(targets)
    
    return np.array(X), np.array(y)

def build_model(input_shape, output_length):
    """Build and compile the LSTM model."""
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(output_length, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    # Load and prepare data
    print("Loading and preparing data...")
    data = load_and_prepare_data('dataset/AAPL.csv')
    print(f"Data shape: {data.shape}")
    
    # Normalize numerical features
    scaler = MinMaxScaler()
    features = data.columns[1:]  # Exclude date
    data_scaled = pd.DataFrame(scaler.fit_transform(data[features]), columns=features)
    
    # Create sequences
    print("Creating sequences...")
    X, y = create_sequences(data_scaled, LAG_WINDOW)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    # Split into training and testing sets
    split_idx = int(len(X) * (1 - TEST_SIZE))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Build and train the model
    print("Building and training the model...")
    model = build_model(input_shape=(LAG_WINDOW, len(features)), output_length=LAG_WINDOW)
    
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Evaluate the model
    print("Evaluating the model...")
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # Focus on the prediction for the last day in each window
    y_test_last_day = y_test[:, -1]
    y_pred_last_day = y_pred_binary[:, -1]
    
    # Calculate accuracy for the last day predictions
    last_day_accuracy = accuracy_score(y_test_last_day, y_pred_last_day)
    print(f"Last Day Prediction Accuracy: {last_day_accuracy:.4f}")
    
    # Print classification report for the last day
    print("\nClassification Report for Last Day Predictions:")
    print(classification_report(y_test_last_day, y_pred_last_day))
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    
    # Plot actual vs predicted movements for a sample period
    sample_size = min(50, len(y_test_last_day))
    plt.figure(figsize=(15, 6))
    plt.plot(y_test_last_day[:sample_size], label='Actual Movement', marker='o')
    plt.plot(y_pred_last_day[:sample_size], label='Predicted Movement', marker='x')
    plt.title('Actual vs Predicted Stock Movements (Last Day of Each Window)')
    plt.xlabel('Sample Index')
    plt.ylabel('Movement (1=Up, 0=Down)')
    plt.legend()
    plt.grid(True)
    plt.savefig('prediction_results.png')
    plt.close()
    
    print("Analysis complete. Check the generated plots for visualization.")

if __name__ == "__main__":
    main() 