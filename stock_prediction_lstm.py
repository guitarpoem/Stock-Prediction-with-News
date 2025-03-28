import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
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
    """Create sequences for LSTM model to predict the next day's movement."""
    X, y = [], []
    
    for i in range(len(data) - lag_window - 1):
        # Input sequence: features from days i to i+lag_window-1
        seq = data[i:i+lag_window].values
        
        # Target: movement direction for the next day (day i+lag_window)
        # 1 if movement_percent > 0 else 0
        target = 1 if data['movement_percent'].iloc[i+lag_window] > 0 else 0
        
        X.append(seq)
        y.append(target)
    
    return np.array(X), np.array(y)

def build_model(input_shape):
    """Build and compile the LSTM model."""
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
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
    # scaler = MinMaxScaler()
    features = data.columns[1:]  # Exclude date
    # data_scaled = pd.DataFrame(scaler.fit_transform(data[features]), columns=features)
    data_scaled = data[features]  # Use raw data without normalization
    
    # Create sequences
    print("Creating sequences...")
    X, y = create_sequences(data_scaled, LAG_WINDOW)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    # Display sample data
    print("\nSample X data (first 3 sequences):")
    for i in range(min(3, len(X))):
        print(f"Sequence {i+1}:")
        sample_df = pd.DataFrame(X[i], columns=features)
        print(sample_df)
        print(f"Target y[{i}]: {y[i]} ({'Up' if y[i] == 1 else 'Down'})")
        print()
    
    # Split into training and testing sets
    split_idx = int(len(X) * (1 - TEST_SIZE))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Build and train the model
    print("Building and training the model...")
    model = build_model(input_shape=(LAG_WINDOW, len(features)))
    
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
    y_pred_binary = (y_pred > 0.5).astype(int).flatten()
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred_binary)
    print(f"Prediction Accuracy: {accuracy:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_binary))
    
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
    sample_size = min(50, len(y_test))
    plt.figure(figsize=(15, 6))
    plt.plot(y_test[:sample_size], label='Actual Movement', marker='o')
    plt.plot(y_pred_binary[:sample_size], label='Predicted Movement', marker='x')
    plt.title('Actual vs Predicted Stock Movements (Next Day)')
    plt.xlabel('Sample Index')
    plt.ylabel('Movement (1=Up, 0=Down)')
    plt.legend()
    plt.grid(True)
    plt.savefig('prediction_results.png')
    plt.close()
    
    print("Analysis complete. Check the generated plots for visualization.")

if __name__ == "__main__":
    main() 