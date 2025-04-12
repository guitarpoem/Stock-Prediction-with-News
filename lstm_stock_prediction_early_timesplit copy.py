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
        # Create binary target: 1 if next day's close is higher than current day's close
        y.append(1 if data[i + sequence_length][close_idx] > data[i + sequence_length - 1][close_idx] else 0)
    
    return np.array(X), np.array(y), scaler

def build_model(input_shape, learning_rate):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    # Add gradient clipping to the optimizer
    optimizer = Adam(learning_rate, clipnorm=1.0)
    
    model.compile(optimizer=optimizer,
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
    learning_rate = 0.0005
    use_sentiment = True
    # use_sentiment = False
    verbose = 1  # Set to 1 to show training progress, 0 to hide it
    early_stopping_patience = 30  # Number of epochs to wait before early stopping
    
    stock_name = 'AAPL'

    # Load and prepare data
    X, y, scaler = load_data(f'combined_{stock_name}.csv', sequence_length, use_sentiment)
    
    # Split data chronologically (last 20% for testing)
    split_idx = int(len(X) * (1 - test_size))
    X_train_val, X_test = X[:split_idx], X[split_idx:]
    y_train_val, y_test = y[:split_idx], y[split_idx:]
    
    # Split training data into train and validation sets (20% of remaining data for validation)
    val_size = 0.2
    val_split_idx = int(len(X_train_val) * (1 - val_size))
    X_train, X_val = X_train_val[:val_split_idx], X_train_val[val_split_idx:]
    y_train, y_val = y_train_val[:val_split_idx], y_train_val[val_split_idx:]
    
    # Print data split information
    print("\nData Split Information:")
    print(f"Total samples: {len(X)}")
    print(f"Training samples: {len(X_train)} ({len(X_train)/len(X)*100:.2f}%)")
    print(f"Validation samples: {len(X_val)} ({len(X_val)/len(X)*100:.2f}%)")
    print(f"Testing samples: {len(X_test)} ({len(X_test)/len(X)*100:.2f}%)")
    
    # Print distribution of classes in train, validation and test sets
    print("\nClass Distribution:")
    print("Training set:")
    print(f"Class 0 (Price Down): {np.sum(y_train == 0)} ({np.sum(y_train == 0)/len(y_train)*100:.2f}%)")
    print(f"Class 1 (Price Up): {np.sum(y_train == 1)} ({np.sum(y_train == 1)/len(y_train)*100:.2f}%)")
    print("\nValidation set:")
    print(f"Class 0 (Price Down): {np.sum(y_val == 0)} ({np.sum(y_val == 0)/len(y_val)*100:.2f}%)")
    print(f"Class 1 (Price Up): {np.sum(y_val == 1)} ({np.sum(y_val == 1)/len(y_val)*100:.2f}%)")
    print("\nTesting set:")
    print(f"Class 0 (Price Down): {np.sum(y_test == 0)} ({np.sum(y_test == 0)/len(y_test)*100:.2f}%)")
    print(f"Class 1 (Price Up): {np.sum(y_test == 1)} ({np.sum(y_test == 1)/len(y_test)*100:.2f}%)")
    
    # Build model
    model = build_model((sequence_length, X.shape[2]), learning_rate)
    
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
        validation_data=(X_val, y_val),
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