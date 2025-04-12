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
from imblearn.over_sampling import SMOTE
from collections import Counter

def load_data(file_path, sequence_length=5, use_sentiment=True):
    # Load the data
    df = pd.read_csv(file_path)
    
    # Convert sentiment to numerical values
    sentiment_map = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
    df['Sentiment'] = df['Sentiment'].map(sentiment_map)
    
    # Select features
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    if use_sentiment:
        features.insert(0, 'Sentiment')
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
    epochs = 5
    batch_size = 64
    learning_rate = 0.0001
    # use_sentiment = True
    use_sentiment = False
    verbose = 1  # Set to 1 to show training progress, 0 to hide it
    early_stopping_patience = 5  # Number of epochs to wait before early stopping
    
    stock_name = 'AMZN'

    # Load and prepare data
    X, y, scaler = load_data(f'combined_{stock_name}.csv', sequence_length, use_sentiment)
    
    # Split data chronologically (last 20% for testing)
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Print class distribution before SMOTE
    print("Class distribution before SMOTE:")
    print("Training set:", Counter(y_train))
    print("Test set:", Counter(y_test))
    
    # Reshape training data for SMOTE
    X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
    
    # Apply SMOTE only to training data
    smote = SMOTE(random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_reshaped, y_train)
    
    # Reshape back to original shape
    X_train_resampled = X_train_resampled.reshape(-1, sequence_length, X.shape[2])
    
    # Print class distribution after SMOTE
    print("\nClass distribution after SMOTE:")
    print("Training set:", Counter(y_train_resampled))
    print("Test set:", Counter(y_test))
    
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
        X_train_resampled, y_train_resampled,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        verbose=verbose,
        callbacks=[early_stopping]
    )
    
    # Evaluate model
    train_score = model.evaluate(X_train_resampled, y_train_resampled, verbose=0)
    test_score = model.evaluate(X_test, y_test, verbose=0)
    
    # Predict test set
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    print(f"\nTraining Accuracy: {train_score[1]:.4f}")
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