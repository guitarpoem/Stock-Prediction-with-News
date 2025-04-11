import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import statistics

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
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer=Adam(learning_rate=0.0005),
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    return model

def run_experiment(use_sentiment):
    print(f"\nRunning experiment with {'sentiment' if use_sentiment else 'no sentiment'}")
    
    # Load and prepare data
    X, y, scaler = load_data('combined_AAPL.csv', sequence_length=5, use_sentiment=use_sentiment)
    
    # Split data in time order (last 20% for testing)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Build model
    model = build_model((5, X.shape[2]))
    
    # Define early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=50,
        restore_best_weights=True,
        verbose=0
    )
    
    # Train model
    model.fit(
        X_train, y_train,
        epochs=150,
        batch_size=64,
        validation_data=(X_test, y_test),
        verbose=0,
        callbacks=[early_stopping]
    )
    
    # Evaluate model
    train_score = model.evaluate(X_train, y_train, verbose=0)
    test_score = model.evaluate(X_test, y_test, verbose=0)
    
    # Calculate confusion matrix
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"Training Accuracy: {train_score[1]:.4f}")
    print(f"Testing Accuracy: {test_score[1]:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    
    return {
        'train_acc': train_score[1],
        'test_acc': test_score[1],
        'confusion_matrix': cm
    }

def main():
    # Run experiment with sentiment
    print("\nRunning experiment WITH sentiment analysis...")
    sentiment_results = run_experiment(use_sentiment=True)
    
    # Run experiment without sentiment
    print("\nRunning experiment WITHOUT sentiment analysis...")
    no_sentiment_results = run_experiment(use_sentiment=False)
    
    # Print comparison
    print("\nComparison:")
    print("\nWith Sentiment Analysis:")
    print(f"Training Accuracy: {sentiment_results['train_acc']:.4f}")
    print(f"Testing Accuracy: {sentiment_results['test_acc']:.4f}")
    print(f"Confusion Matrix:\n{sentiment_results['confusion_matrix']}")
    
    print("\nWithout Sentiment Analysis:")
    print(f"Training Accuracy: {no_sentiment_results['train_acc']:.4f}")
    print(f"Testing Accuracy: {no_sentiment_results['test_acc']:.4f}")
    print(f"Confusion Matrix:\n{no_sentiment_results['confusion_matrix']}")
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    
    # Plot accuracy comparison
    plt.subplot(2, 2, 1)
    plt.bar(['With Sentiment', 'Without Sentiment'], 
            [sentiment_results['train_acc'], no_sentiment_results['train_acc']])
    plt.title('Training Accuracy Comparison')
    plt.ylabel('Accuracy')
    
    plt.subplot(2, 2, 2)
    plt.bar(['With Sentiment', 'Without Sentiment'], 
            [sentiment_results['test_acc'], no_sentiment_results['test_acc']])
    plt.title('Testing Accuracy Comparison')
    plt.ylabel('Accuracy')
    
    # Plot confusion matrices
    plt.subplot(2, 2, 3)
    plt.imshow(sentiment_results['confusion_matrix'], cmap='Blues')
    plt.title('Confusion Matrix (With Sentiment)')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    plt.subplot(2, 2, 4)
    plt.imshow(no_sentiment_results['confusion_matrix'], cmap='Blues')
    plt.title('Confusion Matrix (Without Sentiment)')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('experiment_comparison.png')
    plt.close()

if __name__ == "__main__":
    main() 