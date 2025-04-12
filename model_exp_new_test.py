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

def run_experiment(use_sentiment, stock_name):
    print(f"\nRunning experiment with {'sentiment' if use_sentiment else 'no sentiment'}")
    
    # Load and prepare data
    X, y, scaler = load_data(f'combined_{stock_name}.csv', sequence_length=5, use_sentiment=use_sentiment)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    
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
    
    # Get predictions and confusion matrix
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"Training Accuracy: {train_score[1]:.4f}, Testing Accuracy: {test_score[1]:.4f}")
    print("Confusion Matrix:")
    print(f"True Negative: {cm[0,0]}, False Positive: {cm[0,1]}")
    print(f"False Negative: {cm[1,0]}, True Positive: {cm[1,1]}")
    
    return {
        'train_acc': train_score[1],
        'test_acc': test_score[1],
        'confusion_matrix': cm
    }

def main():
    stocks = ['AMZN', 'GOOG']
    # stocks = ['AMZN', 'GOOG', 'AAPL', 'BAC', 'C', 'D']
    
    for stock_name in stocks:
        print(f"\nRunning experiments for {stock_name}...")
        
        # Run experiments with sentiment
        print(f"Running experiment WITH sentiment analysis for {stock_name}...")
        sentiment_results = run_experiment(use_sentiment=True, stock_name=stock_name)
        
        # Run experiments without sentiment
        print(f"Running experiment WITHOUT sentiment analysis for {stock_name}...")
        no_sentiment_results = run_experiment(use_sentiment=False, stock_name=stock_name)
        
        # Save results to file
        with open(f'results/{stock_name}_results.txt', 'w') as f:
            f.write(f"Results for {stock_name}\n")
            f.write("\nWith Sentiment Analysis:\n")
            f.write(f"Training Accuracy: {sentiment_results['train_acc']:.4f}\n")
            f.write(f"Testing Accuracy: {sentiment_results['test_acc']:.4f}\n")
            
            f.write("\nWithout Sentiment Analysis:\n")
            f.write(f"Training Accuracy: {no_sentiment_results['train_acc']:.4f}\n")
            f.write(f"Testing Accuracy: {no_sentiment_results['test_acc']:.4f}\n")
        
        # Plot comparison
        plt.figure(figsize=(10, 6))
        
        # Plot training accuracy comparison
        plt.subplot(2, 1, 1)
        plt.bar(['With Sentiment', 'Without Sentiment'], 
                [sentiment_results['train_acc'], no_sentiment_results['train_acc']])
        plt.title(f'{stock_name} - Training Accuracy Comparison')
        plt.ylabel('Accuracy')
        
        # Plot testing accuracy comparison
        plt.subplot(2, 1, 2)
        plt.bar(['With Sentiment', 'Without Sentiment'], 
                [sentiment_results['test_acc'], no_sentiment_results['test_acc']])
        plt.title(f'{stock_name} - Testing Accuracy Comparison')
        plt.ylabel('Accuracy')
        
        plt.tight_layout()
        plt.savefig(f'results/{stock_name}_comparison.png')
        plt.close()
        
        print(f"Results and plots saved for {stock_name} in the results directory")

if __name__ == "__main__":
    main() 