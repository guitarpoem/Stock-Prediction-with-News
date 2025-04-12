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
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

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

class LSTMTradingStrategy(Strategy):
    def init(self):
        self.model = None
        self.scaler = None
        self.sequence_length = 5
        self.use_sentiment = True
        
    def next(self):
        if len(self.data) < self.sequence_length:
            return
            
        # Prepare the sequence for prediction
        sequence = []
        for i in range(self.sequence_length):
            idx = len(self.data) - self.sequence_length + i
            features = [
                self.data.Open[idx],
                self.data.High[idx],
                self.data.Low[idx],
                self.data.Close[idx],
                self.data.Volume[idx]
            ]
            if self.use_sentiment:
                features.insert(0, self.data.Sentiment[idx])
            sequence.append(features)
        
        # Normalize the sequence
        sequence = self.scaler.transform(sequence)
        sequence = np.array([sequence])
        
        # Get prediction
        prediction = self.model.predict(sequence, verbose=0)[0][0]
        
        # Trading logic
        if prediction > 0.7 and not self.position:
            self.buy()
        elif prediction < 0.3 and self.position:
            self.position.close()

def run_backtest(model, scaler, test_data, use_sentiment=True):
    # Prepare the data for backtesting
    df = test_data.copy()
    df['Sentiment'] = df['Sentiment'].map({'Positive': 1, 'Neutral': 0, 'Negative': -1})
    
    # Convert index to datetime
    df.index = pd.to_datetime(df.index)
    
    # Create strategy class with model and scaler
    class CustomLSTMTradingStrategy(LSTMTradingStrategy):
        def init(self):
            super().init()
            self.model = model
            self.scaler = scaler
            self.use_sentiment = use_sentiment
    
    # Create and run backtest
    bt = Backtest(
        df,
        CustomLSTMTradingStrategy,
        cash=1000,
        commission=.002,
        exclusive_orders=True
    )
    
    # Run backtest
    stats = bt.run()
    return stats

def run_experiment(use_sentiment):
    print(f"\nRunning experiment with {'sentiment' if use_sentiment else 'no sentiment'}")
    
    # Load and prepare data
    X, y, scaler = load_data('combined_AAPL.csv', sequence_length=5, use_sentiment=use_sentiment)
    
    # Split data in time order (last 20% for testing)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Load the test data for backtesting
    test_data = pd.read_csv('combined_AAPL.csv').iloc[split_idx:]
    
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
    
    # Run backtest
    backtest_stats = run_backtest(model, scaler, test_data, use_sentiment)
    
    print(f"Training Accuracy: {train_score[1]:.4f}")
    print(f"Testing Accuracy: {test_score[1]:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print("\nBacktest Results:")
    print(f"Return: {backtest_stats['Return [%]']:.2f}%")
    print(f"Buy & Hold Return: {backtest_stats['Buy & Hold Return [%]']:.2f}%")
    print(f"Max. Drawdown: {backtest_stats['Max. Drawdown [%]']:.2f}%")
    print(f"# Trades: {backtest_stats['# Trades']}")
    
    return {
        'train_acc': train_score[1],
        'test_acc': test_score[1],
        'confusion_matrix': cm,
        'backtest_stats': backtest_stats
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