import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import statistics
import backtrader as bt
from datetime import datetime
import os

class SentimentStrategy(bt.Strategy):
    params = (
        ('buy_threshold', 0.5),
        ('sell_threshold', 0.5),
    )

    def __init__(self):
        self.predictions = self.datas[0].predictions
        self.data = self.datas[0]
        self.order = None

    def next(self):
        if self.order:
            return

        if not self.position:
            if self.predictions[0] > self.params.buy_threshold:
                self.order = self.buy()
        else:
            if self.predictions[0] <= self.params.sell_threshold:
                self.order = self.sell()

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
            else:
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')

        self.order = None

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} {txt}')

class PandasData(bt.feeds.PandasData):
    lines = ('predictions',)
    params = (
        ('datetime', None),
        ('open', -1),
        ('high', -1),
        ('low', -1),
        ('close', -1),
        ('volume', -1),
        ('openinterest', -1),
        ('predictions', -1),
    )

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
    
    return np.array(X), np.array(y), scaler, df

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

def simulate_trading(predictions, actual_prices, initial_capital=1000):
    # Create a cerebro entity
    cerebro = bt.Cerebro()
    
    # Add a strategy
    cerebro.addstrategy(SentimentStrategy)
    
    # Create a Data Feed
    df = pd.DataFrame({
        'datetime': pd.date_range(start='1/1/2020', periods=len(actual_prices)),
        'open': actual_prices,
        'high': actual_prices,
        'low': actual_prices,
        'close': actual_prices,
        'volume': np.ones(len(actual_prices)) * 1000,  # dummy volume
        'predictions': predictions
    })
    
    # Convert datetime to datetime64[ns]
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Set datetime as index
    df.set_index('datetime', inplace=True)
    
    data = PandasData(dataname=df)
    cerebro.adddata(data)
    
    # Set our desired cash start
    cerebro.broker.setcash(initial_capital)
    
    # Set the commission
    cerebro.broker.setcommission(commission=0.001)  # 0.1% commission
    
    # Print out the starting conditions
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    
    # Run over everything
    cerebro.run()
    
    # Print out the final result
    final_value = cerebro.broker.getvalue()
    print('Final Portfolio Value: %.2f' % final_value)
    
    return final_value

def run_experiment(use_sentiment, stock_name):
    print(f"\nRunning experiment with {'sentiment' if use_sentiment else 'no sentiment'}")
    
    # Load and prepare data
    X, y, scaler, df = load_data(f'combined_{stock_name}.csv', sequence_length=5, use_sentiment=use_sentiment)
    
    # Split data in time order (last 20% for testing)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Create model directory if it doesn't exist
    model_dir = f'./models/{stock_name}'
    os.makedirs(model_dir, exist_ok=True)
    
    # Define model path
    model_path = f'{model_dir}/model_{"with_sentiment" if use_sentiment else "without_sentiment"}.h5'
    
    # Check if model exists
    if os.path.exists(model_path):
        print(f"Loading existing model for {stock_name} {'with' if use_sentiment else 'without'} sentiment")
        model = load_model(model_path)
    else:
        print(f"Building new model for {stock_name} {'with' if use_sentiment else 'without'} sentiment")
        model = build_model((5, X.shape[2]))
        
        # Define callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=50,
            restore_best_weights=True,
            verbose=0
        )
        
        model_checkpoint = ModelCheckpoint(
            model_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=0
        )
        
        # Train model
        model.fit(
            X_train, y_train,
            epochs=150,
            batch_size=64,
            validation_data=(X_test, y_test),
            verbose=0,
            callbacks=[early_stopping, model_checkpoint]
        )
    
    # Evaluate model
    train_score = model.evaluate(X_train, y_train, verbose=0)
    test_score = model.evaluate(X_test, y_test, verbose=0)
    
    # Get predictions for trading simulation
    test_predictions = model.predict(X_test)
    test_predictions = test_predictions.flatten()  # Convert to 1D array
    
    # Get corresponding prices
    test_prices = df['Close'].values[split_idx + 5:]  # +5 because of sequence length
    
    # Simulate trading
    final_capital = simulate_trading(test_predictions, test_prices)
    
    # Calculate buy and hold returns
    initial_price = test_prices[0]
    final_price = test_prices[-1]
    buy_hold_return = (1000 * final_price) / initial_price
    
    return {
        'train_acc': train_score[1],
        'test_acc': test_score[1],
        'final_capital': final_capital,
        'buy_hold_return': buy_hold_return
    }

def main():
    stocks = ['AAPL', 'AMZN', 'GOOG', 'BAC', 'C', 'D']
    
    for stock_name in stocks:
        print(f"\nRunning experiments for {stock_name}...")
        
        # Run experiments with sentiment
        sentiment_results = run_experiment(use_sentiment=True, stock_name=stock_name)
        
        # Run experiments without sentiment
        no_sentiment_results = run_experiment(use_sentiment=False, stock_name=stock_name)
        
        # Save results to file
        with open(f'./results_2/{stock_name}_results.txt', 'w') as f:
            f.write(f"Results for {stock_name}\n")
            f.write("\nWith Sentiment Analysis:\n")
            f.write(f"Training Accuracy: {sentiment_results['train_acc']:.4f}\n")
            f.write(f"Testing Accuracy: {sentiment_results['test_acc']:.4f}\n")
            f.write(f"Final Capital: ${sentiment_results['final_capital']:.2f}\n")
            
            f.write("\nWithout Sentiment Analysis:\n")
            f.write(f"Training Accuracy: {no_sentiment_results['train_acc']:.4f}\n")
            f.write(f"Testing Accuracy: {no_sentiment_results['test_acc']:.4f}\n")
            f.write(f"Final Capital: ${no_sentiment_results['final_capital']:.2f}\n")
            
            f.write("\nBuy and Hold Strategy:\n")
            f.write(f"Final Value: ${sentiment_results['buy_hold_return']:.2f}\n")
        
        print(f"Results saved for {stock_name} in the results_2 directory")

if __name__ == "__main__":
    main() 