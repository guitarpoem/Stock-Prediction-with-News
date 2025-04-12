import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def load_data(file_path, use_sentiment=True):
    # Load the data
    df = pd.read_csv(file_path)
    
    # Convert sentiment to numerical values
    sentiment_map = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
    df['Sentiment'] = df['Sentiment'].map(sentiment_map)
    
    # Select features
    features = ['Close']  # ARIMA works with univariate time series
    if use_sentiment:
        features.insert(0, 'Sentiment')
    
    # Normalize the data
    scaler = MinMaxScaler()
    data = scaler.fit_transform(df[features])
    
    # Create target: 1 if next day's close is higher than current day's close
    y = []
    for i in range(len(data) - 1):
        y.append(1 if data[i + 1][-1] > data[i][-1] else 0)
    
    return data, np.array(y), scaler

def build_model(data, order=(5,1,0)):
    """
    Build and fit ARIMA model
    order: (p,d,q) where p is the number of lag observations,
           d is the degree of differencing, and q is the size of the moving average window
    """
    model = ARIMA(data, order=order)
    model_fit = model.fit()
    return model_fit

def main():
    # Parameters
    test_size = 0.2
    random_state = 42
    use_sentiment = True
    
    # Load and prepare data
    data, y, scaler = load_data('combined_AAPL.csv', use_sentiment)
    
    # Split data
    train_size = int(len(data) * (1 - test_size))
    train_data = data[:train_size]
    test_data = data[train_size:]
    y_train = y[:train_size]
    y_test = y[train_size:]
    
    # Build and fit model
    model = build_model(train_data[:, -1])  # Use only Close price for ARIMA
    
    # Make predictions
    predictions = model.forecast(steps=len(test_data))
    
    # Convert predictions to binary (1 if predicted price is higher than previous price)
    y_pred = []
    for i in range(1, len(predictions)):
        y_pred.append(1 if predictions[i] > predictions[i-1] else 0)
    
    # Confusion matrix
    cm = confusion_matrix(y_test[1:], y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    # Calculate accuracy
    accuracy = np.mean(y_test[1:] == y_pred)
    print(f"Testing Accuracy: {accuracy:.4f}")
    
    # Plot actual vs predicted
    plt.figure(figsize=(12, 6))
    plt.plot(y_test[1:], label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.title('ARIMA Model Predictions')
    plt.xlabel('Time')
    plt.ylabel('Price Movement (1=Up, 0=Down)')
    plt.legend()
    plt.savefig('arima_predictions.png')
    plt.close()

if __name__ == "__main__":
    main() 