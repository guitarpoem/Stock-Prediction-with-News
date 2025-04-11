# LSTM Model for Stock Movement Prediction

## Overview
This project creates an LSTM model to predict stock price movements (up or down) based on historical data. The model uses the past N days of data to predict the movement on day N+1, where N is a configurable hyperparameter (default: 5).

## Data
The model uses `combined_AAPL.csv` in the working directory, which contains stock price data and sentiment analysis.

## Implementation Details

This project implements an LSTM (Long Short-Term Memory) neural network model to predict stock price movements. The implementation:

- Loads and preprocesses historical stock data
- Creates sequential data points for LSTM training
- Builds and trains a deep learning model
- Evaluates prediction accuracy
- Visualizes training results

### Data Processing
- Loads the CSV file
- Converts sentiment values to numerical (-1, 0, 1)
- Normalizes all features using MinMaxScaler
- Creates sequences of length N for LSTM input

### Model Architecture
- Two LSTM layers with 50 units each
- Dropout layers (0.2) to prevent overfitting
- Dense layers for final prediction
- Binary classification (1 for price increase, 0 for decrease)

### Features Used
- Sentiment (converted to numerical)
- Open price
- High price
- Low price
- Close price
- Volume

### Target
Binary: 1 if the next day's close price is higher than the current day's close price, 0 otherwise.

## Training and Evaluation
- Splits data into train/test sets (80/20)
- Trains for 50 epochs with batch size 32
- Uses Adam optimizer with learning rate 0.001
- Binary cross-entropy loss function
- Saves training history plots

## Configurable Hyperparameters
- `sequence_length`: Number of past days to use for prediction (default: 5)
- `test_size`: Proportion of data to use for testing (default: 0.2)
- `epochs`: Number of training epochs (default: 50)
- `batch_size`: Batch size for training (default: 32)

## Usage
1. Install requirements:
   ```
   pip install -r requirements.txt
   ```

2. Run the script:
   ```
   python lstm_stock_prediction.py
   ```

3. Results:
   - Training and testing accuracy will be printed
   - Training history plot will be saved as 'training_history.png'