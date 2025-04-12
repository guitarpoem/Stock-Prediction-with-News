import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

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
    
    # Create sequences and flatten for XGBoost
    X, y = [], []
    for i in range(len(data) - sequence_length):
        # Flatten the sequence into a single row
        X.append(data[i:(i + sequence_length)].flatten())
        # Create binary target: 1 if next day's close is higher than current day's close
        y.append(1 if data[i + sequence_length][4] > data[i + sequence_length - 1][4] else 0)
    
    return np.array(X), np.array(y), scaler

def build_model():
    # XGBoost parameters with regularization and improved tree parameters
    params = {
        'objective': 'binary:logistic',
        'max_depth': 4,  # Reduced from 6 to prevent overfitting
        'learning_rate': 0.05,  # Reduced learning rate for better generalization
        'n_estimators': 200,  # Increased number of trees
        'subsample': 0.8,  # Randomly sample 80% of data for each tree
        'colsample_bytree': 0.8,  # Randomly sample 80% of features for each tree
        'min_child_weight': 3,  # Minimum sum of instance weight needed in a child
        'gamma': 0.1,  # Minimum loss reduction required to make a split
        'reg_alpha': 0.1,  # L1 regularization term
        'reg_lambda': 1.0,  # L2 regularization term
        'random_state': 42,
        'early_stopping_rounds': 10  # Stop if no improvement for 10 rounds
    }
    
    model = xgb.XGBClassifier(**params)
    return model

def main():
    # Parameters
    sequence_length = 5
    test_size = 0.2
    random_state = 42
    # use_sentiment = False
    use_sentiment = True
    verbose = 1
    
    stock_name = 'GOOG'

    # Load and prepare data
    X, y, scaler = load_data(f'combined_{stock_name}.csv', sequence_length, use_sentiment)
    
    # Split data chronologically (last 20% for testing)
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Build model
    model = build_model()
    
    # Train model
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=verbose
    )
    
    # Evaluate model
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    print("Confusion Matrix:")
    print(cm)
    
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Testing Accuracy: {test_accuracy:.4f}")
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    xgb.plot_importance(model)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig(f'feature_importance_{stock_name}.png')
    plt.close()

if __name__ == "__main__":
    main() 