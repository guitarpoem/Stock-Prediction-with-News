import numpy as np
from lstm_stock_prediction import load_data, build_model
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Set random seeds for reproducibility
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)

# Parameters
sequence_length = 5
k = 3
file_path = 'combined_AAPL.csv'
verbose = 0  # Set to 1 to show training progress, 0 to hide it

# Store results
results = {}

def run_experiment(use_sentiment, learning_rate=0.0005, batch_size=64):
    print(f"\nRunning experiment with sentiment={use_sentiment}")
    
    # Load data
    X, y, _ = load_data(file_path, sequence_length, use_sentiment)
    
    # Initialize KFold with fixed random state
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    fold_results = []
    
    for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
        print(f"\nFold {fold}:")
        
        # Split into train and test
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Further split training data into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        # Build and compile model
        model = build_model((sequence_length, X.shape[2]))
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Train model with validation data
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=150,
            batch_size=batch_size,
            verbose=verbose
        )
        
        # Get final training and validation accuracies
        train_acc = history.history['accuracy'][-1]
        val_acc = history.history['val_accuracy'][-1]
        
        # Get test accuracy
        y_pred = (model.predict(X_test) > 0.5).astype("int32")
        test_acc = accuracy_score(y_test, y_pred)
        
        print(f"Training Accuracy: {train_acc:.4f}")
        print(f"Validation Accuracy: {val_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        
        fold_results.append({
            'train_acc': train_acc,
            'val_acc': val_acc,
            'test_acc': test_acc
        })
    
    # Calculate average accuracies
    avg_train_acc = np.mean([r['train_acc'] for r in fold_results])
    avg_val_acc = np.mean([r['val_acc'] for r in fold_results])
    avg_test_acc = np.mean([r['test_acc'] for r in fold_results])
    
    print(f"\nAverage Results:")
    print(f"Average Training Accuracy: {avg_train_acc:.4f}")
    print(f"Average Validation Accuracy: {avg_val_acc:.4f}")
    print(f"Average Test Accuracy: {avg_test_acc:.4f}")
    
    return {
        'fold_results': fold_results,
        'avg_train_acc': avg_train_acc,
        'avg_val_acc': avg_val_acc,
        'avg_test_acc': avg_test_acc
    }

# Run both experiments
results['With Sentiment'] = run_experiment(use_sentiment=True)
results['Without Sentiment'] = run_experiment(use_sentiment=False)

# Compare results
print("\nComparison of Results:")
print("\nWith Sentiment:")
print(f"Average Test Accuracy: {results['With Sentiment']['avg_test_acc']:.4f}")
print("\nWithout Sentiment:")
print(f"Average Test Accuracy: {results['Without Sentiment']['avg_test_acc']:.4f}")

# Calculate improvement
improvement = (results['With Sentiment']['avg_test_acc'] - 
              results['Without Sentiment']['avg_test_acc']) * 100
print(f"\nImprovement with sentiment: {improvement:.2f}%") 