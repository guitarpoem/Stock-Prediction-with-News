import numpy as np
from lstm_stock_prediction import load_data, build_model
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from keras.optimizers import Adam

# Parameters
sequence_length = 5
k = 5
file_path = 'combined_AAPL.csv'
verbose = 0  # Set to 1 to show training progress, 0 to hide it

# Store results
results = {}

# Case 1: With sentiment feature
use_sentiment = True
learning_rate = 0.0005
batch_size = 64

print("Case 1: With Sentiment Feature")

kf = KFold(n_splits=k, shuffle=True, random_state=42)
accuracies = []

X, y, _ = load_data(file_path, sequence_length, use_sentiment)

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    model = build_model((sequence_length, X.shape[2]))
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    model.fit(X_train, y_train, epochs=150, batch_size=batch_size, verbose=verbose)
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

print(f"Average Accuracy: {np.mean(accuracies):.4f}")
results['With Sentiment'] = np.mean(accuracies)

# Case 2: Without sentiment feature
use_sentiment = False
learning_rate = 0.001
batch_size = 32

print("\nCase 2: Without Sentiment Feature")

kf = KFold(n_splits=k, shuffle=True, random_state=42)
accuracies = []

X, y, _ = load_data(file_path, sequence_length, use_sentiment)

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    model = build_model((sequence_length, X.shape[2]))
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    model.fit(X_train, y_train, epochs=150, batch_size=batch_size, verbose=verbose)
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

print(f"Average Accuracy: {np.mean(accuracies):.4f}")
results['Without Sentiment'] = np.mean(accuracies)

# Display all results
print("\nOverall Results:")
for case, accuracy in results.items():
    print(f"{case}: Average Accuracy = {accuracy:.4f}") 