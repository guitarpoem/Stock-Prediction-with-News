import pandas as pd
import os

# Create directories if they don't exist
os.makedirs('price/processed', exist_ok=True)

# Read the CSV file
df = pd.read_csv('price/raw/AAPL.csv')

# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Sort by date to ensure chronological order
df = df.sort_values('Date')

# Calculate price change
df['Previous_Close'] = df['Close'].shift(1)
df['Price_Change'] = df['Close'] - df['Previous_Close']
df['Direction'] = df['Price_Change'].apply(lambda x: 'Up' if x > 0 else ('Down' if x < 0 else 'Unchanged'))

# Create a new dataframe with the desired columns
result_df = df[['Date', 'Close', 'Previous_Close', 'Price_Change', 'Direction']].copy()

# Drop the first row as it doesn't have a previous close value
result_df = result_df.dropna()

# Write to a new CSV file
output_path = 'price/processed/AAPL_price_direction.csv'
result_df.to_csv(output_path, index=False)

print(f"Processing complete. Results saved to {output_path}")

# Display a sample of the results
print("\nSample of processed data:")
print(result_df.head()) 