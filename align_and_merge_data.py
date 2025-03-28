# This script will:
# Read each sentiment CSV file from sentiment/processed_2
# Find the corresponding price data file in price/preprocessed
# Parse the price data from the text file, extracting the columns
# Convert date formats to ensure they match
# Merge the sentiment and price data on the date column
# Save the combined data to a CSV file in the dataset folder with the same ticker name
# The resulting CSV files will have columns for date, movement percent, open price, high price, low price, close price, volume, and sentiment.

import os
import pandas as pd
import re

def align_and_merge_data():
    # Create dataset directory if it doesn't exist
    os.makedirs("./dataset", exist_ok=True)
    
    # Get all files from sentiment/processed_2
    sentiment_dir = "./sentiment/processed_2"
    sentiment_files = [f for f in os.listdir(sentiment_dir) if f.endswith('.csv')]
    
    # Get all files from price/preprocessed
    price_dir = "./price/preprocessed"
    price_files = [f for f in os.listdir(price_dir) if f.endswith('.txt')]
    
    # Process each sentiment file
    for sentiment_file in sentiment_files:
        # Extract ticker symbol from filename (remove .csv extension)
        ticker = sentiment_file.split('.')[0]
        
        # Check if corresponding price file exists
        price_file = f"{ticker}.txt"
        if price_file in price_files:
            # Read sentiment data
            sentiment_path = os.path.join(sentiment_dir, sentiment_file)
            sentiment_df = pd.read_csv(sentiment_path)
            
            # Read price data
            price_path = os.path.join(price_dir, price_file)
            with open(price_path, 'r') as f:
                price_content = f.read()
            
            # Extract price data lines (skip header and footer lines with "...")
            price_lines = [line for line in price_content.split('\n') if line and not line.startswith('...')]
            
            # Parse price data
            price_data = []
            for line in price_lines:
                parts = re.split(r'\s+', line.strip())
                if len(parts) >= 7:  # Ensure we have all expected columns
                    date = parts[0]
                    movement_percent = parts[1]
                    open_price = parts[2]
                    high_price = parts[3]
                    low_price = parts[4]
                    close_price = parts[5]
                    volume = parts[6]
                    price_data.append([date, movement_percent, open_price, high_price, 
                                      low_price, close_price, volume])
            
            # Create price dataframe
            price_df = pd.DataFrame(price_data, columns=['date', 'movement_percent', 'open_price', 
                                                        'high_price', 'low_price', 'close_price', 'volume'])
            
            # Convert date formats to be consistent
            sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])
            price_df['date'] = pd.to_datetime(price_df['date'])
            
            # Merge dataframes on date
            merged_df = pd.merge(price_df, sentiment_df, left_on='date', right_on='Date', how='inner')
            
            # Drop duplicate date column
            merged_df = merged_df.drop('Date', axis=1)
            
            # Save to dataset folder
            output_path = os.path.join("./dataset", f"{ticker}.csv")
            merged_df.to_csv(output_path, index=False)
            
            print(f"Merged data for {ticker} saved to {output_path}")
        else:
            print(f"No price data found for {ticker}")
    
    print(f"Total files processed: {len(sentiment_files)}")

if __name__ == "__main__":
    align_and_merge_data() 