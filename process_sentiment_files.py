import os
import csv
import pandas as pd

def process_sentiment_files():
    # Create the output directory if it doesn't exist
    output_dir = "./sentiment/processed"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all CSV files from the raw directory
    raw_dir = "./sentiment/raw"
    csv_files = [f for f in os.listdir(raw_dir) if f.endswith('.csv')]
    
    for file in csv_files:
        input_path = os.path.join(raw_dir, file)
        output_path = os.path.join(output_dir, file)
        
        # Read the CSV file
        df = pd.read_csv(input_path)
        
        # Keep only the Date and Sentiment columns
        if 'Analysis' in df.columns:
            df = df[['Date', 'Sentiment']]
        
        # Save the processed file
        df.to_csv(output_path, index=False)
        print(f"Processed {file}")

if __name__ == "__main__":
    process_sentiment_files() 