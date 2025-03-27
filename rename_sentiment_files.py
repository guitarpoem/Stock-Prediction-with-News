import os
import re

def rename_sentiment_files():
    # Get all files in the sentiment directory
    sentiment_dir = './sentiment'
    files = os.listdir(sentiment_dir)
    
    # Regular expression to match the pattern and extract the stock symbol
    pattern = r'processed_sentiment_analysis_([A-Z]+)_\d+_\d+\.csv'
    
    for filename in files:
        match = re.match(pattern, filename)
        if match:
            # Extract the stock symbol
            stock_symbol = match.group(1)
            # Create new filename
            new_filename = f"{stock_symbol}.csv"
            
            # Create full paths for both old and new filenames
            old_path = os.path.join(sentiment_dir, filename)
            new_path = os.path.join(sentiment_dir, new_filename)
            
            # Rename the file
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} → {new_filename}")

if __name__ == "__main__":
    rename_sentiment_files() 