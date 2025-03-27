import os
import pandas as pd
import shutil

def filter_sentiment_files():
    # Directory containing the processed CSV files
    processed_dir = "./sentiment/processed"
    output_dir = "./sentiment/processed_2"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all CSV files from the processed directory
    csv_files = [f for f in os.listdir(processed_dir) if f.endswith('.csv')]
    
    # List to store files with less than 20% missing rate
    low_missing_files = []
    
    # Process each file
    for file in csv_files:
        file_path = os.path.join(processed_dir, file)
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Count total rows and missing values in this file
        file_total = len(df)
        file_missing = len(df[df['Sentiment'] == 'Missing'])
        
        # Calculate percentage for this file
        if file_total > 0:
            missing_percentage = (file_missing / file_total) * 100
            
            # Check if missing rate is less than 20%
            if missing_percentage < 20:
                low_missing_files.append(file)
                
                # Copy the file to the new directory
                output_path = os.path.join(output_dir, file)
                shutil.copy2(file_path, output_path)
    
    # Print results
    print(f"Total files processed: {len(csv_files)}")
    print(f"Files with less than 20% missing rate: {len(low_missing_files)}")
    print(f"Files copied to {output_dir}:")
    for file in sorted(low_missing_files):
        print(f"  - {file}")

if __name__ == "__main__":
    filter_sentiment_files() 