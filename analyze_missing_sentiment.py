import os
import pandas as pd
import matplotlib.pyplot as plt

def analyze_missing_sentiment():
    # Directory containing the processed CSV files
    processed_dir = "./sentiment/processed"
    
    # Get all CSV files from the processed directory
    csv_files = [f for f in os.listdir(processed_dir) if f.endswith('.csv')]
    
    # Initialize counters
    total_rows = 0
    missing_count = 0
    
    # Dictionary to store missing percentage for each file
    file_missing_percentages = {}
    file_missing_counts = {}
    file_total_counts = {}
    
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
            file_missing_percentages[file] = missing_percentage
            file_missing_counts[file] = file_missing
            file_total_counts[file] = file_total
        
        # Update overall counters
        total_rows += file_total
        missing_count += file_missing
    
    # Calculate overall percentage
    overall_missing_percentage = (missing_count / total_rows) * 100
    
    print(f"Total rows across all files: {total_rows}")
    print(f"Total 'Missing' sentiment values: {missing_count}")
    print(f"Overall percentage of 'Missing' values: {overall_missing_percentage:.2f}%")
    
    # Print individual file percentages
    print("\nMissing values by file (sorted by percentage):")
    print("File, Missing Count, Total Count, Missing Percentage")
    for file, percentage in sorted(file_missing_percentages.items(), key=lambda x: x[1], reverse=True):
        print(f"{file}, {file_missing_counts[file]}, {file_total_counts[file]}, {percentage:.2f}%")
    
    # Create a bar chart of the top 10 files with highest missing percentages
    top_files = dict(sorted(file_missing_percentages.items(), key=lambda x: x[1], reverse=True)[:10])
    
    plt.figure(figsize=(12, 6))
    plt.bar(top_files.keys(), top_files.values())
    plt.xticks(rotation=45, ha='right')
    plt.title('Top 10 Files with Highest Percentage of "Missing" Values')
    plt.ylabel('Percentage (%)')
    plt.tight_layout()
    
    # Save the chart
    output_dir = "./sentiment/analysis"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'missing_sentiment_analysis.png'))
    
    print(f"\nAnalysis chart saved to {os.path.join(output_dir, 'missing_sentiment_analysis.png')}")

if __name__ == "__main__":
    analyze_missing_sentiment() 