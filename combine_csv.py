import pandas as pd

def combine_csv_files(stock_name='AAPL'):
    try:
        # Read the CSV files
        df1 = pd.read_csv('sentiment/AAPL.csv')
        df2 = pd.read_csv('price/AAPL.csv')
        
        # Convert Date columns to datetime if they're not already
        df1['Date'] = pd.to_datetime(df1['Date'])
        df2['Date'] = pd.to_datetime(df2['Date'])
        
        # Filter out entries with missing sentiment
        df1 = df1[df1['Sentiment'] != 'Missing']
        
        # Merge the dataframes on Date
        combined_df = pd.merge(df1, df2, on='Date', how='inner')
        
        # Sort by Date
        combined_df = combined_df.sort_values('Date')
        
        # Save to new CSV file
        output_file = f'combined_{stock_name}.csv'
        combined_df.to_csv(output_file, index=False)
        print(f"Successfully combined files and saved to {output_file}")
        print(f"Total entries with sentiment: {len(combined_df)}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    # combine_csv_files('AAPL') 
    # combine_csv_files('AMZN') 
    # combine_csv_files('GOOG') 
    # combine_csv_files('BAC')
    # combine_csv_files('C')
    # combine_csv_files('D')
