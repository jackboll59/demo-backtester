import pandas as pd
import sys
import pyarrow

print("Starting conversion from CSV to Parquet...")

# --- Configuration ---
csv_file = 'data/price_history.csv'
parquet_file = 'price_history.parquet'
dtypes = {'watch_id': 'string', 'price': 'float64'} # Specify known dtypes
timestamp_format = '%Y-%m-%d %H:%M:%S'
columns_to_load = ['watch_id', 'timestamp', 'price'] # Load only necessary columns

# --- Conversion Logic ---
try:
    print(f"Reading entire {csv_file} using pyarrow engine...")
    full_df = pd.read_csv(
        csv_file,
        engine='pyarrow',
        dtype=dtypes,
        usecols=columns_to_load
    )

    print("Processing loaded data...")
    # Ensure price is numeric (redundant with dtype, but safe)
    full_df['price'] = pd.to_numeric(full_df['price'], errors='coerce')
    # Convert timestamp
    full_df['timestamp'] = pd.to_datetime(
        full_df['timestamp'],
        format=timestamp_format,
        errors='coerce'
    )
    # Drop rows with invalid price or timestamp
    full_df.dropna(subset=['price', 'timestamp'], inplace=True)

    if full_df.empty:
        print("No valid data found in CSV file. Parquet file not created.")
    else:
        print("Sorting data by watch_id and timestamp...")
        full_df.sort_values(by=['watch_id', 'timestamp'], inplace=True)
        
        print(f"Writing data to {parquet_file}...")
        full_df.to_parquet(parquet_file, engine='pyarrow', index=False)
        print("Conversion successful!")

except FileNotFoundError:
    print(f"Error: Input file '{csv_file}' not found.")
except Exception as e:
    print(f"An error occurred during conversion: {e}")
    import traceback
    traceback.print_exc()

print("Conversion script finished.")
