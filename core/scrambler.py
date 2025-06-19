import pandas as pd
import numpy as np
import sys
import random

# Try to import pyarrow, exit if not available for conversion
try:
    import pyarrow
    print("PyArrow engine available.")
except ImportError:
    print("Error: PyArrow is required for this conversion script.")
    print("Please install it using: pip install pyarrow")
    sys.exit(1)

print("Starting scrambling and conversion process...")

# --- Configuration ---
input_file = 'price_history.parquet'
output_file = 'price_history_scrambled.parquet'
seed = 42  # For reproducibility

# --- Scrambling Logic ---
try:
    print(f"Reading data from {input_file}...")
    df = pd.read_parquet(input_file, engine='pyarrow')
    
    # Validate required columns exist
    required_columns = ['watch_id', 'timestamp', 'price']
    if not all(col in df.columns for col in required_columns):
        print(f"Error: Input file must contain these columns: {required_columns}")
        sys.exit(1)
    
    print("Analyzing data...")
    unique_watch_ids = df['watch_id'].unique()
    num_watches = len(unique_watch_ids)
    print(f"Found {num_watches} unique watch IDs.")
    
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    # Create a scrambling mapping for watch IDs
    scrambled_watch_ids = unique_watch_ids.copy()
    np.random.shuffle(scrambled_watch_ids)
    watch_id_mapping = dict(zip(unique_watch_ids, scrambled_watch_ids))
    
    print("Creating scrambled dataframe...")
    scrambled_df = pd.DataFrame()
    
    # Process each watch_id separately to maintain internal time sequence
    for original_id in unique_watch_ids:
        # Extract data for this watch_id
        watch_data = df[df['watch_id'] == original_id].copy()
        
        # Sort by timestamp to ensure chronological order is preserved
        watch_data = watch_data.sort_values('timestamp')
        
        # Assign the scrambled watch_id
        watch_data['watch_id'] = watch_id_mapping[original_id]
        
        # Append to the output dataframe
        scrambled_df = pd.concat([scrambled_df, watch_data])
    
    # Sort final dataframe by scrambled watch_id and timestamp
    scrambled_df = scrambled_df.sort_values(['watch_id', 'timestamp'])
    
    print(f"Writing scrambled data to {output_file}...")
    scrambled_df.to_parquet(output_file, engine='pyarrow', index=False)
    
    # Print the mapping for reference (can be useful for testing)
    print("\nWatch ID Mapping (Original -> Scrambled):")
    for orig, scrambled in watch_id_mapping.items():
        print(f"  {orig} -> {scrambled}")
    
    print("\nScrambling successful!")
    print(f"Original shape: {df.shape}, Scrambled shape: {scrambled_df.shape}")
    
except FileNotFoundError:
    print(f"Error: Input file '{input_file}' not found.")
except Exception as e:
    print(f"An error occurred during scrambling: {e}")
    import traceback
    traceback.print_exc()

print("Scrambling script finished.")
