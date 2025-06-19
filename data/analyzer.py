import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os


def analyze_data(file_path):
    """
    Analyzes token tracking data to find correlations and visualize relationships.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        sys.exit(1)

    if len(df) < 2:
        print("Error: The dataset must contain at least two rows for analysis.")
        print(f"Please add more data to {file_path}")
        sys.exit(1)

    # Create directory for plots if it doesn't exist
    output_dir = 'data/plots'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Drop unnecessary columns
    df = df.drop(columns=['url', 'watch_id'], errors='ignore')

    # Convert time columns to datetime objects
    df['watch_start_time'] = pd.to_datetime(df['watch_start_time'])
    df['watch_end_time'] = pd.to_datetime(df['watch_end_time'])

    # Feature engineering for time of day and week
    df['hour_of_day'] = df['watch_start_time'].dt.hour
    df['day_of_week'] = df['watch_start_time'].dt.dayofweek # Monday=0, Sunday=6

    # Select columns for analysis
    cols_to_analyze = [
        'perc_change', 'high_perc', 'low_perc', 'age', 'liq', 'mcap', 'vol',
        'hour_of_day', 'day_of_week'
    ]
    
    # Filter out columns that are not present in the dataframe
    cols_to_analyze = [col for col in cols_to_analyze if col in df.columns]
    
    analysis_df = df[cols_to_analyze]

    # --- Correlation Analysis and Heatmap ---
    print("Calculating correlation matrix...")
    correlation_matrix = analysis_df.corr()

    plt.figure(figsize=(14, 12))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Token Watch Data')
    heatmap_path = os.path.join(output_dir, 'correlation_heatmap.png')
    plt.savefig(heatmap_path)
    print(f"Correlation heatmap saved to {heatmap_path}")
    plt.close()

    # --- Scatter Plots for high_perc and low_perc ---
    print("Generating scatter plots...")
    
    # Define variables to plot against high_perc and low_perc
    other_vars = ['age', 'liq', 'mcap', 'vol', 'hour_of_day', 'day_of_week']
    other_vars = [var for var in other_vars if var in analysis_df.columns]

    # Plot high_perc vs other variables
    if 'high_perc' in analysis_df.columns:
        for var in other_vars:
            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=analysis_df, x=var, y='high_perc')
            plt.title(f'high_perc vs {var}')
            plt.grid(True)
            plot_filename = os.path.join(output_dir, f'high_perc_vs_{var}.png')
            plt.savefig(plot_filename)
            print(f"Plot saved to {plot_filename}")
            plt.close()

    # Plot low_perc vs other variables
    if 'low_perc' in analysis_df.columns:
        for var in other_vars:
            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=analysis_df, x=var, y='low_perc')
            plt.title(f'low_perc vs {var}')
            plt.grid(True)
            plot_filename = os.path.join(output_dir, f'low_perc_vs_{var}.png')
            plt.savefig(plot_filename)
            print(f"Plot saved to {plot_filename}")
            plt.close()

    print(f"\nAnalysis complete. Plots are saved in the '{output_dir}' directory.")


def analyze_performance(file_path):
    """
    Performs a detailed analysis on percentage change metrics.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        sys.exit(1)

    if 'perc_change' not in df.columns:
        print("Error: 'perc_change' column not found in the data.")
        return

    # --- Analysis of perc_change ---
    positive_perc_change = df[df['perc_change'] > 0]
    positive_perc_change_percentage = (len(positive_perc_change) / len(df)) * 100
    
    print("\n--- Performance Analysis ---")
    print(f"Percentage of trades with positive perc_change: {positive_perc_change_percentage:.2f}%")
    
    # --- Descriptive Statistics ---
    print("\nDescriptive Statistics for Percentage Changes:")
    stats_cols = ['perc_change', 'high_perc', 'low_perc']
    stats_cols = [col for col in stats_cols if col in df.columns]
    
    if stats_cols:
        print(df[stats_cols].describe())

    # --- Further Analysis ---
    if 'perc_change' in df.columns:
        avg_positive_change = df[df['perc_change'] > 0]['perc_change'].mean()
        avg_negative_change = df[df['perc_change'] < 0]['perc_change'].mean()
        print(f"\nAverage perc_change on positive trades: {avg_positive_change:.2f}%")
        print(f"Average perc_change on negative trades: {avg_negative_change:.2f}%")

    if 'high_perc' in df.columns:
        prob_high_perc_gt_20 = (len(df[df['high_perc'] > 20]) / len(df)) * 100
        print(f"Probability of high_perc > 20%: {prob_high_perc_gt_20:.2f}%")

    if 'low_perc' in df.columns:
        prob_low_perc_lt_minus_20 = (len(df[df['low_perc'] < -20]) / len(df)) * 100
        print(f"Probability of low_perc < -20%: {prob_low_perc_lt_minus_20:.2f}%")
    print("\n--- End of Performance Analysis ---")


if __name__ == '__main__':
    file_path = 'data/watch_tracking.csv'
    analyze_data(file_path)
    analyze_performance(file_path)
