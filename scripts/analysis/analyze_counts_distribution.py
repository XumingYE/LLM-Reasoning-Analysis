import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

def analyze_distribution(data_path, output_dir):
    """
    Loads data, calculates descriptive statistics, and generates plots for
    counts, query length, and content length.
    """
    print(f"Loading data from {data_path} to analyze 'counts' distribution...")
    df = pd.read_json(data_path, lines=True)

    # --- Calculate Statistics ---
    print("\n--- Descriptive Statistics for 'counts' ---")
    stats = df['counts'].describe()
    print(stats)

    print("\n--- High-End Quantile Information for 'counts' ---")
    quantiles = [0.8, 0.9, 0.95, 0.98, 0.99, 1.0]
    high_quantiles = df['counts'].quantile(quantiles)
    print(high_quantiles)

    # --- Prepare for Plotting ---
    print("\n--- Generating Plots ---")
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate lengths for plotting
    df['query_length'] = (df['instruction'].fillna('').str.len() + df['input'].fillna('').str.len())
    df['content_length'] = df['content'].fillna('').str.len()

    plt.style.use('seaborn-v0_8-whitegrid')

    # --- Plot 1: Counts Distribution ---
    plt.figure(figsize=(12, 7))
    sns.histplot(df['counts'], bins=50, kde=True)
    plt.title('Distribution of "counts" (Wait-Only)', fontsize=16)
    plt.xlabel('Counts', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.yscale('log') # Use log scale for better visibility of tail
    counts_dist_path = os.path.join(output_dir, 'counts_distribution.png')
    plt.savefig(counts_dist_path, dpi=300)
    print(f"Saved counts distribution plot to {counts_dist_path}")
    plt.close()

    # --- Plot 2: Query Length vs. Counts ---
    plt.figure(figsize=(12, 7))
    sns.scatterplot(data=df, x='query_length', y='counts', alpha=0.5)
    plt.title('Query Length vs. Counts (Wait-Only)', fontsize=16)
    plt.xlabel('Query Length (characters)', fontsize=12)
    plt.ylabel('Counts', fontsize=12)
    plt.xscale('log')
    plt.yscale('log')
    query_len_path = os.path.join(output_dir, 'query_length_vs_counts.png')
    plt.savefig(query_len_path, dpi=300)
    print(f"Saved query length vs. counts plot to {query_len_path}")
    plt.close()

    # --- Plot 3: Content Length vs. Counts ---
    plt.figure(figsize=(12, 7))
    sns.scatterplot(data=df, x='content_length', y='counts', alpha=0.5)
    plt.title('Content Length vs. Counts (Wait-Only)', fontsize=16)
    plt.xlabel('Content Length (characters)', fontsize=12)
    plt.ylabel('Counts', fontsize=12)
    plt.xscale('log')
    plt.yscale('log')
    content_len_path = os.path.join(output_dir, 'content_length_vs_counts.png')
    plt.savefig(content_len_path, dpi=300)
    print(f"Saved content length vs. counts plot to {content_len_path}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze and visualize the distribution of 'counts' and its relation to other features.")
    parser.add_argument('data_file', type=str, help='Path to the input jsonl data file.')
    parser.add_argument('--output_dir', type=str, default='analysis_results/data/wait_only_analysis', help='Directory to save the output plots.')
    
    args = parser.parse_args()
    analyze_distribution(args.data_file, args.output_dir)
