import pandas as pd
import os

def compare_distributions():
    """
    Loads two datasets (original counts and wait-only counts), calculates descriptive
    statistics for their 'counts' columns, and saves the comparison to a markdown file.
    """
    # Define paths
    original_data_path = 'data/merged_with_labels_and_counts.jsonl'
    wait_only_data_path = 'data/cleaned_data_wait_only.jsonl'
    output_path = 'data/counts_distribution_comparison.md'

    print(f"Loading original counts data from {original_data_path}...")
    try:
        df_original = pd.read_json(original_data_path, lines=True)
    except FileNotFoundError:
        print(f"Error: Original data file not found at {original_data_path}")
        return

    print(f"Loading wait-only counts data from {wait_only_data_path}...")
    try:
        df_wait_only = pd.read_json(wait_only_data_path, lines=True)
    except FileNotFoundError:
        print(f"Error: Wait-only data file not found at {wait_only_data_path}")
        return

    # --- Calculate Statistics ---
    print("Calculating descriptive statistics for both datasets...")
    stats_original = df_original['counts'].describe()
    stats_wait_only = df_wait_only['counts'].describe()

    # Calculate high-end quantiles
    quantiles = [0.8, 0.9, 0.95, 0.98, 0.99, 1.0]
    high_quantiles_original = df_original['counts'].quantile(quantiles)
    high_quantiles_wait_only = df_wait_only['counts'].quantile(quantiles)

    # --- Format Output ---
    output_md = "# Comparison of `counts` Distributions\n\n"
    output_md += "This report compares the descriptive statistics of the `counts` field from two different calculation methods.\n\n"

    # --- Table for main statistics ---
    output_md += "## Main Descriptive Statistics\n\n"
    
    # Combine stats into a single DataFrame for easy table creation
    comparison_df = pd.DataFrame({
        'Metric': ['Count', 'Mean', 'Std Dev', 'Min', '25%', '50% (Median)', '75%', 'Max'],
        'Original Counts (Keywords)': [
            f"{stats_original['count']:.0f}",
            f"{stats_original['mean']:.2f}",
            f"{stats_original['std']:.2f}",
            f"{stats_original['min']:.0f}",
            f"{stats_original['25%']:.0f}",
            f"{stats_original['50%']:.0f}",
            f"{stats_original['75%']:.0f}",
            f"{stats_original['max']:.0f}"
        ],
        'New Counts (Wait-Only)': [
            f"{stats_wait_only['count']:.0f}",
            f"{stats_wait_only['mean']:.2f}",
            f"{stats_wait_only['std']:.2f}",
            f"{stats_wait_only['min']:.0f}",
            f"{stats_wait_only['25%']:.0f}",
            f"{stats_wait_only['50%']:.0f}",
            f"{stats_wait_only['75%']:.0f}",
            f"{stats_wait_only['max']:.0f}"
        ]
    })
    output_md += comparison_df.to_markdown(index=False)
    output_md += "\n\n"

    # --- Table for high-end quantiles ---
    output_md += "## High-End Quantile Information\n\n"
    
    quantile_df = pd.DataFrame({
        'Quantile': [f"{q*100:.0f}%" for q in quantiles],
        'Original Counts (Keywords)': [f"{v:.0f}" for v in high_quantiles_original],
        'New Counts (Wait-Only)': [f"{v:.0f}" for v in high_quantiles_wait_only]
    })
    output_md += quantile_df.to_markdown(index=False)
    output_md += "\n"

    # --- Save Output ---
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(output_md)
        print(f"Successfully saved comparison report to {output_path}")
    except Exception as e:
        print(f"Error saving file: {e}")

if __name__ == '__main__':
    compare_distributions()
