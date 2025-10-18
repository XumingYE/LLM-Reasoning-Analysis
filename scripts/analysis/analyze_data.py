
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

def analyze_data(input_file, output_dir, file_prefix):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load the data from the jsonl file
    data = []
    with open(input_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    df = pd.DataFrame(data)
    
    # Create query column and calculate lengths
    df['query'] = df['instruction'].fillna('') + df['input'].fillna('')
    df['query_length'] = df['query'].str.len()
    df['content_length'] = df['content'].str.len()
    
    # --- Visualization ---
    sns.set_style("whitegrid")
    
    # 1. Distribution of counts
    plt.figure(figsize=(10, 6))
    sns.histplot(df['counts'], bins=30, kde=True)
    plt.title(f'Distribution of Counts ({file_prefix})')
    plt.xlabel('Counts')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(output_dir, f'{file_prefix}_counts_distribution.png'))
    plt.close()
    
    # 2. Query length vs. counts
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='query_length', y='counts', alpha=0.5)
    plt.title(f'Query Length vs. Counts ({file_prefix})')
    plt.xlabel('Query Length')
    plt.ylabel('Counts')
    plt.savefig(os.path.join(output_dir, f'{file_prefix}_query_length_vs_counts.png'))
    plt.close()
    
    # 3. Content length vs. counts
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='content_length', y='counts', alpha=0.5)
    plt.title(f'Content Length vs. Counts ({file_prefix})')
    plt.xlabel('Content Length')
    plt.ylabel('Counts')
    plt.savefig(os.path.join(output_dir, f'{file_prefix}_content_length_vs_counts.png'))
    plt.close()
    
    # --- Noise Analysis ---
    noisy_samples = df[(df['content_length'] > 2000) & (df['counts'] <= 5)]
    noise_output_path = os.path.join(output_dir, f'{file_prefix}_noisy_samples.jsonl')
    noisy_samples.to_json(noise_output_path, orient='records', lines=True)
    
    print(f"Analysis complete for {file_prefix}.")
    print(f"Plots saved in {output_dir}")
    print(f"Found {len(noisy_samples)} noisy samples, saved to {noise_output_path}")

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python analyze_data.py <input_file> <output_dir> <file_prefix>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_dir_path = sys.argv[2]
    prefix = sys.argv[3]
    analyze_data(input_path, output_dir_path, prefix)
