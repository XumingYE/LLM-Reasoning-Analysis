import pandas as pd

def analyze_distribution():
    data_path = 'data/merged_with_labels_and_counts.jsonl'
    print(f"Loading data from {data_path} to analyze 'counts' distribution...")
    df = pd.read_json(data_path, lines=True)

    print("\n--- Descriptive Statistics for 'counts' ---")
    # Calculate and print descriptive statistics
    stats = df['counts'].describe()
    print(stats)

    print("\n--- High-End Quantile Information for 'counts' ---")
    # Calculate and print high-end quantiles to understand the tail
    quantiles = [0.8, 0.9, 0.95, 0.98, 0.99, 1.0]
    high_quantiles = df['counts'].quantile(quantiles)
    print(high_quantiles)

if __name__ == "__main__":
    analyze_distribution()

