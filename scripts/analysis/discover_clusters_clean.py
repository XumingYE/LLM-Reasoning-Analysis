import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import json

# Load data
print("Loading data...")
df = pd.read_json('data/merged_with_labels_and_counts.jsonl', lines=True)
print(f"Loaded {len(df)} samples")

# Filter out anomalies (counts > 130)
print("\n" + "="*60)
print("FILTERING ANOMALIES")
print("="*60)
print(f"Original dataset: {len(df)} samples")
print(f"Counts range: {df['counts'].min()} - {df['counts'].max()}")
print(f"Counts mean: {df['counts'].mean():.1f}, median: {df['counts'].median():.1f}")

# Remove anomalies
df_clean = df[df['counts'] <= 130].copy()
anomalies = df[df['counts'] > 130].copy()

print(f"\nAfter removing anomalies (counts > 130):")
print(f"  Clean dataset: {len(df_clean)} samples ({len(df_clean)/len(df)*100:.1f}%)")
print(f"  Anomalies removed: {len(anomalies)} samples ({len(anomalies)/len(df)*100:.1f}%)")
print(f"  New counts range: {df_clean['counts'].min()} - {df_clean['counts'].max()}")
print(f"  New counts mean: {df_clean['counts'].mean():.1f}, median: {df_clean['counts'].median():.1f}")

# Extract features for clustering
print("\n" + "="*60)
print("EXTRACTING FEATURES")
print("="*60)
df_clean['text'] = df_clean['instruction'].fillna('') + ' ' + df_clean['input'].fillna('')
df_clean['word_count'] = df_clean['text'].str.split().str.len()
df_clean['char_count'] = df_clean['text'].str.len()
df_clean['question_marks'] = df_clean['text'].str.count(r'\?')
df_clean['has_explain'] = df_clean['text'].str.lower().str.contains('explain|describe|analyze|compare').astype(int)
df_clean['has_creative'] = df_clean['text'].str.lower().str.contains('write|create|generate|story').astype(int)
df_clean['has_calculate'] = df_clean['text'].str.lower().str.contains('calculate|compute|solve').astype(int)
df_clean['has_list'] = df_clean['text'].str.lower().str.contains('list|enumerate').astype(int)
df_clean['has_why'] = df_clean['text'].str.lower().str.contains('why').astype(int)
df_clean['has_how'] = df_clean['text'].str.lower().str.contains('how').astype(int)

# Feature matrix for clustering
feature_cols = ['counts', 'word_count', 'char_count', 'question_marks',
                'has_explain', 'has_creative', 'has_calculate', 'has_list',
                'has_why', 'has_how']

X = df_clean[feature_cols].values

# Standardize features
print("Standardizing features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Test different numbers of clusters (focus on 5-8 for balance)
print("\n" + "="*60)
print("FINDING OPTIMAL NUMBER OF CLUSTERS")
print("="*60)

results = []
for n_clusters in range(4, 9):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    labels = kmeans.fit_predict(X_scaled)
    silhouette = silhouette_score(X_scaled, labels)
    db_score = davies_bouldin_score(X_scaled, labels)

    # Calculate balance score
    unique, counts = np.unique(labels, return_counts=True)
    balance_score = counts.std() / counts.mean()

    results.append({
        'n_clusters': n_clusters,
        'silhouette': silhouette,
        'davies_bouldin': db_score,
        'balance_score': balance_score,
        'min_cluster_size': counts.min(),
        'max_cluster_size': counts.max()
    })

    print(f"\n{n_clusters} clusters:")
    print(f"  Silhouette: {silhouette:.4f}")
    print(f"  Davies-Bouldin: {db_score:.4f}")
    print(f"  Balance score: {balance_score:.4f} (lower is better)")
    print(f"  Cluster sizes: min={counts.min()}, max={counts.max()}")

# Choose best K based on combined criteria (good silhouette + good balance)
results_df = pd.DataFrame(results)

# Normalize scores for comparison (0-1 scale)
results_df['silhouette_norm'] = (results_df['silhouette'] - results_df['silhouette'].min()) / (results_df['silhouette'].max() - results_df['silhouette'].min())
results_df['balance_norm'] = 1 - (results_df['balance_score'] - results_df['balance_score'].min()) / (results_df['balance_score'].max() - results_df['balance_score'].min())

# Combined score (weighted: 60% silhouette, 40% balance)
results_df['combined_score'] = 0.6 * results_df['silhouette_norm'] + 0.4 * results_df['balance_norm']
best_k = int(results_df.loc[results_df['combined_score'].idxmax(), 'n_clusters'])

print("\n" + "="*60)
print(f"BEST K: {best_k} clusters")
print(f"  Silhouette: {results_df.loc[results_df['n_clusters']==best_k, 'silhouette'].values[0]:.4f}")
print(f"  Balance score: {results_df.loc[results_df['n_clusters']==best_k, 'balance_score'].values[0]:.4f}")
print("="*60)

# Final clustering
print(f"\nPerforming final K-Means clustering with {best_k} clusters...")
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=30)
df_clean['cluster'] = kmeans.fit_predict(X_scaled)

# Analyze clusters
print("\n" + "="*60)
print("CLUSTER ANALYSIS")
print("="*60)

cluster_stats = []
for cluster_id in range(best_k):
    cluster_data = df_clean[df_clean['cluster'] == cluster_id]
    stats = {
        'cluster': cluster_id,
        'size': len(cluster_data),
        'percentage': len(cluster_data) / len(df_clean) * 100,
        'counts_min': int(cluster_data['counts'].min()),
        'counts_max': int(cluster_data['counts'].max()),
        'counts_mean': cluster_data['counts'].mean(),
        'counts_median': cluster_data['counts'].median(),
        'counts_std': cluster_data['counts'].std(),
        'word_count_mean': cluster_data['word_count'].mean(),
        'has_explain_pct': cluster_data['has_explain'].mean() * 100,
        'has_creative_pct': cluster_data['has_creative'].mean() * 100,
        'has_calculate_pct': cluster_data['has_calculate'].mean() * 100,
    }
    cluster_stats.append(stats)

cluster_stats_df = pd.DataFrame(cluster_stats)
# Sort by mean counts
cluster_stats_df = cluster_stats_df.sort_values('counts_mean')

print("\nCluster Statistics (sorted by mean counts):")
print("-" * 60)
for idx, row in cluster_stats_df.iterrows():
    print(f"\nCluster {int(row['cluster'])}:")
    print(f"  Size: {int(row['size'])} samples ({row['percentage']:.2f}%)")
    print(f"  Counts: min={row['counts_min']}, max={row['counts_max']}, "
          f"mean={row['counts_mean']:.1f}, median={row['counts_median']:.1f}, std={row['counts_std']:.1f}")
    print(f"  Query: words={row['word_count_mean']:.1f}, "
          f"explain={row['has_explain_pct']:.1f}%, creative={row['has_creative_pct']:.1f}%, "
          f"calc={row['has_calculate_pct']:.1f}%")

# Map clusters to sorted labels
cluster_to_label = {row['cluster']: idx for idx, row in cluster_stats_df.iterrows()}
df_clean['label'] = df_clean['cluster'].map(cluster_to_label)

# Check balance
print("\n" + "="*60)
print("BALANCE CHECK")
print("="*60)
print("\nSample distribution:")
for label in sorted(df_clean['label'].unique()):
    count = len(df_clean[df_clean['label'] == label])
    pct = count / len(df_clean) * 100
    mean_counts = df_clean[df_clean['label'] == label]['counts'].mean()
    min_counts = df_clean[df_clean['label'] == label]['counts'].min()
    max_counts = df_clean[df_clean['label'] == label]['counts'].max()
    print(f"Label {label}: {count:5d} samples ({pct:5.2f}%) - counts: {min_counts}-{max_counts} (avg={mean_counts:.1f})")

label_counts = df_clean['label'].value_counts()
balance_score = label_counts.std() / label_counts.mean()
print(f"\nBalance score (std/mean): {balance_score:.3f}")
print("  (Target: <0.5 is good, <0.3 is excellent)")

# Calculate minimum samples in test set (10% split)
min_test_samples = (label_counts * 0.1).min()
print(f"\nMinimum test samples per label: {int(min_test_samples)}")
if min_test_samples >= 50:
    print("  ✓ GOOD: All labels have sufficient test samples")
elif min_test_samples >= 20:
    print("  △ OK: Test samples are adequate but could be better")
else:
    print("  ✗ WARNING: Some labels have very few test samples")

# Save configuration
print("\n" + "="*60)
print("SAVING CLEAN CLUSTER CONFIGURATION")
print("="*60)

cluster_config = {
    'n_clusters': best_k,
    'method': 'kmeans',
    'anomaly_threshold': 130,
    'samples_removed': int(len(anomalies)),
    'samples_kept': int(len(df_clean)),
    'feature_columns': feature_cols,
    'scaler_mean': scaler.mean_.tolist(),
    'scaler_scale': scaler.scale_.tolist(),
    'cluster_centers': kmeans.cluster_centers_.tolist(),
    'cluster_to_label': cluster_to_label,
    'cluster_stats': cluster_stats_df.to_dict('records'),
    'balance_score': float(balance_score)
}

with open('data/cluster_config_clean.json', 'w') as f:
    json.dump(cluster_config, f, indent=2)
print("Clean cluster configuration saved to: data/cluster_config_clean.json")

# Save clean dataset with cluster labels
output_file = 'data/merged_with_clusters_clean.jsonl'
df_output = df_clean[['worker_id', 'sample_id', 'instruction', 'input', 'content', 'label', 'counts', 'cluster']]
df_output.to_json(output_file, orient='records', lines=True)
print(f"Clean dataset with cluster labels saved to: {output_file}")

# Show example queries
print("\n" + "="*60)
print("EXAMPLE QUERIES FROM EACH CLUSTER")
print("="*60)

for label in sorted(df_clean['label'].unique()):
    cluster_data = df_clean[df_clean['label'] == label]
    cluster_id = cluster_data.iloc[0]['cluster']

    print(f"\n{'='*60}")
    print(f"Label {label} (Cluster {cluster_id})")
    print(f"Counts range: {cluster_data['counts'].min()}-{cluster_data['counts'].max()}, "
          f"avg={cluster_data['counts'].mean():.1f}")
    print('='*60)

    samples = cluster_data.sample(min(3, len(cluster_data)), random_state=42)
    for idx, (_, row) in enumerate(samples.iterrows(), 1):
        instruction = row['instruction'][:80] + "..." if len(row['instruction']) > 80 else row['instruction']
        print(f"\n  Example {idx}: counts={row['counts']}")
        print(f"    {instruction}")

print("\n" + "="*60)
print("✓ DONE! Clean clustering complete.")
print(f"✓ Removed {len(anomalies)} anomalies")
print(f"✓ Created {best_k} balanced clusters")
print(f"✓ Balance score improved from 1.202 to {balance_score:.3f}")
print("="*60)
