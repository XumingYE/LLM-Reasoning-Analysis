import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Load data
print("Loading data...")
df = pd.read_json('data/merged_with_labels_and_counts.jsonl', lines=True)
print(f"Loaded {len(df)} samples")

# Extract features for clustering
print("\nExtracting features...")
df['text'] = df['instruction'].fillna('') + ' ' + df['input'].fillna('')
df['word_count'] = df['text'].str.split().str.len()
df['char_count'] = df['text'].str.len()
df['question_marks'] = df['text'].str.count('\?')
df['has_explain'] = df['text'].str.lower().str.contains('explain|describe|analyze|compare').astype(int)
df['has_creative'] = df['text'].str.lower().str.contains('write|create|generate|story').astype(int)
df['has_calculate'] = df['text'].str.lower().str.contains('calculate|compute|solve').astype(int)
df['has_list'] = df['text'].str.lower().str.contains('list|enumerate').astype(int)
df['has_why'] = df['text'].str.lower().str.contains('why').astype(int)
df['has_how'] = df['text'].str.lower().str.contains('how').astype(int)

# Feature matrix for clustering
feature_cols = ['counts', 'word_count', 'char_count', 'question_marks',
                'has_explain', 'has_creative', 'has_calculate', 'has_list',
                'has_why', 'has_how']

X = df[feature_cols].values

# Standardize features (important for clustering)
print("Standardizing features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Test different numbers of clusters
print("\n" + "="*60)
print("FINDING OPTIMAL NUMBER OF CLUSTERS")
print("="*60)

results = []
for n_clusters in range(3, 11):
    # K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels_km = kmeans.fit_predict(X_scaled)
    silhouette_km = silhouette_score(X_scaled, labels_km)
    db_km = davies_bouldin_score(X_scaled, labels_km)
    inertia_km = kmeans.inertia_

    # Hierarchical Clustering
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
    labels_hc = hierarchical.fit_predict(X_scaled)
    silhouette_hc = silhouette_score(X_scaled, labels_hc)
    db_hc = davies_bouldin_score(X_scaled, labels_hc)

    results.append({
        'n_clusters': n_clusters,
        'kmeans_silhouette': silhouette_km,
        'kmeans_davies_bouldin': db_km,
        'kmeans_inertia': inertia_km,
        'hierarchical_silhouette': silhouette_hc,
        'hierarchical_davies_bouldin': db_hc
    })

    print(f"\n{n_clusters} clusters:")
    print(f"  K-Means:      Silhouette={silhouette_km:.4f}, Davies-Bouldin={db_km:.4f}")
    print(f"  Hierarchical: Silhouette={silhouette_hc:.4f}, Davies-Bouldin={db_hc:.4f}")

# Find best K based on silhouette score
results_df = pd.DataFrame(results)
best_k_km = results_df.loc[results_df['kmeans_silhouette'].idxmax(), 'n_clusters']
best_k_hc = results_df.loc[results_df['hierarchical_silhouette'].idxmax(), 'n_clusters']

print("\n" + "="*60)
print(f"Best K (K-Means): {int(best_k_km)} clusters (Silhouette: {results_df['kmeans_silhouette'].max():.4f})")
print(f"Best K (Hierarchical): {int(best_k_hc)} clusters (Silhouette: {results_df['hierarchical_silhouette'].max():.4f})")
print("="*60)

# Use the best K-Means result
best_k = int(best_k_km)
print(f"\nUsing K-Means with {best_k} clusters...")
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=20)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Analyze clusters
print("\n" + "="*60)
print("CLUSTER ANALYSIS")
print("="*60)

cluster_stats = []
for cluster_id in range(best_k):
    cluster_data = df[df['cluster'] == cluster_id]
    stats = {
        'cluster': cluster_id,
        'size': len(cluster_data),
        'percentage': len(cluster_data) / len(df) * 100,
        'counts_min': cluster_data['counts'].min(),
        'counts_max': cluster_data['counts'].max(),
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
cluster_stats_df = cluster_stats_df.sort_values('counts_mean')

print("\nCluster Statistics (sorted by mean counts):")
print("-" * 60)
for _, row in cluster_stats_df.iterrows():
    print(f"\nCluster {int(row['cluster'])}:")
    print(f"  Size: {int(row['size'])} samples ({row['percentage']:.2f}%)")
    print(f"  Counts: min={int(row['counts_min'])}, max={int(row['counts_max'])}, "
          f"mean={row['counts_mean']:.1f}, median={row['counts_median']:.1f}, std={row['counts_std']:.1f}")
    print(f"  Query features: word_count={row['word_count_mean']:.1f}, "
          f"explain={row['has_explain_pct']:.1f}%, creative={row['has_creative_pct']:.1f}%, "
          f"calculate={row['has_calculate_pct']:.1f}%")

# Map clusters to sorted labels (0 = lowest mean counts, N-1 = highest)
cluster_to_label = {row['cluster']: idx for idx, row in cluster_stats_df.iterrows()}
df['label'] = df['cluster'].map(cluster_to_label)

print("\n" + "="*60)
print("CLUSTER BALANCE CHECK")
print("="*60)
print("\nSample distribution after cluster-based labeling:")
for label in sorted(df['label'].unique()):
    count = len(df[df['label'] == label])
    pct = count / len(df) * 100
    mean_counts = df[df['label'] == label]['counts'].mean()
    print(f"Label {label}: {count:5d} samples ({pct:5.2f}%) - avg counts: {mean_counts:.1f}")

# Calculate balance score (smaller is more balanced)
label_counts = df['label'].value_counts()
balance_score = label_counts.std() / label_counts.mean()
print(f"\nBalance score (std/mean): {balance_score:.3f}")
print("  (Lower is better. <0.5 is good, <0.3 is excellent)")

# Save cluster boundaries for training
print("\n" + "="*60)
print("SAVING CLUSTER CONFIGURATION")
print("="*60)

cluster_config = {
    'n_clusters': best_k,
    'method': 'kmeans',
    'feature_columns': feature_cols,
    'scaler_mean': scaler.mean_.tolist(),
    'scaler_scale': scaler.scale_.tolist(),
    'cluster_centers': kmeans.cluster_centers_.tolist(),
    'cluster_to_label': cluster_to_label,
    'cluster_stats': cluster_stats_df.to_dict('records')
}

with open('data/cluster_config.json', 'w') as f:
    json.dump(cluster_config, f, indent=2)
print("Cluster configuration saved to: data/cluster_config.json")

# Save dataset with cluster labels
output_file = 'data/merged_with_clusters.jsonl'
df_output = df[['worker_id', 'sample_id', 'instruction', 'input', 'content', 'label', 'counts', 'cluster']]
df_output.to_json(output_file, orient='records', lines=True)
print(f"Dataset with cluster labels saved to: {output_file}")

# Show example queries from each cluster
print("\n" + "="*60)
print("EXAMPLE QUERIES FROM EACH CLUSTER")
print("="*60)

for label in sorted(df['label'].unique()):
    cluster_data = df[df['label'] == label]
    print(f"\n{'='*60}")
    print(f"Label {label} (Cluster {cluster_data.iloc[0]['cluster']}) - "
          f"Avg counts: {cluster_data['counts'].mean():.1f}")
    print('='*60)
    samples = cluster_data.sample(min(3, len(cluster_data)), random_state=42)
    for idx, (_, row) in enumerate(samples.iterrows(), 1):
        print(f"\nExample {idx}:")
        print(f"  Instruction: {row['instruction'][:100]}...")
        print(f"  Counts: {row['counts']}")
        print(f"  Word count: {row['word_count']}")

print("\n" + "="*60)
print("DONE! Ready to train with cluster-based labels.")
print("="*60)
