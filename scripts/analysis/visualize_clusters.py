
import pandas as pd
import json
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import os

def visualize_clusters():
    """
    This function loads the clustered data, applies PCA for dimensionality reduction,
    and creates a visualization of the clusters.
    """
    # Define paths
    data_path = 'data/merged_with_clusters_clean.jsonl'
    config_path = 'data/cluster_config_clean.json'
    output_dir = 'analysis_results/data'
    output_path = os.path.join(output_dir, 'cluster_visualization_pca.png')

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load data and config
    print(f"Loading data from {data_path}...")
    df = pd.read_json(data_path, lines=True)
    
    print(f"Loading config from {config_path}...")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    feature_cols = config['feature_columns']
    labels = df['label']

    # --- Re-calculate features as they are not in the file ---
    print("Re-calculating features for visualization...")
    df['text'] = df['instruction'].fillna('') + ' ' + df['input'].fillna('')
    df['word_count'] = df['text'].str.split().str.len()
    df['char_count'] = df['text'].str.len()
    df['question_marks'] = df['text'].str.count(r'\?')
    
    # Keyword features
    df['has_explain'] = df['text'].str.lower().str.contains('explain|describe|analyze|compare').astype(int)
    df['has_creative'] = df['text'].str.lower().str.contains('write|create|generate|story').astype(int)
    df['has_calculate'] = df['text'].str.lower().str.contains('calculate|compute|solve').astype(int)
    df['has_list'] = df['text'].str.lower().str.contains('list|enumerate').astype(int)
    df['has_why'] = df['text'].str.lower().str.contains('why').astype(int)
    df['has_how'] = df['text'].str.lower().str.contains('how').astype(int)
    # --- End of feature calculation ---
    
    # Prepare feature matrix
    print("Preparing feature matrix...")
    X = df[feature_cols].values
    
    # Standardize features
    print("Standardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    print("Applying PCA for dimensionality reduction...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Create a DataFrame for plotting
    df_pca = pd.DataFrame(data=X_pca, columns=['PCA Component 1', 'PCA Component 2'])
    df_pca['label'] = labels
    df_pca = df_pca.sort_values('label') # Sort for consistent legend order

    # Create the plot
    print(f"Generating plot and saving to {output_path}...")
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(16, 10))
    
    # Use a qualitative palette
    palette = sns.color_palette("deep", n_colors=df_pca['label'].nunique())
    
    sns.scatterplot(
        x='PCA Component 1',
        y='PCA Component 2',
        hue='label',
        palette=palette,
        data=df_pca,
        s=50,
        alpha=0.7,
        edgecolor='k',
        linewidth=0.5
    )

    # Get handles and labels for the legend
    handles, legend_labels = plt.gca().get_legend_handles_labels()
    
    # Create descriptive labels for the legend
    cluster_stats = config['cluster_stats']
    label_descriptions = {}
    for stat in cluster_stats:
        label_id = config['cluster_to_label'][str(float(stat['cluster']))]
        avg_counts = stat['counts_mean']
        
        # Identify dominant feature
        dominant_feature = 'General'
        if stat.get('has_explain_pct', 0) > 90:
            dominant_feature = 'Explanation'
        elif stat.get('has_creative_pct', 0) > 90:
            dominant_feature = 'Creative'
        elif stat.get('has_calculate_pct', 0) > 90:
            dominant_feature = 'Coding/Calc'
        
        label_descriptions[str(label_id)] = f'Label {label_id}: {dominant_feature} (Avg Counts: {avg_counts:.1f})'

    # Sort legend labels numerically and apply descriptions
    sorted_legend_info = sorted(zip(handles, legend_labels), key=lambda x: int(x[1]))
    sorted_handles = [info[0] for info in sorted_legend_info]
    sorted_labels_text = [label_descriptions[info[1]] for info in sorted_legend_info]

    plt.legend(sorted_handles, sorted_labels_text, title='Clusters', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    plt.title('2D PCA Visualization of K-means Clusters', fontsize=18, weight='bold')
    plt.xlabel('Principal Component 1', fontsize=12)
    plt.ylabel('Principal Component 2', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Save the figure
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print("Visualization complete.")
    print(f"Plot saved to {output_path}")

if __name__ == '__main__':
    visualize_clusters()
