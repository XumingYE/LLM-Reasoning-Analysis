import argparse
import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

def extract_enhanced_features(text):
    """Extract features for text enhancement (same as training)"""
    word_count = len(text.split())

    # Task type detection
    has_explain = any(word in text.lower() for word in ['explain', 'describe', 'analyze', 'compare', 'evaluate', 'discuss'])
    has_creative = any(word in text.lower() for word in ['write', 'create', 'generate', 'compose', 'design', 'imagine', 'story'])
    has_calculate = any(word in text.lower() for word in ['calculate', 'compute', 'solve', 'find', 'determine'])
    has_list = any(word in text.lower() for word in ['list', 'enumerate', 'identify', 'name'])
    has_why = 'why' in text.lower()
    has_how = 'how' in text.lower()

    # Build feature tokens
    feature_tokens = []

    if word_count < 10:
        feature_tokens.append('[SHORT]')
    elif word_count < 25:
        feature_tokens.append('[MEDIUM]')
    else:
        feature_tokens.append('[LONG]')

    if has_explain:
        feature_tokens.append('[EXPLAIN]')
    if has_creative:
        feature_tokens.append('[CREATIVE]')
    if has_calculate:
        feature_tokens.append('[CALCULATE]')
    if has_list:
        feature_tokens.append('[LIST]')
    if has_why:
        feature_tokens.append('[WHY]')
    if has_how:
        feature_tokens.append('[HOW]')

    feature_prefix = ' '.join(feature_tokens)
    enhanced_text = f"{feature_prefix} {text}" if feature_tokens else text

    return enhanced_text


def load_cluster_info(config_path='data/cluster_config_clean.json'):
    """Load cluster configuration and statistics"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def get_label_description(label, cluster_stats):
    """Get human-readable description for a label"""
    # Find the cluster stats for this label
    stats = None
    for stat in cluster_stats:
        if stat['cluster'] == label:
            stats = stat
            break

    if stats is None:
        return f"Label {label}"

    # Build description based on characteristics
    descriptions = []

    # Task type based on keyword percentages
    if stats['has_explain_pct'] > 60:
        descriptions.append("Explanation/Analysis")
    if stats['has_creative_pct'] > 60:
        descriptions.append("Creative/Writing")
    if stats['has_calculate_pct'] > 60:
        descriptions.append("Calculation/Coding")

    # Add complexity indicator
    mean_counts = stats['counts_mean']
    if mean_counts < 10:
        complexity = "Simple"
    elif mean_counts < 15:
        complexity = "Moderate"
    else:
        complexity = "Complex"

    # Word count characteristic
    word_count = stats['word_count_mean']
    if word_count > 50:
        length = "Long-form"
    elif word_count > 15:
        length = "Medium-form"
    else:
        length = "Short-form"

    if descriptions:
        task_type = " + ".join(descriptions)
    else:
        task_type = "General"

    return f"{task_type} ({complexity}, {length})"


def run_inference():
    parser = argparse.ArgumentParser(description="Run inference on queries using cluster-based model")
    parser.add_argument('--model_path', type=str, default='predictor_deberta_clusters_clean',
                        help='Path to trained model')
    parser.add_argument('--config_path', type=str, default='data/cluster_config_clean.json',
                        help='Path to cluster configuration')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to .jsonl file with test queries (optional)')
    parser.add_argument('--query', type=str, default=None,
                        help='Single query to test (optional)')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to test from data_path')
    parser.add_argument('--output_path', type=str, default='cluster_inference_results.md',
                        help='Path to save results')
    parser.add_argument('--show_probabilities', action='store_true',
                        help='Show prediction probabilities for all labels')
    args = parser.parse_args()

    # Load model and tokenizer
    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Model loaded on {device}")

    # Load cluster info
    print(f"Loading cluster configuration from {args.config_path}...")
    cluster_config = load_cluster_info(args.config_path)
    cluster_stats = cluster_config['cluster_stats']

    # Map cluster IDs to labels (sorted by mean counts)
    cluster_to_label = {stat['cluster']: idx for idx, stat in enumerate(cluster_stats)}
    label_to_cluster = {v: k for k, v in cluster_to_label.items()}

    # Prepare label descriptions
    label_descriptions = {}
    for label in range(cluster_config['n_clusters']):
        cluster_id = label_to_cluster[label]
        label_descriptions[label] = get_label_description(cluster_id, cluster_stats)

    print(f"\nLabel Descriptions:")
    print("="*70)
    for label in sorted(label_descriptions.keys()):
        print(f"  Label {label}: {label_descriptions[label]}")
    print("="*70 + "\n")

    # Prepare queries
    queries = []

    if args.query:
        # Single query from command line
        queries.append({
            'instruction': args.query,
            'input': '',
            'source': 'command_line'
        })
    elif args.data_path:
        # Load from file
        print(f"Loading queries from {args.data_path}...")
        df = pd.read_json(args.data_path, lines=True)

        # Sample random queries
        if 'label' in df.columns:
            # If dataset has labels, do stratified sampling
            sampled = df.groupby('label', group_keys=False).apply(
                lambda x: x.sample(min(len(x), max(1, args.num_samples // cluster_config['n_clusters'])), random_state=42)
            ).head(args.num_samples)
        else:
            sampled = df.sample(min(args.num_samples, len(df)), random_state=42)

        for _, row in sampled.iterrows():
            query_dict = {
                'instruction': row.get('instruction', ''),
                'input': row.get('input', ''),
                'source': 'dataset'
            }
            if 'label' in row:
                query_dict['true_label'] = int(row['label'])
            if 'counts' in row:
                query_dict['true_counts'] = int(row['counts'])
            queries.append(query_dict)

        print(f"Loaded {len(queries)} queries")
    else:
        # Demo queries
        print("No query or data_path provided. Using demo queries...")
        queries = [
            {'instruction': 'What are the three primary colors?', 'input': '', 'source': 'demo'},
            {'instruction': 'Write a short story about a dragon', 'input': '', 'source': 'demo'},
            {'instruction': 'Explain how photosynthesis works', 'input': '', 'source': 'demo'},
            {'instruction': 'Calculate the area of a circle with radius 5', 'input': '', 'source': 'demo'},
            {'instruction': 'List five benefits of exercise', 'input': '', 'source': 'demo'},
            {'instruction': 'Why is the sky blue?', 'input': '', 'source': 'demo'},
            {'instruction': 'How do I make chocolate chip cookies?', 'input': '', 'source': 'demo'},
            {'instruction': 'Compare democracy and autocracy', 'input': '', 'source': 'demo'},
        ]

    # Run inference
    print(f"\nRunning inference on {len(queries)} queries...\n")

    results = []
    correct = 0
    total_with_labels = 0

    with open(args.output_path, 'w', encoding='utf-8') as f:
        f.write("# Cluster-Based Model Inference Results\n\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Total queries: {len(queries)}\n\n")

        f.write("## Label Descriptions\n\n")
        for label in sorted(label_descriptions.keys()):
            cluster_id = label_to_cluster[label]
            stats = next(s for s in cluster_stats if s['cluster'] == cluster_id)
            f.write(f"**Label {label}**: {label_descriptions[label]}\n")
            f.write(f"  - Counts range: {stats['counts_min']}-{stats['counts_max']} (avg: {stats['counts_mean']:.1f})\n")
            f.write(f"  - Size: {stats['size']} samples ({stats['percentage']:.2f}%)\n\n")

        f.write("\n---\n\n")
        f.write("## Predictions\n\n")

        for idx, query in enumerate(queries, 1):
            # Prepare text
            instruction = query['instruction']
            input_text = query.get('input', '')
            combined_text = instruction + ' ' + input_text if input_text else instruction
            enhanced_text = extract_enhanced_features(combined_text)

            # Tokenize
            inputs = tokenizer(enhanced_text, return_tensors="pt", padding=True,
                              truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Predict
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                predicted_label = int(torch.argmax(logits, dim=-1).item())

            # Get cluster ID for predicted label
            predicted_cluster = label_to_cluster[predicted_label]
            predicted_stats = next(s for s in cluster_stats if s['cluster'] == predicted_cluster)

            # Check accuracy if true label exists
            is_correct = None
            if 'true_label' in query:
                total_with_labels += 1
                is_correct = (predicted_label == query['true_label'])
                if is_correct:
                    correct += 1

            # Store result
            result = {
                'query': combined_text[:100],
                'predicted_label': predicted_label,
                'predicted_description': label_descriptions[predicted_label],
                'confidence': float(probabilities[predicted_label]),
                'is_correct': is_correct
            }
            results.append(result)

            # Write to file
            f.write(f"### Query {idx}\n\n")
            f.write(f"**Instruction**: {instruction}\n\n")
            if input_text:
                f.write(f"**Input**: {input_text}\n\n")

            f.write(f"**Prediction**:\n")
            f.write(f"  - **Label**: {predicted_label}\n")
            f.write(f"  - **Type**: {label_descriptions[predicted_label]}\n")
            f.write(f"  - **Confidence**: {probabilities[predicted_label]:.2%}\n")
            f.write(f"  - **Expected counts range**: {predicted_stats['counts_min']}-{predicted_stats['counts_max']} (avg: {predicted_stats['counts_mean']:.1f})\n")

            if 'true_label' in query:
                f.write(f"\n**Ground Truth**:\n")
                f.write(f"  - **True Label**: {query['true_label']}\n")
                f.write(f"  - **True Type**: {label_descriptions[query['true_label']]}\n")
                if 'true_counts' in query:
                    f.write(f"  - **True Counts**: {query['true_counts']}\n")
                f.write(f"  - **Correct**: {'✓ YES' if is_correct else '✗ NO'}\n")

            if args.show_probabilities:
                f.write(f"\n**All Label Probabilities**:\n")
                sorted_indices = np.argsort(probabilities)[::-1]
                for label_idx in sorted_indices[:5]:  # Top 5
                    f.write(f"  - Label {label_idx} ({label_descriptions[label_idx]}): {probabilities[label_idx]:.2%}\n")

            f.write("\n---\n\n")

    # Print summary
    print("="*70)
    print("INFERENCE SUMMARY")
    print("="*70)

    if total_with_labels > 0:
        accuracy = correct / total_with_labels
        print(f"Accuracy: {correct}/{total_with_labels} = {accuracy:.2%}")

    # Show distribution of predictions
    predicted_labels = [r['predicted_label'] for r in results]
    print(f"\nPrediction Distribution:")
    for label in sorted(set(predicted_labels)):
        count = predicted_labels.count(label)
        print(f"  Label {label} ({label_descriptions[label]}): {count} queries ({count/len(queries)*100:.1f}%)")

    print(f"\nResults saved to: {args.output_path}")
    print("="*70)


if __name__ == "__main__":
    run_inference()
