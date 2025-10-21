import argparse
import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def run_inference():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run inference on random samples using the trained weighted classifier.")
    parser.add_argument('--model_path', type=str, default='predictor_weighted_classifier_manual_10_bins', help='Path to the trained model directory.')
    parser.add_argument('--data_path', type=str, default='data/merged_with_labels_and_counts.jsonl', help='Path to the .jsonl dataset.')
    parser.add_argument('--output_path', type=str, default='weighted_classifier_inference_results.md', help='Path to save the inference results.')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of random samples to test.')
    args = parser.parse_args()

    # --- 1. Load Model and Tokenizer ---
    print(f"Loading model and tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    model.eval() # Set model to evaluation mode

    # --- 2. Load Data and Define Bins ---
    print(f"Loading dataset from {args.data_path}...")
    df = pd.read_json(args.data_path, lines=True)

    bin_edges = [0, 1, 4, 8, 17, 28, 41, 66, 98, 200, 1000]
    num_bins = len(bin_edges) - 1
    labels = list(range(num_bins))
    df['label'] = pd.cut(df['counts'], bins=bin_edges, labels=labels, right=True, include_lowest=True)
    df.dropna(subset=['label'], inplace=True)
    df['label'] = df['label'].astype(int)

    # --- 3. Split Data and Select Random Samples from Test Set ---
    # We must split the data in the same way as training to ensure we only sample from the test set.
    from datasets import Dataset
    df['text'] = (df['instruction'].fillna('') + ' ' + df['input'].fillna(''))
    raw_dataset = Dataset.from_pandas(df)
    
    from datasets import ClassLabel
    raw_dataset = raw_dataset.cast_column("label", ClassLabel(num_classes=num_bins))
    
    dataset_split = raw_dataset.train_test_split(test_size=0.1, seed=42, stratify_by_column='label')
    eval_df = dataset_split['test'].to_pandas()

    print(f"Sampling from the evaluation set ({len(eval_df)} samples)...")
    # Ensure we don't sample more than available
    num_samples = min(args.num_samples, len(eval_df))
    random_samples = eval_df.sample(n=num_samples, random_state=42)

    # --- 4. Run Inference and Save Results ---
    print(f"Running inference on {args.num_samples} samples and saving to {args.output_path}...")
    with open(args.output_path, 'w', encoding='utf-8') as f:
        f.write("# Weighted Classifier Inference Results\n\n")
        f.write("This file contains predictions from the **weighted classification model** on random samples.\n\n")
        f.write("## Bin Definitions\n")
        f.write("The model predicts a bin index. Here are the `counts` ranges for each bin:\n")
        for i in labels:
            min_val = df[df['label'] == i]['counts'].min()
            max_val = df[df['label'] == i]['counts'].max()
            f.write(f"- **Bin {i}:** `counts` from {min_val} to {max_val}\n")
        f.write("\n---\n\n")

        for index, row in random_samples.iterrows():
            # Prepare input
            text = row['instruction'] + ' ' + row['input']
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

            # Get prediction
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                prediction = torch.argmax(logits, dim=-1).item()

            # Write to file
            f.write(f"### Sample {index}\n\n")
            f.write(f"- **Instruction:** `{row['instruction']}`\n")
            f.write(f"- **Input:** `{row['input']}`\n")
            f.write(f"- **Content:**\n```\n{row['content']}\n```\n\n")
            f.write("**Analysis:**\n")
            f.write(f"- **Actual `counts`:** {row['counts']}\n")
            f.write(f"- **Actual Bin:** {row['label']}\n")
            f.write(f"- **Predicted Bin:** {prediction}\n\n")
            f.write("---\n\n")

    print("Inference complete.")

if __name__ == "__main__":
    run_inference()
