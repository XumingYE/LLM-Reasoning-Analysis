import argparse
import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def run_inference():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run inference on random samples using the trained regression model.")
    parser.add_argument('--model_path', type=str, default='predictor_regressor_model_equal_freq', help='Path to the trained model directory.')
    parser.add_argument('--data_path', type=str, default='data/merged_with_labels_and_counts.jsonl', help='Path to the .jsonl dataset.')
    parser.add_argument('--output_path', type=str, default='inference_results.md', help='Path to save the inference results.')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of random samples to test.')
    parser.add_argument('--num_bins', type=int, default=8, help='Number of bins used during training to recalculate bin edges.')
    args = parser.parse_args()

    # --- 1. Load Model and Tokenizer ---
    # The tokenizer must be loaded from the original base model, not the fine-tuned one.
    base_model_path = '/home/yexuming/model/hg/bert-base-uncased'
    print(f"Loading tokenizer from {base_model_path}...")
    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    model.eval() # Set model to evaluation mode

    # --- 2. Load Data and Recalculate Bins ---
    print(f"Loading dataset from {args.data_path}...")
    df = pd.read_json(args.data_path, lines=True)

    print(f"Re-calculating {args.num_bins} bins to understand ranges...")
    try:
        df['label'], bin_edges = pd.qcut(df['counts'], q=args.num_bins, labels=False, retbins=True, duplicates='drop')
        actual_num_bins = df['label'].nunique()
    except ValueError as e:
        print(f"Error: Could not create bins. {e}.")
        return

    bin_ranges = {}
    for i in range(actual_num_bins):
        min_val = df[df['label'] == i]['counts'].min()
        max_val = df[df['label'] == i]['counts'].max()
        bin_ranges[i] = f"{min_val} to {max_val}"

    # --- 3. Select Random Samples ---
    random_samples = df.sample(n=args.num_samples, random_state=42)

    # --- 4. Run Inference and Save Results ---
    print(f"Running inference on {args.num_samples} samples and saving to {args.output_path}...")
    with open(args.output_path, 'w', encoding='utf-8') as f:
        f.write("# Inference Results\n\n")
        f.write("This file contains predictions from the regression model on a few random samples.\n\n")
        f.write("## Bin Definitions\n")
        f.write("The model predicts a bin index. Here are the `counts` ranges for each bin:\n")
        for i, r in bin_ranges.items():
            f.write(f"- **Bin {i}:** `counts` from {r}\n")
        f.write("\n---\n\n")

        for index, row in random_samples.iterrows():
            # Prepare input
            text = row['instruction'] + ' ' + row['input']
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

            # Get prediction
            with torch.no_grad():
                outputs = model(**inputs)
                prediction = outputs.logits.item()

            # Write to file
            f.write(f"### Sample {index}\n\n")
            f.write(f"- **Instruction:** `{row['instruction']}`\n")
            f.write(f"- **Input:** `{row['input']}`\n")
            f.write(f"- **Content:**\n```\n{row['content']}\n```\n\n")
            f.write("**Analysis:**\n")
            f.write(f"- **Actual `counts`:** {row['counts']}\n")
            f.write(f"- **Actual Bin:** {row['label']}\n")
            f.write(f"- **Predicted Bin (raw float):** {prediction:.4f}\n")
            f.write(f"- **Predicted Bin (rounded):** {round(prediction)}\n\n")
            f.write("---\n\n")

    print("Inference complete.")

if __name__ == "__main__":
    run_inference()
