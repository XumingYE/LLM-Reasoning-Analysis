
import pandas as pd
import numpy as np
import torch
import argparse
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

def run_inference(args):
    # --- 1. Load Model and Tokenizer ---
    print(f"Loading tokenizer from {args.tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    model.eval() # Set model to evaluation mode

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # --- 2. Load and Prepare Data ---
    print(f"Loading data from {args.data_path}...")
    df = pd.read_json(args.data_path, lines=True)

    # Recreate the exact same train/test split as in training
    _, eval_df = train_test_split(df, test_size=0.1, random_state=42)

    # Use manual, rule-based stratified sampling
    if args.num_samples_per_bin > 0:
        bins = [
            eval_df[eval_df['counts'] == 1],
            eval_df[eval_df['counts'] == 2],
            eval_df[eval_df['counts'] == 3],
            eval_df[(eval_df['counts'] >= 4) & (eval_df['counts'] <= 10)],
            eval_df[eval_df['counts'] > 10]
        ]
        
        samples = []
        for i, bin_df in enumerate(bins):
            # Ensure we don't try to sample more than available
            n_samples = min(len(bin_df), args.num_samples_per_bin)
            if n_samples > 0:
                samples.append(bin_df.sample(n=n_samples, random_state=42))
        
        if samples:
            eval_df = pd.concat(samples)
        else:
            eval_df = pd.DataFrame() # Empty dataframe if no samples found

    elif args.num_samples > 0 and args.num_samples < len(eval_df):
        eval_df = eval_df.sample(n=args.num_samples, random_state=42)
    
    print(f"Running inference on {len(eval_df)} samples...")

    # --- 3. Run Inference and Format Output ---
    markdown_output = "# DeBERTa Regressor Inference Results\n\n"
    markdown_output += "This file contains predictions from the DeBERTa regression model on random test samples.\n\n"

    for _, row in tqdm(eval_df.iterrows(), total=len(eval_df)):
        # Prepare input
        text = row['instruction'] + ' ' + row.get('input', '')
        inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)

        # Get prediction
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_log_counts = outputs.logits.item()

        # Inverse transform the prediction
        predicted_counts = np.expm1(predicted_log_counts)

        # Append to markdown
        markdown_output += f"### Sample ID: {row.get('sample_id', 'N/A')}\n\n"
        markdown_output += f"- **Instruction:** `{row['instruction']}`\n"
        if row.get('input'):
            markdown_output += f"- **Input:** `{row['input']}`\n"
        
        # Truncate content for readability
        content = row.get('content', '')
        truncated_content = content[:1500] + '... [truncated]' if len(content) > 1500 else content
        markdown_output += f"- **Content:**\n```\n{truncated_content}\n```\n\n"
        
        markdown_output += f"**Analysis:**\n"
        markdown_output += f"- **Actual `counts`:** {row['counts']}\n"
        markdown_output += f"- **Predicted `counts` (raw float):** {predicted_counts:.4f}\n"
        markdown_output += f"- **Predicted `counts` (rounded):** {round(predicted_counts)}\n\n"
        markdown_output += "---\n\n"

    # --- 4. Save Output ---
    with open(args.output_path, 'w', encoding='utf-8') as f:
        f.write(markdown_output)

    print(f"Inference complete. Results saved to {args.output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run inference with the trained DeBERTa regression model.")
    
    parser.add_argument("--model_path", type=str, default="predictor_deberta_regressor_wait_only/checkpoint-6648", help="Path to the trained model directory.")
    parser.add_argument("--data_path", type=str, default="data/cleaned_data_wait_only.jsonl", help="Path to the cleaned data file.")
    parser.add_argument("--tokenizer_path", type=str, default="microsoft/deberta-v3-base", help="Path or name of the tokenizer.")
    parser.add_argument("--output_path", type=str, default="inference_results_deberta_regressor.md", help="Path to save the markdown output file.")
    parser.add_argument("--num_samples_per_bin", type=int, default=2, help="Number of samples to select from each quantile-based bin for stratified sampling.")

    args = parser.parse_args()
    run_inference(args)
