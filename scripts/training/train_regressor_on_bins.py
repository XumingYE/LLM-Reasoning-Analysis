import argparse
import json
import os
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import mean_squared_error, mean_absolute_error

def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Train a regression model to predict manually defined 'counts' bins.")
    parser.add_argument('--model_path', type=str, default='/home/yexuming/model/hg/bert-base-uncased', help='Path or name of the pretrained base model.')
    parser.add_argument('--data_path', type=str, default='data/merged_with_labels_and_counts.jsonl', help='Path to the .jsonl dataset.')
    parser.add_argument('--output_dir', type=str, default='predictor_regressor_model_manual_10_bins', help='Directory to save the trained model.')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=4, help='Per-device training and evaluation batch size.')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate.')
    parser.add_argument('--gpu_ids', type=str, default='0', help='Comma-separated list of GPU IDs to use.')
    args = parser.parse_args()

    # Set CUDA_VISIBLE_DEVICES
    if args.gpu_ids:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids

    # --- 1. Load Data and Create Bins ---
    print(f"Loading dataset from {args.data_path}...")
    df = pd.read_json(args.data_path, lines=True)

    # Define manual bin edges
    bin_edges = [0, 1, 8, 16, 25, 35, 56, 81, 140, 239, 3000]
    num_bins = len(bin_edges) - 1
    labels = list(range(num_bins))
    print(f"Using manually defined bins. Creating {num_bins} bins...")

    # Create bins using manual edges
    df['label'] = pd.cut(df['counts'], bins=bin_edges, labels=labels, right=True, include_lowest=True)

    # Remove samples where label could not be assigned (if any)
    df.dropna(subset=['label'], inplace=True)
    df['label'] = df['label'].astype(int)

    # Print bin information for clarity
    print("\n--- Original 'counts' Ranges and Sample Counts for Each Bin ---")
    for i in labels:
        bin_data = df[df['label'] == i]
        min_val = bin_data['counts'].min()
        max_val = bin_data['counts'].max()
        count = len(bin_data)
        print(f"Bin {i}: `counts` from {min_val} to {max_val} ({count} samples)")
    print("------------------------------------------------------------\n")

    # Combine text fields for model input
    df['text'] = (df['instruction'].fillna('') + ' ' + df['input'].fillna(''))

    # Convert labels to float for regression
    df['label'] = df['label'].astype(float)

    # Convert back to Hugging Face Dataset
    raw_dataset = Dataset.from_pandas(df[['text', 'label']])

    # Split the dataset
    dataset_split = raw_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset_split['train']
    eval_dataset = dataset_split['test']

    # --- 2. Load Tokenizer ---
    print(f"Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'}) # Corrected: escaped backslash for newline

    # --- 3. Preprocess Data ---
    def preprocess_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)

    print("Tokenizing datasets...")
    tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
    tokenized_eval_dataset = eval_dataset.map(preprocess_function, batched=True)

    # --- 4. Load Model for Regression ---
    print(f"Loading model from {args.model_path} for regression...")
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=1)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    # --- 5. Define Regression Metrics ---
    def compute_metrics(p):
        preds = p.predictions.flatten()
        labels = p.label_ids.flatten()
        mse = mean_squared_error(labels, preds)
        mae = mean_absolute_error(labels, preds)
        return {'mse': mse, 'mae': mae}

    # --- 6. Define Training Arguments ---
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f'{args.output_dir}/logs',
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        load_best_model_at_end=True,
        metric_for_best_model="mae",
        greater_is_better=False,
        save_total_limit=2,
        report_to="none",
    )

    # --- 7. Initialize and Train ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        compute_metrics=compute_metrics,
    )
    print("Starting regression training...")
    trainer.train()
    print("Training finished.")

    # --- 8. Evaluate and Save ---
    print("Evaluating final model...")
    eval_results = trainer.evaluate()
    print(f"Final evaluation results: {eval_results}")
    trainer.save_model(args.output_dir)
    print(f"Regression model saved successfully to {args.output_dir}")

if __name__ == "__main__":
    main()
