
import pandas as pd
import numpy as np
import torch
import argparse
import json
import os
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset

def compute_metrics(eval_pred):
    """Computes MSE and MAE on the original scale."""
    predictions, labels = eval_pred
    # Inverse transform from log scale
    predictions = np.expm1(predictions.flatten())
    labels = np.expm1(labels.flatten())
    
    mse = np.mean((predictions - labels) ** 2)
    mae = np.mean(np.abs(predictions - labels))
    
    return {
        'mse': mse,
        'mae': mae,
    }

def train_regressor(args):
    # --- 1. Load and Prepare Data ---
    print(f"Loading data from {args.data_path}...")
    df = pd.read_json(args.data_path, lines=True)

    # Create input text
    df['text'] = df['instruction'].fillna('') + ' ' + df['input'].fillna('')

    # Apply log transformation to the target variable
    df['label'] = np.log1p(df['counts'])
    
    print(f"Loaded {len(df)} records.")

    # --- 2. Train/Test Split ---
    train_df, eval_df = train_test_split(df, test_size=0.1, random_state=42)
    print(f"Training set size: {len(train_df)}")
    print(f"Evaluation set size: {len(eval_df)}")

    # Convert to Hugging Face Dataset
    train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
    eval_dataset = Dataset.from_pandas(eval_df[['text', 'label']])

    # --- 3. Tokenization ---
    print(f"Loading tokenizer for {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    eval_dataset = eval_dataset.map(tokenize_function, batched=True)

    # --- 4. Model Loading ---
    print(f"Loading model {args.model_path} for regression...")
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=1)

    # --- 5. Training ---
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f'{args.output_dir}/logs',
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="mae",
        greater_is_better=False,
        report_to="none", # Disable wandb/tensorboard reporting
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    # --- 6. Evaluation ---
    print("Evaluating final model on the test set...")
    eval_results = trainer.evaluate()

    print("\n--- Final Evaluation Results ---")
    print(eval_results)

    # Save results to a file
    results_path = os.path.join(args.output_dir, "final_evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(eval_results, f, indent=4)
    
    print(f"\nTraining complete. Model saved to {args.output_dir}")
    print(f"Evaluation results saved to {results_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a DeBERTa regression model on log-transformed counts.")
    
    parser.add_argument("--data_path", type=str, default="data/cleaned_data_wait_only.jsonl", help="Path to the cleaned data file.")
    parser.add_argument("--model_path", type=str, default="microsoft/deberta-v3-base", help="Path or name of the base DeBERTa model.")
    parser.add_argument("--output_dir", type=str, default="predictor_deberta_regressor_wait_only", help="Directory to save the trained model and results.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Training and evaluation batch size.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate.")

    args = parser.parse_args()
    train_regressor(args)
