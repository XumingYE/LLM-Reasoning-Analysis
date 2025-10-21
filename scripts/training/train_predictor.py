
import argparse
import json
import os
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def main():
    # --- Argument Parsing ---
    # This script is designed to be future-proof.
    # It defaults to a small model (bert-base-uncased) that works in low-memory environments.
    # For high-memory environments, you can pass the path to a larger model like DeepSeek.
    parser = argparse.ArgumentParser(description="Train a regression model to predict 'counts'.")
    parser.add_argument('--model_path', type=str, default='bert-base-uncased', help='Path or name of the pretrained base model.')
    parser.add_argument('--data_path', type=str, default='analysis_results/qwen2_14B_analysis/results_14B_with_counts.jsonl', help='Path to the .jsonl dataset.')
    parser.add_argument('--output_dir', type=str, default='predictor_model', help='Directory to save the trained model.')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=8, help='Per-device training and evaluation batch size. Reduce for larger models if memory is an issue.')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate.')

    args = parser.parse_args()

    # --- 1. Load and Prepare Dataset ---
    print(f"Loading and preparing dataset from {args.data_path}...")
    
    data_list = []
    with open(args.data_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            text = (data.get('instruction') or '') + ' ' + (data.get('input') or '')
            label = float(data.get('counts', 0.0))
            data_list.append({'text': text, 'label': label})

    raw_dataset = Dataset.from_list(data_list)
    dataset_split = raw_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset_split['train']
    eval_dataset = dataset_split['test']
    print(f"Dataset prepared: {len(train_dataset)} training samples, {len(eval_dataset)} evaluation samples.")

    # --- 2. Load Tokenizer ---
    print(f"Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- 3. Preprocess Data ---
    def preprocess_function(examples):
        result = tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)
        result['labels'] = [float(label) for label in examples['label']]
        return result

    print("Tokenizing datasets...")
    tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
    tokenized_eval_dataset = eval_dataset.map(preprocess_function, batched=True)

    # --- 4. Load Model for Regression (using default float32 for compatibility) ---
    print(f"Loading model from {args.model_path} for regression...")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path, 
        num_labels=1,  # 1 for regression
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    # --- 5. Define Training Arguments (using a minimal, compatible set) ---
    print("Defining training arguments...")
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
        metric_for_best_model="mse",
        greater_is_better=False,
        save_total_limit=2,
        report_to="none",
    )

    # --- 6. Define Evaluation Metrics ---
    def compute_metrics(p):
        preds = p.predictions.flatten()
        labels = p.label_ids.flatten()
        mse = mean_squared_error(labels, preds)
        mae = mean_absolute_error(labels, preds)
        return {'mse': mse, 'mae': mae}

    # --- 7. Initialize Trainer ---
    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        compute_metrics=compute_metrics
    )

    # --- 8. Train ---
    print("Starting training...")
    trainer.train()
    print("Training finished.")

    # --- 9. Evaluate and Save Manually ---
    print("Evaluating final model...")
    eval_results = trainer.evaluate()
    print(f"Final evaluation results: {eval_results}")

    print(f"Saving the final model to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    print(f"Model saved successfully to {args.output_dir}")

if __name__ == "__main__":
    main()
