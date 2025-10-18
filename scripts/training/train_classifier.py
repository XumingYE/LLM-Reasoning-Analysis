
import argparse
import json
import os
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Train a classification model to predict 'counts' bins.")
    parser.add_argument('--model_path', type=str, default='/home/yexuming/model/hg/bert-base-uncased', help='Path or name of the pretrained base model.')
    parser.add_argument('--data_path', type=str, default='analysis_results/qwen2_14B_analysis/results_14B_with_counts.jsonl', help='Path to the .jsonl dataset.')
    parser.add_argument('--output_dir', type=str, default='predictor_classifier_model', help='Directory to save the trained model.')
    parser.add_argument('--num_bins', type=int, default=10, help='Number of bins to classify counts into.')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=8, help='Per-device training and evaluation batch size.')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate.')
    args = parser.parse_args()

    # --- 1. Load Data and Create Bins ---
    print(f"Loading dataset from {args.data_path}...")
    df = pd.read_json(args.data_path, lines=True)

    # Create bins using quantile-based method
    print(f"Creating {args.num_bins} bins for 'counts'...")
    try:
        # Use pd.cut to create bins with equal width ranges
        df['label'], bin_edges = pd.cut(df['counts'], bins=args.num_bins, labels=False, retbins=True, include_lowest=True)
    except ValueError as e:
        print(f"Error: Could not create bins. {e}. Try a smaller number of bins.")
        return

    # Print bin information for clarity
    print("\n--- Count Ranges for Each Class ---")
    for i in range(args.num_bins):
        min_val = df[df['label'] == i]['counts'].min()
        max_val = df[df['label'] == i]['counts'].max()
        print(f"Class {i}: counts from {min_val} to {max_val}")
    print("-----------------------------------\n")

    # Combine text fields for model input
    df['text'] = (df['instruction'].fillna('') + ' ' + df['input'].fillna(''))

    # Convert back to Hugging Face Dataset
    raw_dataset = Dataset.from_pandas(df[['text', 'label']])
    # Cast the label column to a ClassLabel type for stratification
    from datasets import ClassLabel
    raw_dataset = raw_dataset.cast_column("label", ClassLabel(num_classes=args.num_bins))

    # Stratify split to maintain class distribution in train and test sets
    dataset_split = raw_dataset.train_test_split(test_size=0.1, seed=42, stratify_by_column='label')
    train_dataset = dataset_split['train']
    eval_dataset = dataset_split['test']

    # --- 2. Load Tokenizer ---
    print(f"Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # --- 3. Preprocess Data ---
    def preprocess_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)

    print("Tokenizing datasets...")
    tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
    tokenized_eval_dataset = eval_dataset.map(preprocess_function, batched=True)

    # --- 4. Load Model for Classification ---
    print(f"Loading model from {args.model_path} for classification with {args.num_bins} labels...")
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=args.num_bins)

    # --- 5. Define Classification Metrics ---
    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids, preds, average='macro')
        acc = accuracy_score(p.label_ids, preds)
        return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

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
    print("Starting classification training...")
    trainer.train()
    print("Training finished.")

    # --- 8. Evaluate and Save ---
    print("Evaluating final model...")
    eval_results = trainer.evaluate()
    print(f"Final evaluation results: {eval_results}")
    trainer.save_model(args.output_dir)
    print(f"Classification model saved successfully to {args.output_dir}")

if __name__ == "__main__":
    main()
