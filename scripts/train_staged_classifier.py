
import argparse
import json
import os
import pandas as pd
from datasets import Dataset, ClassLabel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Train a two-stage classification model.")
    parser.add_argument('--model_path', type=str, default='/home/yexuming/model/hg/bert-base-uncased', help='Path to the base model.')
    parser.add_argument('--data_path', type=str, default='analysis_results/qwen2_14B_analysis/results_14B_with_counts.jsonl', help='Path to the dataset.')
    parser.add_argument('--output_dir', type=str, default='predictor_staged_classifier_model', help='Directory to save the final model and intermediate checkpoints.')
    parser.add_argument('--num_bins', type=int, default=10, help='Number of bins for classification.')
    parser.add_argument('--total_epochs', type=int, default=4, help='Total number of training epochs for both stages.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training and evaluation.')
    args = parser.parse_args()

    # --- 1. Load Data and Create Bins (Equal Frequency) ---
    print(f"Loading dataset from {args.data_path}...")
    df = pd.read_json(args.data_path, lines=True)

    print(f"Creating {args.num_bins} equal frequency bins for 'counts'...")
    try:
        df['label'], _ = pd.qcut(df['counts'], q=args.num_bins, labels=False, retbins=True, duplicates='drop')
        actual_num_bins = df['label'].nunique()
        if actual_num_bins != args.num_bins:
            print(f"Warning: Due to duplicate values, only {actual_num_bins} unique bins were created.")
            args.num_bins = actual_num_bins
    except ValueError as e:
        print(f"Error creating bins: {e}")
        return

    df['text'] = (df['instruction'].fillna('') + ' ' + df['input'].fillna(''))
    raw_dataset = Dataset.from_pandas(df[['text', 'label']])
    raw_dataset = raw_dataset.cast_column("label", ClassLabel(num_classes=args.num_bins))
    dataset_split = raw_dataset.train_test_split(test_size=0.1, seed=42, stratify_by_column='label')
    train_dataset = dataset_split['train']
    eval_dataset = dataset_split['test']

    # --- 2. Tokenization ---
    print(f"Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    def preprocess_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)
    print("Tokenizing datasets...")
    tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
    tokenized_eval_dataset = eval_dataset.map(preprocess_function, batched=True)

    # --- 3. Model and Metrics Setup ---
    print(f"Loading model from {args.model_path}...")
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=args.num_bins)
    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids, preds, average='macro', zero_division=0)
        acc = accuracy_score(p.label_ids, preds)
        return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

    # --- STAGE 1: Train Classification Head Only ---
    print("\n--- STAGE 1: Starting training for classification head only ---")
    # Freeze the base model layers
    for param in model.bert.parameters():
        param.requires_grad = False

    stage1_epochs = args.total_epochs / 2
    stage1_output_dir = os.path.join(args.output_dir, 'stage1')

    training_args_stage1 = TrainingArguments(
        output_dir=stage1_output_dir,
        num_train_epochs=stage1_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=5e-4, # Higher learning rate for the new head
        warmup_steps=200,
        weight_decay=0.01,
        logging_dir=f'{stage1_output_dir}/logs',
        logging_steps=100,
        report_to="none",
    )

    trainer_stage1 = Trainer(
        model=model,
        args=training_args_stage1,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        compute_metrics=compute_metrics,
    )
    trainer_stage1.train()
    print("--- STAGE 1: Finished ---")

    # --- STAGE 2: Fine-tune Entire Model ---
    print("\n--- STAGE 2: Starting fine-tuning for the entire model ---")
    # Unfreeze the base model layers
    for param in model.bert.parameters():
        param.requires_grad = True

    stage2_epochs = args.total_epochs / 2
    stage2_output_dir = os.path.join(args.output_dir, 'stage2')

    training_args_stage2 = TrainingArguments(
        output_dir=stage2_output_dir,
        num_train_epochs=stage2_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=2e-5,  # Lower learning rate for full fine-tuning
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f'{stage2_output_dir}/logs',
        logging_steps=100,
        report_to="none",
    )

    # We re-initialize the Trainer. This is crucial as it will create a new optimizer
    # that includes the now-unfrozen BERT parameters.
    trainer_stage2 = Trainer(
        model=model,
        args=training_args_stage2,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        compute_metrics=compute_metrics,
    )
    trainer_stage2.train()
    print("--- STAGE 2: Finished ---")

    # --- Final Evaluation and Save ---
    print("\n--- Final Evaluation ---")
    eval_results = trainer_stage2.evaluate()
    print(f"Final evaluation results: {eval_results}")

    final_model_path = os.path.join(args.output_dir, 'final')
    trainer_stage2.save_model(final_model_path)
    print(f"Two-stage training complete. Final model saved to {final_model_path}")

if __name__ == "__main__":
    main()
