import argparse
import json
import os
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

from transformers.modeling_outputs import SequenceClassifierOutput

# Custom Trainer to handle class weights for imbalanced datasets
class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Let the model compute its own outputs, including the (unweighted) loss and logits
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Extract labels
        labels = inputs.get("labels")

        # Manually compute the weighted loss
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(model.device))
        weighted_loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        # Return our weighted loss instead of the model's original loss
        return (weighted_loss, outputs) if return_outputs else weighted_loss

def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Train a weighted classification model on manually defined 'counts' bins.")
    parser.add_argument('--model_path', type=str, default='/home/yexuming/model/hg/bert-base-uncased', help='Path or name of the pretrained base model.')
    parser.add_argument('--data_path', type=str, default='data/merged_with_labels_and_counts.jsonl', help='Path to the .jsonl dataset.')
    parser.add_argument('--output_dir', type=str, default='predictor_weighted_classifier_manual_10_bins', help='Directory to save the trained model.')
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

    bin_edges = [0, 1, 4, 8, 17, 28, 41, 66, 98, 200, 1000]
    num_bins = len(bin_edges) - 1
    labels = list(range(num_bins))
    print(f"Using manually defined bins. Creating {num_bins} bins...")

    df['label'] = pd.cut(df['counts'], bins=bin_edges, labels=labels, right=True, include_lowest=True)
    df.dropna(subset=['label'], inplace=True)
    df['label'] = df['label'].astype(int)

    print("\n--- Sample Counts for Each Bin ---")
    print(df['label'].value_counts().sort_index())
    print("------------------------------------")

    # --- 2. Calculate Class Weights ---
    print("Calculating class weights for handling imbalance...")
    class_weights_array = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(df['label']),
        y=df['label'].to_numpy()
    )
    class_weights_tensor = torch.tensor(class_weights_array, dtype=torch.float)
    print("\n--- Calculated Class Weights ---")
    for i, weight in enumerate(class_weights_tensor):
        print(f"Bin {i}: Weight {weight:.2f}")
    print("------------------------------\n")

    # --- 3. Prepare Dataset ---
    df['text'] = (df['instruction'].fillna('') + ' ' + df['input'].fillna(''))
    raw_dataset = Dataset.from_pandas(df[['text', 'label']])
    from datasets import ClassLabel
    raw_dataset = raw_dataset.cast_column("label", ClassLabel(num_classes=num_bins))
    dataset_split = raw_dataset.train_test_split(test_size=0.1, seed=42, stratify_by_column='label')
    train_dataset = dataset_split['train']
    eval_dataset = dataset_split['test']

    # --- 4. Tokenize Data ---
    print(f"Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'}) # Corrected: escaped the quote within the dictionary

    def preprocess_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)

    print("Tokenizing datasets...")
    tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=['text'])
    tokenized_eval_dataset = eval_dataset.map(preprocess_function, batched=True, remove_columns=['text'])

    # --- 5. Load Model ---
    print(f"Loading model from {args.model_path} for classification with {num_bins} labels...")
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=num_bins)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    # --- 6. Define Metrics and Trainer ---
    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        labels = p.label_ids
        f1 = f1_score(labels, preds, average='macro')
        acc = accuracy_score(labels, preds)
        return {'accuracy': acc, 'f1_macro': f1}

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
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        compute_metrics=compute_metrics,
        class_weights=class_weights_tensor
    )

    # --- 7. Train ---
    print("Starting weighted classification training...")
    trainer.train()
    print("Training finished.")

    # --- 8. Evaluate and Save ---
    print("Evaluating final model...")
    eval_results = trainer.evaluate()
    print(f"Final evaluation results: {eval_results}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Weighted classification model saved successfully to {args.output_dir}")

if __name__ == "__main__":
    main()
