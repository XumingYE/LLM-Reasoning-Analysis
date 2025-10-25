
import pandas as pd
import numpy as np
import torch
import argparse
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset

# 1. Define Binning Strategy
def apply_manual_bins(counts):
    if counts == 1:
        return 0
    elif counts == 2:
        return 1
    elif counts == 3:
        return 2
    elif counts == 4:
        return 3
    elif 5 <= counts <= 8:
        return 4
    else: # counts > 8
        return 5

# 2. Custom Trainer for Weighted Loss
class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(self.args.device) if class_weights is not None else None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# 3. Metrics Calculation
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    f1 = f1_score(labels, predictions, average='macro')
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1_macro': f1,
    }

def train_classifier(args):
    # --- Load and Prepare Data ---
    print(f"Loading data from {args.data_path}...")
    df = pd.read_json(args.data_path, lines=True)

    df['text'] = df['instruction'].fillna('') + ' ' + df['input'].fillna('')
    df['label'] = df['counts'].apply(apply_manual_bins)
    num_labels = df['label'].nunique()
    
    print(f"Loaded {len(df)} records.")
    print(f"Number of classes: {num_labels}")
    print("Bin distribution:\n", df['label'].value_counts().sort_index())

    # --- Train/Test Split ---
    train_df, eval_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df['label'])
    train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
    eval_dataset = Dataset.from_pandas(eval_df[['text', 'label']])

    # --- Calculate Class Weights ---
    print("Calculating class weights for handling imbalance...")
    class_weights = compute_class_weight('balanced', classes=np.unique(train_df['label']), y=train_df['label'])
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
    print("Class weights:", class_weights)

    # --- Tokenization ---
    print(f"Loading tokenizer for {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    eval_dataset = eval_dataset.map(tokenize_function, batched=True)

    # --- Model Loading ---
    print(f"Loading model {args.model_path} for classification...")
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=num_labels)

    # --- Training ---
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
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        report_to="none",
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        class_weights=class_weights_tensor
    )

    print("Starting training...")
    trainer.train()

    # --- Evaluation ---
    print("Evaluating final model on the test set...")
    eval_results = trainer.evaluate()

    print("\n--- Final Evaluation Results ---")
    print(eval_results)

    results_path = os.path.join(args.output_dir, "final_evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(eval_results, f, indent=4)
    
    print(f"\nTraining complete. Model saved to {args.output_dir}")
    print(f"Evaluation results saved to {results_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a DeBERTa classifier with weighted loss.")
    
    parser.add_argument("--data_path", type=str, default="data/cleaned_data_wait_only.jsonl", help="Path to the cleaned data file.")
    parser.add_argument("--model_path", type=str, default="microsoft/deberta-v3-base", help="Path or name of the base DeBERTa model.")
    parser.add_argument("--output_dir", type=str, default="predictor_deberta_classifier_wait_only", help="Directory to save the trained model.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Training and evaluation batch size.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate.")

    args = parser.parse_args()
    train_classifier(args)
