import argparse
import json
import os
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

# Custom Trainer with class weights
class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        logits = outputs.get("logits")
        labels = inputs.get("labels")

        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(model.device))
        weighted_loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (weighted_loss, outputs) if return_outputs else weighted_loss


def extract_enhanced_features(row):
    """Extract features for text enhancement"""
    instruction = row['instruction'] if pd.notna(row['instruction']) else ''
    input_text = row['input'] if pd.notna(row['input']) else ''
    combined_text = instruction + ' ' + input_text

    word_count = len(combined_text.split())

    # Task type detection
    has_explain = any(word in combined_text.lower() for word in ['explain', 'describe', 'analyze', 'compare', 'evaluate', 'discuss'])
    has_creative = any(word in combined_text.lower() for word in ['write', 'create', 'generate', 'compose', 'design', 'imagine', 'story'])
    has_calculate = any(word in combined_text.lower() for word in ['calculate', 'compute', 'solve', 'find', 'determine'])
    has_list = any(word in combined_text.lower() for word in ['list', 'enumerate', 'identify', 'name'])
    has_why = 'why' in combined_text.lower()
    has_how = 'how' in combined_text.lower()

    # Build feature tokens
    feature_tokens = []

    if word_count < 10:
        feature_tokens.append('[SHORT]')
    elif word_count < 25:
        feature_tokens.append('[MEDIUM]')
    else:
        feature_tokens.append('[LONG]')

    if has_explain:
        feature_tokens.append('[EXPLAIN]')
    if has_creative:
        feature_tokens.append('[CREATIVE]')
    if has_calculate:
        feature_tokens.append('[CALCULATE]')
    if has_list:
        feature_tokens.append('[LIST]')
    if has_why:
        feature_tokens.append('[WHY]')
    if has_how:
        feature_tokens.append('[HOW]')

    feature_prefix = ' '.join(feature_tokens)
    enhanced_text = f"{feature_prefix} {combined_text}" if feature_tokens else combined_text

    return enhanced_text


def main():
    parser = argparse.ArgumentParser(description="Train with cluster-based labels (clean dataset)")
    parser.add_argument('--model_path', type=str, default='/home/yexuming/model/hg/deberta-v3-large',
                        help='Path to pretrained model')
    parser.add_argument('--data_path', type=str, default='data/merged_with_clusters_clean.jsonl',
                        help='Path to clean clustered dataset')
    parser.add_argument('--output_dir', type=str, default='predictor_deberta_clusters_clean',
                        help='Output directory')
    parser.add_argument('--epochs', type=int, default=5, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size per device')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--gpu_ids', type=str, default='1', help='GPU IDs')
    parser.add_argument('--max_length', type=int, default=512, help='Max sequence length')
    parser.add_argument('--warmup_steps', type=int, default=500, help='Warmup steps')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2, help='Gradient accumulation')
    parser.add_argument('--early_stopping_patience', type=int, default=5, help='Early stopping patience')
    args = parser.parse_args()

    if args.gpu_ids:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
        print(f"Using GPU(s): {args.gpu_ids}")

    # Load clean clustered data
    print(f"Loading clustered dataset from {args.data_path}...")
    df = pd.read_json(args.data_path, lines=True)
    print(f"Loaded {len(df)} samples")

    num_labels = int(df['label'].max() + 1)
    print(f"Number of labels: {num_labels}")

    # Show distribution
    print("\n--- Label Distribution ---")
    for label in sorted(df['label'].unique()):
        count = len(df[df['label'] == label])
        pct = count / len(df) * 100
        mean_counts = df[df['label'] == label]['counts'].mean()
        print(f"Label {label}: {count:5d} samples ({pct:5.2f}%) - avg counts: {mean_counts:.1f}")
    print("-------------------------\n")

    # Calculate class weights
    print("Calculating class weights...")
    class_weights_array = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(df['label']),
        y=df['label'].to_numpy()
    )
    class_weights_tensor = torch.tensor(class_weights_array, dtype=torch.float)
    print("\n--- Class Weights ---")
    for i, weight in enumerate(class_weights_tensor):
        print(f"Label {i}: Weight {weight:.2f}")
    print("---------------------\n")

    # Apply feature engineering
    print("Applying feature engineering...")
    df['enhanced_text'] = df.apply(extract_enhanced_features, axis=1)

    # Prepare dataset
    raw_dataset = Dataset.from_pandas(df[['enhanced_text', 'label']])
    from datasets import ClassLabel
    raw_dataset = raw_dataset.cast_column("label", ClassLabel(num_classes=num_labels))
    dataset_split = raw_dataset.train_test_split(test_size=0.1, seed=42, stratify_by_column='label')
    train_dataset = dataset_split['train']
    eval_dataset = dataset_split['test']

    print(f"Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")

    # Tokenize
    print(f"Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def preprocess_function(examples):
        return tokenizer(examples['enhanced_text'], padding='max_length', truncation=True, max_length=args.max_length)

    print("Tokenizing datasets...")
    tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=['enhanced_text'])
    tokenized_eval_dataset = eval_dataset.map(preprocess_function, batched=True, remove_columns=['enhanced_text'])

    # Load model
    print(f"Loading model from {args.model_path}...")
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=num_labels)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Metrics
    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        labels = p.label_ids
        f1 = f1_score(labels, preds, average='macro')
        acc = accuracy_score(labels, preds)

        per_class_f1 = f1_score(labels, preds, average=None)
        metrics = {'accuracy': acc, 'f1_macro': f1}
        for i, f1_val in enumerate(per_class_f1):
            metrics[f'f1_label_{i}'] = f1_val

        return metrics

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        weight_decay=0.01,
        logging_dir=f'{args.output_dir}/logs',
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=3,
        report_to="none",
        fp16=True,
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        compute_metrics=compute_metrics,
        class_weights=class_weights_tensor
    )

    # Train
    print("\n" + "="*60)
    print("Starting training with cluster-based labels...")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print("="*60 + "\n")

    trainer.train()
    print("Training finished.")

    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    eval_results = trainer.evaluate()

    for key, value in sorted(eval_results.items()):
        if 'f1_label' in key:
            label_num = key.split('_')[-1]
            print(f"  Label {label_num} F1: {value:.4f}")
        else:
            print(f"  {key}: {value:.4f}")
    print("="*60 + "\n")

    # Save
    results_file = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(results_file, 'w') as f:
        json.dump(eval_results, f, indent=2)

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
