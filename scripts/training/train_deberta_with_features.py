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
import re

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


def extract_enhanced_features(row):
    """
    Extract features from instruction and input to enhance the text representation.
    """
    instruction = row['instruction'] if pd.notna(row['instruction']) else ''
    input_text = row['input'] if pd.notna(row['input']) else ''
    combined_text = instruction + ' ' + input_text

    # Basic length features
    word_count = len(combined_text.split())
    char_count = len(combined_text)
    avg_word_length = char_count / max(word_count, 1)

    # Question complexity indicators
    question_count = combined_text.count('?')
    has_multiple_questions = question_count > 1

    # Task type detection (these often correlate with reasoning depth)
    explain_keywords = ['explain', 'describe', 'analyze', 'compare', 'evaluate', 'discuss', 'elaborate']
    creative_keywords = ['write', 'create', 'generate', 'compose', 'design', 'imagine', 'story']
    calculate_keywords = ['calculate', 'compute', 'solve', 'find', 'determine']
    list_keywords = ['list', 'enumerate', 'identify', 'name']

    has_explain = any(word in combined_text.lower() for word in explain_keywords)
    has_creative = any(word in combined_text.lower() for word in creative_keywords)
    has_calculate = any(word in combined_text.lower() for word in calculate_keywords)
    has_list = any(word in combined_text.lower() for word in list_keywords)

    # Complexity indicators
    has_comparison = any(word in combined_text.lower() for word in ['compare', 'contrast', 'difference', 'versus', 'vs'])
    has_conditional = any(word in combined_text.lower() for word in ['if', 'when', 'suppose', 'assume'])
    has_why = 'why' in combined_text.lower()
    has_how = 'how' in combined_text.lower()

    # Count of complex punctuation
    semicolon_count = combined_text.count(';')
    colon_count = combined_text.count(':')

    # Build feature string as special tokens
    feature_tokens = []

    # Length category
    if word_count < 10:
        feature_tokens.append('[SHORT]')
    elif word_count < 25:
        feature_tokens.append('[MEDIUM]')
    else:
        feature_tokens.append('[LONG]')

    # Task type
    if has_explain:
        feature_tokens.append('[EXPLAIN]')
    if has_creative:
        feature_tokens.append('[CREATIVE]')
    if has_calculate:
        feature_tokens.append('[CALCULATE]')
    if has_list:
        feature_tokens.append('[LIST]')

    # Complexity markers
    if has_comparison:
        feature_tokens.append('[COMPARE]')
    if has_conditional:
        feature_tokens.append('[CONDITIONAL]')
    if has_multiple_questions:
        feature_tokens.append('[MULTI_Q]')
    if has_why:
        feature_tokens.append('[WHY]')
    if has_how:
        feature_tokens.append('[HOW]')

    # Combine original text with feature tokens
    # Put features at the beginning so the model sees them first
    feature_prefix = ' '.join(feature_tokens)
    enhanced_text = f"{feature_prefix} {combined_text}" if feature_tokens else combined_text

    return enhanced_text


def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Train DeBERTa weighted classifier with feature engineering.")
    parser.add_argument('--model_path', type=str, default='/home/yexuming/model/hg/deberta-v3-large',
                        help='Path or name of the pretrained base model.')
    parser.add_argument('--data_path', type=str, default='data/merged_with_labels_and_counts.jsonl',
                        help='Path to the .jsonl dataset.')
    parser.add_argument('--output_dir', type=str, default='predictor_deberta_large_with_features',
                        help='Directory to save the trained model.')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=2, help='Per-device training and evaluation batch size.')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help='Learning rate (lower for large models).')
    parser.add_argument('--gpu_ids', type=str, default='1', help='Comma-separated list of GPU IDs to use.')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length.')
    parser.add_argument('--warmup_steps', type=int, default=500, help='Number of warmup steps.')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                        help='Gradient accumulation steps to simulate larger batch size.')
    args = parser.parse_args()

    # Set CUDA_VISIBLE_DEVICES
    if args.gpu_ids:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
        print(f"Using GPU(s): {args.gpu_ids}")

    # --- 1. Load Data and Create Bins ---
    print(f"Loading dataset from {args.data_path}...")
    df = pd.read_json(args.data_path, lines=True)
    print(f"Loaded {len(df)} samples")

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

    # --- 3. Apply Feature Engineering ---
    print("Applying feature engineering...")
    df['enhanced_text'] = df.apply(extract_enhanced_features, axis=1)

    # Show some examples
    print("\n--- Feature Engineering Examples ---")
    for i in range(min(3, len(df))):
        print(f"\nExample {i+1}:")
        print(f"Original: {df.iloc[i]['instruction'][:100]}...")
        print(f"Enhanced: {df.iloc[i]['enhanced_text'][:150]}...")
    print("------------------------------------\n")

    # --- 4. Prepare Dataset ---
    raw_dataset = Dataset.from_pandas(df[['enhanced_text', 'label']])
    from datasets import ClassLabel
    raw_dataset = raw_dataset.cast_column("label", ClassLabel(num_classes=num_bins))
    dataset_split = raw_dataset.train_test_split(test_size=0.1, seed=42, stratify_by_column='label')
    train_dataset = dataset_split['train']
    eval_dataset = dataset_split['test']

    print(f"Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")

    # --- 5. Tokenize Data ---
    print(f"Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # DeBERTa should already have a pad token, but let's verify
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set pad_token to eos_token")

    def preprocess_function(examples):
        return tokenizer(examples['enhanced_text'], padding='max_length', truncation=True, max_length=args.max_length)

    print("Tokenizing datasets...")
    tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=['enhanced_text'])
    tokenized_eval_dataset = eval_dataset.map(preprocess_function, batched=True, remove_columns=['enhanced_text'])

    # --- 6. Load Model ---
    print(f"Loading DeBERTa-v3-large model from {args.model_path} for classification with {num_bins} labels...")
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=num_bins)

    # Resize token embeddings if we added new tokens (we didn't in this case, but good practice)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    print(f"Model loaded. Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # --- 7. Define Metrics and Trainer ---
    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        labels = p.label_ids
        f1 = f1_score(labels, preds, average='macro')
        acc = accuracy_score(labels, preds)

        # Also compute per-class F1 for detailed analysis
        per_class_f1 = f1_score(labels, preds, average=None)

        metrics = {'accuracy': acc, 'f1_macro': f1}
        for i, f1_val in enumerate(per_class_f1):
            metrics[f'f1_bin_{i}'] = f1_val

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
        eval_steps=500,  # More frequent evaluation
        save_strategy="steps",
        save_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=3,  # Keep more checkpoints for large model
        report_to="none",
        fp16=True,  # Use mixed precision for faster training on GPU
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        compute_metrics=compute_metrics,
        class_weights=class_weights_tensor
    )

    # --- 8. Train ---
    print("Starting weighted classification training with DeBERTa-v3-large...")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    trainer.train()
    print("Training finished.")

    # --- 9. Evaluate and Save ---
    print("Evaluating final model...")
    eval_results = trainer.evaluate()
    print("\n" + "="*50)
    print("FINAL EVALUATION RESULTS")
    print("="*50)
    for key, value in eval_results.items():
        if 'f1_bin' in key:
            bin_num = key.split('_')[-1]
            print(f"  Bin {bin_num} F1: {value:.4f}")
        else:
            print(f"  {key}: {value:.4f}")
    print("="*50 + "\n")

    # Save evaluation results to file
    results_file = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(results_file, 'w') as f:
        json.dump(eval_results, f, indent=2)
    print(f"Evaluation results saved to {results_file}")

    print(f"Saving model to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"DeBERTa model with feature engineering saved successfully to {args.output_dir}")

if __name__ == "__main__":
    main()
