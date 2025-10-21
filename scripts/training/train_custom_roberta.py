import argparse
import json
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset, ClassLabel
from transformers import AutoTokenizer, RobertaModel, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

# 1. Custom Model with a more powerful classification head
class CustomRobertaForSequenceClassification(nn.Module):
    def __init__(self, model_path, num_labels):
        super().__init__()
        self.num_labels = num_labels
        self.roberta = RobertaModel.from_pretrained(model_path)
        
        # Freeze most of the layers to save resources and prevent catastrophic forgetting
        for param in self.roberta.parameters():
            param.requires_grad = False
        # Unfreeze the top 2 layers for fine-tuning
        for param in self.roberta.encoder.layer[-2:].parameters():
            param.requires_grad = True

        # Define a more robust classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.roberta.config.hidden_size, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, self.num_labels)
        )

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        # Use the [CLS] token's representation for classification
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)

        loss = None
        if labels is not None:
            # The loss will be computed outside in the WeightedTrainer
            pass

        return {"logits": logits, "loss": loss} # Return a dict to be compatible with Trainer

# 2. Custom Trainer for Weighted Loss
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
        device = next(model.parameters()).device
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(device))
        weighted_loss = loss_fct(logits.view(-1, self.model.num_labels), labels.view(-1))
        
        # Return our weighted loss instead of the model's original loss
        return (weighted_loss, outputs) if return_outputs else weighted_loss

def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Train a custom RoBERTa model with a weighted classification head.")
    parser.add_argument('--model_path', type=str, default='/home/yexuming/model/hg/roberta-large', help='Path to the pretrained RoBERTa model.')
    parser.add_argument('--data_path', type=str, default='data/merged_with_labels_and_counts.jsonl', help='Path to the .jsonl dataset.')
    parser.add_argument('--output_dir', type=str, default='custom_roberta_weighted_classifier', help='Directory to save the trained model.')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=4, help='Per-device training batch size.')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate for the classifier head.')
    parser.add_argument('--gpu_ids', type=str, default='0', help='Comma-separated list of GPU IDs to use.')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids

    # --- Data Preparation (Manual Binning) ---
    print("Loading and binning data...")
    df = pd.read_json(args.data_path, lines=True)
    bin_edges = [0, 1, 4, 8, 17, 28, 41, 66, 98, 200, 1000]
    num_labels = len(bin_edges) - 1
    df['label'] = pd.cut(df['counts'], bins=bin_edges, labels=range(num_labels), right=True, include_lowest=True).astype(int)
    df.dropna(subset=['label'], inplace=True)
    df['label'] = df['label'].astype(int)

    # --- Class Weights ---
    print("Calculating class weights...")
    class_weights = compute_class_weight('balanced', classes=np.unique(df['label']), y=df['label'].values)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

    # --- Dataset Creation and Tokenization ---
    df['text'] = df['instruction'].fillna('') + ' ' + df['input'].fillna('')
    raw_dataset = Dataset.from_pandas(df[['text', 'label']])
    raw_dataset = raw_dataset.cast_column("label", ClassLabel(num_classes=num_labels))
    dataset_split = raw_dataset.train_test_split(test_size=0.1, seed=42, stratify_by_column='label')
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    def preprocess_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)

    print("Tokenizing datasets...")
    tokenized_train_dataset = dataset_split['train'].map(preprocess_function, batched=True, remove_columns=['text'])
    tokenized_eval_dataset = dataset_split['test'].map(preprocess_function, batched=True, remove_columns=['text'])

    # --- Model Initialization ---
    print("Initializing custom RoBERTa model...")
    model = CustomRobertaForSequenceClassification(args.model_path, num_labels=num_labels)

    # --- Metrics and Training Arguments ---
    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        return {'f1_macro': f1_score(p.label_ids, preds, average='macro')}

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2, # Speed up evaluation
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

    # --- Trainer Initialization and Training ---
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        compute_metrics=compute_metrics,
        class_weights=class_weights_tensor
    )

    print("Starting training with custom RoBERTa model...")
    trainer.train()
    print("Training finished.")

    # --- Final Evaluation and Saving ---
    print("Evaluating final model...")
    eval_results = trainer.evaluate()
    print(f"Final evaluation results: {eval_results}")
    
    # To save the full model (including the RoBERTa base), we save the state_dict
    final_model_path = os.path.join(args.output_dir, 'final_model')
    os.makedirs(final_model_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(final_model_path, 'pytorch_model.bin'))
    tokenizer.save_pretrained(final_model_path)
    print(f"Custom RoBERTa model saved successfully to {final_model_path}")

if __name__ == "__main__":
    main()
