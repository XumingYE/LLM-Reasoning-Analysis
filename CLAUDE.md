# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project analyzes the relationship between user queries, LLM-generated content, and Chain-of-Thought (CoT) complexity. The core metric is `counts`, which quantifies reasoning chain complexity by counting key transition keywords in the model's thinking process. The project trains predictive models to estimate reasoning complexity based on user questions.

## Data Format

The main dataset is in `data/merged_with_labels_and_counts.jsonl`, with entries structured as:
- `instruction`: The user's query
- `input`: Additional context (may be empty)
- `content`: Model's generated response including `<think>` tags
- `counts`: Complexity metric (number of reasoning transitions + 1)
- `label`: Class label (when using binned classification)

## Key Commands

### Training Models

**Weighted Classifier (Recommended):**
```bash
python scripts/training/train_weighted_classifier.py \
  --model_path /path/to/bert-base-uncased \
  --data_path data/merged_with_labels_and_counts.jsonl \
  --output_dir predictor_weighted_classifier_manual_10_bins \
  --epochs 3 \
  --batch_size 4 \
  --learning_rate 2e-5 \
  --gpu_ids 0
```

**Regression Model:**
```bash
python scripts/training/train_predictor.py \
  --model_path bert-base-uncased \
  --data_path data/merged_with_labels_and_counts.jsonl \
  --output_dir predictor_model \
  --epochs 3 \
  --batch_size 8
```

**Classifier with Quantile Binning:**
```bash
python scripts/training/train_classifier.py \
  --model_path /path/to/bert-base-uncased \
  --data_path data/merged_with_labels_and_counts.jsonl \
  --output_dir predictor_classifier_model \
  --num_bins 8 \
  --epochs 3 \
  --batch_size 4 \
  --gpu_ids 0
```

**Regressor on Bins:**
```bash
python scripts/training/train_regressor_on_bins.py --num_bins 8
```

### Running Inference

**Weighted Classifier Inference:**
```bash
python scripts/run_weighted_classifier_inference.py \
  --model_path predictor_weighted_classifier_manual_10_bins \
  --data_path data/merged_with_labels_and_counts.jsonl \
  --output_path weighted_classifier_inference_results.md \
  --num_samples 10
```

**Standard Model Inference:**
```bash
python scripts/run_inference.py \
  --model_path predictor_model \
  --data_path data/merged_with_labels_and_counts.jsonl \
  --output_path inference_results.md \
  --num_samples 10
```

### Data Processing

**Regenerate counts field:**
```bash
python scripts/data_processing/add_counts_field.py \
  data/merged_with_labels.jsonl \
  data/merged_with_labels_and_counts.jsonl
```

### Analysis Scripts

**Analyze counts distribution:**
```bash
python scripts/analysis/analyze_counts_distribution.py
```

**Analyze data patterns:**
```bash
python scripts/analysis/analyze_data.py
```

**Find high counts samples:**
```bash
python scripts/analysis/find_high_counts.py
```

**Find zero count samples:**
```bash
python scripts/analysis/get_zero_count_samples.py
```

## Architecture

### Model Approaches

The project has evolved through several modeling approaches:

1. **Regression Model** (`train_predictor.py`): Direct prediction of `counts` values using MSE loss. Baseline approach with ~899 MSE.

2. **Quantile Binning Classifier** (`train_classifier.py`): Converts regression to classification using `pd.qcut` for equal-frequency bins. Better for identifying different complexity levels across the distribution.

3. **Regressor on Bins** (`train_regressor_on_bins.py`): Predicts bin indices as continuous values, allowing the loss function to penalize predictions farther from true bins. Achieved ~1.13 MAE.

4. **Weighted Classifier** (`train_weighted_classifier.py`) - **RECOMMENDED**: Classification with class imbalance handling via weighted loss. Uses manually-defined bins optimized for the data distribution.

### Binning Strategy

The **manual 10-bin strategy** is the recommended approach:

```python
bin_edges = [0, 1, 4, 8, 17, 28, 41, 66, 98, 200, 1000]
```

- Bins 0-2: Fine-grained for simple queries (counts 1-8)
- Bins 3-6: Medium complexity queries (counts 9-66)
- Bins 7-9: High complexity reasoning (counts 67+)

This binning addresses:
- **Data skew**: ~40% of samples have counts=1
- **Long tail**: Some samples have counts >900
- **Semantic meaning**: Each bin represents a distinct complexity tier

### Custom Trainer

The `WeightedTrainer` class (in `train_weighted_classifier.py`) handles class imbalance:
- Inherits from HuggingFace `Trainer`
- Overrides `compute_loss` to apply class weights
- Uses `sklearn.utils.class_weight.compute_class_weight` with `'balanced'` strategy
- Applies weights via `torch.nn.CrossEntropyLoss(weight=class_weights)`

Class weights inversely correlate with sample frequency, ensuring the model learns from rare complex examples.

### Counts Calculation

The `counts` field is calculated by `scripts/data_processing/add_counts_field.py`:
- Extracts content before `</think>` tag
- Counts occurrences of transition keywords (case-sensitive)
- Keywords include: 'Wait', 'Alternatively', 'But wait', 'Hmm', 'Let me', 'Okay', etc.
- Formula: `counts = num_keywords + 1`
- **Important**: Case-sensitive matching ensures proper semantic meaning

### Model Evaluation

Primary metrics:
- **Classification**: F1-macro score (emphasizes balanced performance across all bins)
- **Regression**: MSE and MAE

The F1-macro metric is critical because accuracy can be misleading with imbalanced data (e.g., 84.9% accuracy with only 0.13 F1-macro when bins are poorly distributed).

## Directory Structure

```
├── data/                          # Training data
│   ├── merged_with_labels_and_counts.jsonl  # Main dataset
│   └── ...
├── scripts/
│   ├── training/                  # Model training scripts
│   │   ├── train_weighted_classifier.py     # Recommended
│   │   ├── train_classifier.py
│   │   ├── train_predictor.py
│   │   ├── train_regressor_on_bins.py
│   │   ├── train_custom_roberta.py
│   │   └── train_staged_classifier.py
│   ├── data_processing/           # Data preparation
│   │   └── add_counts_field.py
│   ├── analysis/                  # Analysis tools
│   │   ├── analyze_counts_distribution.py
│   │   ├── analyze_data.py
│   │   ├── find_high_counts.py
│   │   └── get_zero_count_samples.py
│   ├── run_inference.py
│   └── run_weighted_classifier_inference.py
├── predictor_weighted_classifier_manual_10_bins/  # Best model
├── predictor_model/               # Regression baseline
├── predictor_classifier_model/    # Quantile classifier
└── custom_roberta_weighted_classifier/  # RoBERTa variant
```

## Model Selection Guide

**For best results**: Use the weighted classifier with manual 10-bin strategy. It:
- Handles severe class imbalance (40% of samples in bin 0)
- Learns to identify rare but important high-complexity queries
- Uses F1-macro for balanced evaluation across all complexity tiers
- Provides interpretable bin predictions

**For baseline comparison**: Use the regression model for direct MSE/MAE metrics.

**For research**: Experiment with different binning strategies using `train_classifier.py` or `train_regressor_on_bins.py`.

## Important Notes

### GPU Configuration
- Use `--gpu_ids` parameter to specify GPU devices (e.g., `--gpu_ids 0,1`)
- Default batch size is 4; adjust based on available VRAM
- Models are based on BERT/RoBERTa architectures, requiring CUDA-capable GPUs for efficient training

### Data Processing
- Always regenerate `counts` after modifying the keyword list in `add_counts_field.py`
- The current keyword list is case-sensitive and should not be modified without retraining all models
- Train/test split uses `seed=42` and stratified sampling to ensure consistent evaluation

### Training Considerations
- Weighted classifier training may show lower accuracy initially but higher F1-macro
- Monitor F1-macro instead of accuracy for imbalanced classification tasks
- Checkpoints are saved every 1000 steps with a limit of 2 total checkpoints
- Training logs are saved to `{output_dir}/logs/`

### Inference
- Inference scripts automatically apply the same train/test split to ensure test set isolation
- Results are saved in Markdown format for easy review
- Both scripts sample randomly from the test set with `random_state=42` for reproducibility
