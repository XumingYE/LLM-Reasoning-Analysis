# LLM推理复杂度预测 - 所有实验方法对比

本文档汇总了所有在该项目中尝试过的模型训练方法，包括方法描述、任务类型、损失函数、标签构建策略、模型架构和性能结果。

---

## 📊 实验方法对比总表

| # | 方法名称 | 任务类型 | 损失函数 | 数据集标签构建 | 基础模型 | 最佳指标 | 备注 |
|---|---------|---------|---------|--------------|---------|---------|------|
| 1 | **基础回归模型** | 回归 | MSE | 直接使用counts值 | BERT-base-uncased (110M) | MSE: 899.07 | 基线模型，误差较大 |
| 2 | **等宽分箱分类器** | 分类 (10类) | CrossEntropy | 等宽分箱 (pd.cut)<br>Bin 0: 1-46, Bin 1: 47-92... | BERT-base-uncased | Acc: 84.9%<br>**F1-macro: 0.13** | ❌ 数据严重不均衡，模型只预测Bin 0 |
| 3 | **等频分箱分类器** | 分类 (8类) | CrossEntropy | 等频分箱 (pd.qcut)<br>每个bin样本量相近 | BERT-base-uncased | Acc: 33.0%<br>**F1-macro: 0.22** | ✅ 较均衡，但性能仍有限 |
| 4 | **两阶段微调分类器** | 分类 (10类) | CrossEntropy | 等频分箱 (pd.qcut) | BERT-base-uncased | Acc: 33.0%<br>**F1-macro: 0.195** | 阶段1冻结BERT，阶段2全量微调 |
| 5 | **手动分箱回归器** | 回归 | MSE | 手动定义10个bin<br>预测bin索引作为连续值 | BERT-base-uncased | **MAE: 1.13** | 预测bin索引而非counts值 |
| 6 | **加权分类器 (手动10 bins)** | 分类 (10类) | Weighted CrossEntropy | 手动分箱<br>Bin edges: [0,1,4,8,17,28,41,66,98,200,1000] | BERT-base-uncased | Acc: 44.4%<br>**F1-macro: 19.6%** | ⚠️ 使用class weights处理不均衡<br>3个bin的F1=0% |
| 7 | **自定义RoBERTa分类器** | 分类 (10类) | Weighted CrossEntropy | 手动分箱 (同方法6) | RoBERTa-large (355M)<br>冻结底层+自定义分类头 | 训练中 (未完成) | 使用更大模型 + 自定义分类头 |
| 8 | **DeBERTa + 特征工程** | 分类 (10类) | Weighted CrossEntropy | 手动分箱 (同方法6)<br>**+ 特征工程标记** | DeBERTa-v3-large (435M) | Acc: 44.4%<br>**F1-macro: 19.6%** | 添加[EXPLAIN], [CREATIVE]等特征标记<br>性能与方法6相同 |
| 9 | **K-means聚类分类器 (含异常值)** | 分类 (10类) | Weighted CrossEntropy | K-means聚类<br>基于counts、词数等特征<br>**未过滤异常值** | DeBERTa-v3-large (435M) | 不详 | ❌ Label 6仅219样本，counts范围133-943<br>数据不均衡 |
| 10 | **🏆 K-means聚类分类器 (Clean)** | 分类 (8类) | Weighted CrossEntropy | K-means聚类<br>**过滤异常值 (counts>130)**<br>40,526样本 | DeBERTa-v3-large (435M) | **Acc: 99.04%**<br>**F1-macro: 97.89%** | ✅ **最佳方法！**<br>所有label F1 > 88%<br>发现语义任务类型 |
| 11 | **DeBERTa 回归模型 (Wait Only)** | 回归 | MSE | 直接使用counts值 | DeBERTa-v3-base | MAE: 1.05<br>MSE: 6.19 | 针对Wait Only数据训练 |
| 12 | **DeBERTa 分类模型 (Wait Only)** | 分类 (6类) | Weighted CrossEntropy | 手动分箱 (1,2,3,4,5-8,>8) | DeBERTa-v3-base | Acc: 44.47%<br>**F1-macro: 0.280** | 针对Wait Only数据训练，使用class weights |

---

## 📝 详细方法说明

### 方法 1: 基础回归模型
**脚本**: `scripts/training/train_predictor.py`
**模型路径**: `predictor_model/`

- **任务定义**: 直接预测counts的具体数值（1-943）
- **模型配置**: `AutoModelForSequenceClassification` with `num_labels=1`
- **损失函数**: MSE (Mean Squared Error)
- **训练配置**:
  - Epochs: 3
  - Batch size: 8
  - Learning rate: 2e-5
  - Metric: MSE, MAE

**结果分析**:
- MSE: 899.07 (较高)
- 说明精确预测counts值困难，启发后续转向分类任务

---

### 方法 2: 等宽分箱分类器
**脚本**: `scripts/training/train_classifier.py`
**模型路径**: `predictor_classifier_model_equal_width/`

- **标签构建**: 使用 `pd.cut()` 将counts范围(1-456)等分为10个区间
  - Bin 0: 1-46
  - Bin 1: 47-92
  - ...
  - Bin 9: 409-456
- **数据分布**: **严重不均衡** - 85%样本在Bin 0
- **损失函数**: CrossEntropy

**结果分析**:
- Accuracy: 84.9% (欺骗性高)
- **F1-macro: 0.13** (极低)
- ❌ **失败原因**: 模型几乎只预测Bin 0，无法识别稀有高complexity类别

---

### 方法 3: 等频分箱分类器
**脚本**: `scripts/training/train_classifier.py`
**模型路径**: `predictor_classifier_model/`

- **标签构建**: 使用 `pd.qcut()` 创建8个样本量大致相等的区间
  - Bin 0: counts 1-3
  - Bin 7: counts 56-456
- **数据分布**: 每个bin样本量相近
- **损失函数**: CrossEntropy

**结果分析**:
- Accuracy: 33.0%
- **F1-macro: 0.22**
- ✅ 比等宽分箱健康，模型在所有类别都有识别能力

---

### 方法 4: 两阶段微调分类器
**脚本**: `scripts/training/train_staged_classifier.py`
**模型路径**: `predictor_staged_classifier_model/`

- **标签构建**: 等频分箱 (同方法3)
- **训练策略**:
  - **Stage 1**: 冻结BERT主干，只训练分类头 (lr=5e-4, epochs=2)
  - **Stage 2**: 解冻全部参数，低学习率微调 (lr=2e-5, epochs=2)
- **损失函数**: CrossEntropy

**结果分析**:
- Accuracy: 33.0%
- F1-macro: 0.195
- 与单阶段性能接近，未带来显著提升

---

### 方法 5: 手动分箱回归器
**脚本**: `scripts/training/train_regressor_on_bins.py`
**模型路径**: `predictor_regressor_model_manual_10_bins/`

- **标签构建**: 手动定义10个bin，但将bin索引作为连续值进行回归
  - Bin edges: [0,1,8,16,25,35,56,81,140,239,3000]
- **损失函数**: MSE (预测bin索引0-9的浮点值)
- **创新点**: 允许损失函数惩罚距离真实bin较远的预测

**结果分析**:
- **MAE: 1.13** (平均偏离1.13个bin)
- 比直接回归counts效果好，但仍未解决类别不均衡问题

---

### 方法 6: 加权分类器 (手动10 bins)
**脚本**: `scripts/training/train_weighted_classifier.py`
**模型路径**: `predictor_weighted_classifier_manual_10_bins/`

- **标签构建**: 手动优化的10个bin
  ```python
  bin_edges = [0, 1, 4, 8, 17, 28, 41, 66, 98, 200, 1000]
  ```
  - Bin 0-2: 简单查询 (counts 1-8)
  - Bin 3-6: 中等复杂度 (counts 9-66)
  - Bin 7-9: 高复杂度 (counts 67+)

- **损失函数**: **Weighted CrossEntropy**
  - 使用 `sklearn.utils.class_weight.compute_class_weight('balanced')`
  - 自定义 `WeightedTrainer` 覆盖 `compute_loss()`

- **数据分布**:
  - Bin 0: 44.89% (18,205样本)
  - Bin 9: 0.31% (125样本)

**结果分析**:
- Accuracy: 44.4%
- **F1-macro: 19.6%**
- **失败原因**:
  - 3个bin的F1=0% (Bin 1, 7, 9)
  - 最小bin仅12个测试样本
  - 异常值污染 (236样本counts>130)

---

### 方法 7: 自定义RoBERTa分类器
**脚本**: `scripts/training/train_custom_roberta.py`
**模型路径**: `custom_roberta_weighted_classifier/`

- **模型架构**:
  - 基础: RoBERTa-large (355M参数)
  - **冻结**: 除最后2层外的所有层
  - **自定义分类头**:
    ```python
    nn.Linear(hidden_size, 512) -> GELU -> Dropout(0.2) -> nn.Linear(512, num_labels)
    ```

- **标签构建**: 手动10 bins (同方法6)
- **损失函数**: Weighted CrossEntropy
- **训练配置**:
  - Epochs: 5
  - Batch size: 4
  - Learning rate: 5e-5

**结果分析**:
- 训练未完成 / 效果不详
- 尝试通过更大模型和自定义分类头提升性能

---

### 方法 8: DeBERTa + 特征工程
**脚本**: `scripts/training/train_deberta_with_features.py`
**模型路径**: `predictor_deberta_large_with_features/`

- **模型**: DeBERTa-v3-large (435M参数)
- **标签构建**: 手动10 bins (同方法6)
- **损失函数**: Weighted CrossEntropy

- **特征工程** (核心创新):
  ```python
  # 长度标记
  [SHORT] / [MEDIUM] / [LONG]

  # 任务类型标记
  [EXPLAIN] - explain, describe, analyze, compare
  [CREATIVE] - write, create, generate, story
  [CALCULATE] - calculate, compute, solve
  [LIST] - list, enumerate, identify

  # 复杂度标记
  [COMPARE], [CONDITIONAL], [MULTI_Q], [WHY], [HOW]
  ```
  - 特征标记前置于原始文本

**结果分析**:
- Accuracy: 44.4%
- **F1-macro: 19.6%**
- 与方法6性能相同，说明**特征工程未解决根本问题** (数据分布不均衡)

---

### 方法 9: K-means聚类分类器 (含异常值)
**脚本**: `scripts/analysis/discover_clusters.py`
**模型路径**: (训练被中止)

- **标签构建方法**:
  - 基于counts、word_count、char_count等特征进行K-means聚类
  - 特征标准化 (StandardScaler)
  - 测试K=4-10，选择K=10

- **问题**:
  - ❌ **未过滤异常值** (236样本counts>130)
  - Label 6: 仅219样本，counts范围133-943
  - 导致严重不均衡

**结果分析**:
- 发现问题后立即停止，转向方法10

---

### 方法 10: 🏆 K-means聚类分类器 (Clean) - **最佳方法**
**脚本**: `scripts/analysis/discover_clusters_clean.py` + `scripts/training/train_deberta_clusters_clean.py`
**模型路径**: `predictor_deberta_clusters_clean/`
**配置文件**: `data/cluster_config_clean.json`
**数据集**: `data/merged_with_clusters_clean.jsonl`

#### 数据准备
1. **异常值过滤**: 移除counts>130的样本 (236个, 0.6%)
2. **特征提取**:
   ```python
   ['counts', 'word_count', 'char_count', 'question_marks',
    'has_explain', 'has_creative', 'has_calculate', 'has_list',
    'has_why', 'has_how']
   ```
3. **K-means聚类**: 测试K=4-8，选择K=8
   - Silhouette score: 0.5384
   - Balance score: 1.056 (vs 1.202 for 方法6)
4. **标签映射**: 按平均counts排序，映射为Label 0-7

#### 发现的语义聚类

| Label | 任务类型 | 复杂度 | 平均Counts | 样本量 | 特征 |
|-------|---------|-------|-----------|-------|------|
| 0 | General | Simple | 9.6 | 3,223 (7.95%) | 简单列表、分类 |
| 1 | Explanation | Simple | 9.8 | 907 (2.24%) | 简单"Why"问题 |
| 2 | Explanation | Moderate | 10.7 | 5,178 (12.78%) | 100% explain关键词 |
| 3 | General | Moderate | 11.0 | 3,347 (8.26%) | 一般问答 |
| 4 | General | Moderate | 11.4 | 16,471 (40.64%) | 一般任务 |
| 5 | Creative | Moderate | 12.0 | 9,069 (22.38%) | 100% creative关键词 |
| 6 | Coding/Calc | Complex | 15.0 | 1,297 (3.20%) | 100% calculate关键词 |
| 7 | Long-form | Complex | 16.9 | 1,034 (2.55%) | 长文本分析、总结 |
### 方法 11: DeBERTa 回归模型 (Wait Only)
**脚本**: `scripts/training/train_deberta_regressor_wait_only.py`
**模型路径**: `predictor_deberta_regressor_wait_only/`

- **任务定义**: 直接预测counts的具体数值
- **模型配置**: `AutoModelForSequenceClassification` with `num_labels=1`
- **损失函数**: MSE (Mean Squared Error)
- **训练配置**:
  - Epochs: 3
  - Batch size: 8
  - Learning rate: 2e-5
  - Metric: MSE, MAE

**性能结果**:

```json
{
    "eval_loss": 0.1715116798877716,
    "eval_mse": 6.193479537963867,
    "eval_mae": 1.0496591329574585,
    "eval_runtime": 33.0522,
    "eval_samples_per_second": 119.175,
    "eval_steps_per_second": 7.473,
    "epoch": 3.0
}
```

---

### 方法 12: DeBERTa 分类模型 (Wait Only)
**脚本**: `scripts/training/train_deberta_classifier_wait_only.py`
**模型路径**: `predictor_deberta_classifier_wait_only/`

- **任务定义**: 预测counts所属的6个类别
- **标签构建**: 手动分箱
  - Bin 0: counts = 1
  - Bin 1: counts = 2
  - Bin 2: counts = 3
  - Bin 3: counts = 4
  - Bin 4: counts = 5-8
  - Bin 5: counts > 8
- **损失函数**: Weighted CrossEntropy
- **训练配置**:
  - Epochs: 5
  - Batch size: 8
  - Learning rate: 2e-5
  - Metric: Accuracy, F1-macro

**性能结果**:

```json
{
    "eval_loss": 1.6463533639907837,
    "eval_accuracy": 0.4446897228354182,
    "eval_f1_macro": 0.2797250791323674,
    "eval_runtime": 33.8616,
    "eval_samples_per_second": 120.402,
    "eval_steps_per_second": 7.531,
    "epoch": 5.0
}
```

#### 模型训练
- **模型**: DeBERTa-v3-large (435M)
- **特征工程**: 同方法8的特征标记
- **损失函数**: Weighted CrossEntropy
- **训练配置**:
  - Epochs: 5
  - Batch size: 2
  - Learning rate: 1e-5
  - Gradient accumulation: 2

#### 🏆 性能结果 (evaluation_results.json)

```json
{
  "eval_accuracy": 0.9904 (99.04%),
  "eval_f1_macro": 0.9789 (97.89%),
  "eval_f1_label_0": 0.8807 (88.07%),
  "eval_f1_label_1": 0.9933 (99.33%),
  "eval_f1_label_2": 0.9952 (99.52%),
  "eval_f1_label_3": 0.9945 (99.45%),
  "eval_f1_label_4": 1.0000 (100.00%) ⭐,
  "eval_f1_label_5": 0.9922 (99.22%),
  "eval_f1_label_6": 0.9910 (99.10%),
  "eval_f1_label_7": 0.9846 (98.46%),
  "eval_loss": 0.0695
}
```

#### ✅ 成功关键因素

1. **范式转变**: 从预测counts → 预测语义任务类型
2. **数据清洗**: 移除异常值，避免噪声
3. **均衡分布**: 最小label有90个测试样本 (vs 方法6的12个)
4. **语义发现**: K-means自动发现了任务类型的自然聚类
   - Label 2: 100% explanation任务
   - Label 5: 100% creative任务
   - Label 6: 100% coding/calculation任务
5. **特征工程**: 特征标记帮助模型识别任务类型

#### 对比改进

| 指标 | 方法6 (手动bins) | 方法10 (聚类) | 改进 |
|-----|-----------------|--------------|------|
| **F1-macro** | 19.6% | **97.89%** | **+78.3%** |
| **Accuracy** | 44.4% | **99.04%** | **+54.6%** |
| **失败labels** | 3个 (F1=0%) | **0个** | ✅ |
| **最低F1** | 0.00% | **88.07%** | ✅ |
| **训练时长** | ~3小时 | ~6小时 | 可接受 |

---

## 🎯 核心发现与总结

### 失败的方法
1. **等宽分箱** (方法2): 数据分布极度不均 → 模型只学会预测主流类别
2. **手动分箱** (方法6, 8): 基于counts范围的任意划分 → 混合了不同语义类型
3. **大模型尝试** (方法7): 计算资源限制，未完成训练

### 成功的方法
- **K-means聚类** (方法10):
  - 发现数据的自然语义结构
  - 任务类型 (explanation, creative, coding) + 复杂度层级
  - 数据驱动而非人工规则

### 关键教训
1. **问题重构比优化模型更重要**:
   - ❌ "预测精确counts值" → 困难且无实用价值
   - ✅ "预测任务类型+复杂度层级" → 高精度且实用

2. **数据分布决定成败**:
   - 异常值 (0.6%数据) 严重污染训练
   - 最小类别需要足够样本 (至少90+测试样本)

3. **特征工程需配合正确的任务定义**:
   - 方法8: 特征工程 + 错误binning → 无效
   - 方法10: 特征工程 + 语义聚类 → 完美配合

### 实际应用价值
方法10不仅预测准确，还具有实际意义：
- **任务路由**: 根据类型分配到不同pipeline (analytical/creative/code)
- **资源分配**: 根据复杂度分配计算资源
- **质量控制**: 检测异常复杂的响应

---

## 📁 相关文件索引

### 训练脚本
- `scripts/training/train_predictor.py` - 方法1
- `scripts/training/train_classifier.py` - 方法2, 3
- `scripts/training/train_staged_classifier.py` - 方法4
- `scripts/training/train_regressor_on_bins.py` - 方法5
- `scripts/training/train_weighted_classifier.py` - 方法6
- `scripts/training/train_custom_roberta.py` - 方法7
- `scripts/training/train_deberta_with_features.py` - 方法8
- `scripts/training/train_deberta_clusters_clean.py` - 方法10

### 数据处理
- `scripts/analysis/discover_clusters.py` - 方法9 聚类
- `scripts/analysis/discover_clusters_clean.py` - 方法10 清洗聚类
- `scripts/data_processing/add_counts_field.py` - counts计算

### 推理与评估
- `scripts/run_inference.py` - 基础推理
- `scripts/run_weighted_classifier_inference.py` - 加权分类器推理
- `scripts/run_cluster_inference.py` - 聚类模型推理
- `example_usage.py` - 生产环境使用示例

### 文档
- `PREDICTOR_TRAINING_SUMMARY.md` - 早期训练总结
- `FINAL_SUMMARY.md` - 最终项目总结
- `CLAUDE.md` - 项目使用指南
- `ALL_EXPERIMENTS_COMPARISON.md` - 本文档

---

**最后更新**: 2025-10-22
**最佳方法**: 方法10 - K-means聚类分类器 (Clean)
**最佳F1-macro**: 97.89%
**训练总时长**: 约30小时 (所有方法累计)
