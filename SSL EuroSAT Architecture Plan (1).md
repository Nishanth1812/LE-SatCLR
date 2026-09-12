# Label-Efficient Satellite Land Cover Classification via Contrastive Self-Supervised Representation Learning

## Objective

Build a two-stage learning pipeline that demonstrates how self-supervised contrastive representation learning can reduce dependence on labeled satellite imagery for land cover classification.

The system should first learn visual representations from the entire unlabeled EuroSAT dataset using SimCLR. The learned encoder should then be evaluated using limited labeled subsets (1% and 10%) through both linear probing and supervised fine-tuning.

The primary hypothesis is:

> A contrastively pre-trained encoder will outperform a supervised model trained from scratch when only a small fraction of labels is available.

---

# Dataset

## Dataset

EuroSAT (Sentinel-2 Satellite Imagery)

### Classes

- Annual Crop
- Forest
- Herbaceous Vegetation
- Highway
- Industrial
- Pasture
- Permanent Crop
- Residential
- River
- Sea/Lake

### Input Variant

Preferred implementation:

- RGB EuroSAT
- Input shape: (64 × 64 × 3)

Optional future extension:

- Multi-Spectral EuroSAT (13 Sentinel-2 bands)

---

# Overall System Architecture

The complete workflow consists of three stages:

1. Self-Supervised Contrastive Pretraining
2. Linear Probe Evaluation
3. Supervised Fine-Tuning

---

# Stage 1: Self-Supervised Contrastive Pretraining

## Goal

Learn meaningful satellite-image representations without using class labels.

All labels must be ignored during this stage.

### Input

Entire EuroSAT dataset

```text
Image x
```

### Data Augmentation Pipeline

For every image, generate two independent augmented views.

```text
x
├── T1(x) → x1
└── T2(x) → x2
```

Possible augmentations:

- Random Resized Crop
- Random Horizontal Flip
- Random Vertical Flip
- Random Rotation
- Color Jitter
- Gaussian Blur

The two augmented views constitute a positive pair.

---

## Encoder

Backbone:

```text
ResNet-18
```

Remove final classification layer.

Output feature dimension:

```text
512
```

Representation:

```text
h = Encoder(x)
```

---

## Projection Head

MLP projection network used only during contrastive training.

Architecture:

```text
512
 ↓
512
 ↓
128
```

Example:

```text
Linear
 ↓
ReLU
 ↓
Linear
```

Output:

```text
z
```

where:

```text
z ∈ R^128
```

---

## Contrastive Learning Objective

Use NT-Xent (Normalized Temperature-scaled Cross Entropy) loss.

For each positive pair:

```text
(x1, x2)
```

maximize similarity between:

```text
z1
z2
```

while minimizing similarity with representations from all other images in the batch.

Desired behavior:

```text
Same image views
      ↓
Close together

Different images
      ↓
Far apart
```

---

## Training Output

After contrastive pretraining:

Discard:

```text
Projection Head
```

Retain:

```text
ResNet-18 Encoder
```

Save encoder weights for downstream tasks.

---

# Stage 2: Linear Probe Evaluation

## Purpose

Measure the quality of the learned representations without allowing the encoder to adapt.

This stage evaluates whether the encoder already organizes satellite classes in feature space.

---

## Label Budgets

Create two labeled subsets:

### 1% Regime

Approximately:

```text
270 labeled samples
```

### 10% Regime

Approximately:

```text
2700 labeled samples
```

Use stratified sampling to preserve class balance.

---

## Architecture

```text
Image
  ↓
Frozen Encoder
  ↓
512-D Features
  ↓
Linear Layer
  ↓
10 Classes
```

---

## Training Rules

Encoder:

```text
Frozen
```

No gradient updates allowed.

Train only:

```text
Linear Classification Head
```

Loss:

```text
Cross Entropy Loss
```

---

## Output

Report:

- Accuracy
- Precision
- Recall
- F1 Score

for both:

```text
1% labels
10% labels
```

---

# Stage 3: Supervised Fine-Tuning

## Purpose

Measure maximum downstream performance obtainable from the learned representations.

Unlike linear probing, the encoder is allowed to adapt to the classification task.

---

## Initialization

Load encoder weights learned during Stage 1.

---

## Architecture

```text
Image
  ↓
Pretrained Encoder
  ↓
Classifier
  ↓
Class Prediction
```

---

## Training Rules

Encoder:

```text
Trainable
```

Classifier:

```text
Trainable
```

Use:

```text
Cross Entropy Loss
```

Recommended:

- Smaller learning rate for encoder
- Larger learning rate for classifier

Example:

```text
Encoder LR = 1e-4

Classifier LR = 1e-3
```

---

## Label Budgets

Perform fine-tuning separately on:

```text
1% labels
```

and

```text
10% labels
```

using the same subsets employed in linear probing.

---

# Supervised Baseline

A baseline model must be trained for comparison.

Architecture:

```text
ResNet-18
```

Random initialization.

No self-supervised pretraining.

Train directly on:

```text
1% labels
```

and

```text
10% labels
```

using standard supervised learning.

Loss:

```text
Cross Entropy
```

---

# Experimental Comparisons

Evaluate the following configurations:

### Experiment 1

```text
Supervised ResNet18
(Random Initialization)
```

1% labels

---

### Experiment 2

```text
SimCLR Encoder
+
Linear Probe
```

1% labels

---

### Experiment 3

```text
SimCLR Encoder
+
Fine-Tuning
```

1% labels

---

### Experiment 4

```text
Supervised ResNet18
(Random Initialization)
```

10% labels

---

### Experiment 5

```text
SimCLR Encoder
+
Linear Probe
```

10% labels

---

### Experiment 6

```text
SimCLR Encoder
+
Fine-Tuning
```

10% labels

---

# Primary Result Reporting

The central outcome of this project is the downstream classification performance achieved at fixed label budgets.

The final report must include a consolidated comparison table:

| Method | Accuracy @ 1% | Accuracy @ 10% |
|----------|----------|----------|
| Supervised From Scratch | X | X |
| SimCLR + Linear Probe | X | X |
| SimCLR + Fine-Tuning | X | X |

This table represents the primary evidence for evaluating label efficiency and overall effectiveness of self-supervised pretraining.

---

# Evaluation Metrics

## Primary Metric

### Downstream Accuracy at Fixed Label Budgets

The primary goal of this project is to evaluate whether self-supervised representation learning improves classification performance under limited-label conditions.

Therefore, the primary metric shall be:

```text
Downstream Classification Accuracy
```

measured at fixed label budgets.

The following metrics must be reported:

```text
Downstream Accuracy @ 1% Labels
Downstream Accuracy @ 10% Labels
```

Example Results Table:

| Method | Accuracy @ 1% | Accuracy @ 10% |
|----------|----------|----------|
| Supervised From Scratch | X | X |
| SimCLR + Linear Probe | X | X |
| SimCLR + Fine-Tuning | X | X |

This table should be treated as the primary result table of the project.

---

## Secondary Metrics

Report:

- Classification Accuracy
- Precision
- Recall
- F1 Score

Per-class metrics may also be reported.

---

# Representation Visualization

## Visualization 1: UMAP Feature Space Visualization (Primary)

Generate embeddings using the pretrained encoder.

Extract:

```text
512-dimensional feature vectors
```

for validation and test samples.

Reduce dimensionality using:

```text
UMAP
```

to obtain a 2-D feature space projection.

Visualize:

```text
Feature Space Clusters
```

Expected observation:

Images belonging to similar land-cover categories should form naturally separated clusters despite no labels being used during pretraining.

Compare:

- Randomly Initialized Encoder
- SSL Pretrained Encoder
- Fine-Tuned Encoder

This shall be the primary representation-learning visualization.

---

## Visualization 2: t-SNE Feature Space Visualization (Optional)

Generate a t-SNE projection of encoder embeddings.

Purpose:

```text
Local Cluster Inspection
```

This visualization can be used as supplementary analysis to compare cluster compactness and class separation.

---

## Visualization 3: Accuracy vs Label Budget Curve

Generate a performance curve showing downstream classification accuracy as a function of label availability.

Example label budgets:

```text
1%
5%
10%
25%
50%
100%
```

Plot:

```text
Label Budget
        vs
Downstream Accuracy
```

Compare:

- Supervised From Scratch
- SimCLR + Linear Probe
- SimCLR + Fine-Tuning

Purpose:

```text
Quantify Label Efficiency
```

This visualization directly demonstrates how self-supervised pretraining reduces dependence on labeled data.

---

## Visualization 4: Confusion Matrix

Generate confusion matrices for:

- Supervised Baseline
- SimCLR Fine-Tuned Model

Purpose:

```text
Class-Level Error Analysis
```

Identify:

- Frequently confused classes
- Classes that benefit most from self-supervised pretraining

---

## Visualization 5: Training Curves

Generate:

- Training Loss Curves
- Validation Accuracy Curves

for:

- SimCLR Pretraining
- Linear Probe Training
- Fine-Tuning Stage

Purpose:

```text
Training Stability and Convergence Analysis
```

---

## Visualization 6: Nearest Neighbor Retrieval

Use encoder embeddings to perform nearest-neighbor retrieval.

Procedure:

1. Select a query image.
2. Extract encoder embedding.
3. Compute nearest neighbors in feature space.
4. Display retrieved images.

Expected observation:

```text
Query Image
      ↓
Nearest Embeddings
      ↓
Semantically Similar Satellite Images
```

Purpose:

```text
Qualitative Representation Evaluation
```

This visualization provides an intuitive demonstration that the encoder has learned meaningful semantic similarity relationships without using labels during pretraining.

---

# Expected Outcome

When labels are scarce:

```text
1% labels
10% labels
```

the self-supervised encoder should significantly outperform a supervised model trained from scratch.

Expected ranking:

```text
SimCLR Fine-Tuning
        >
SimCLR Linear Probe
        >
Supervised From Scratch
```