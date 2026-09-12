# Label-Efficient Satellite Land-Cover Classification

## Project idea

Satellite images are abundant, but manually labeling land-cover types—such as forests, rivers, residential areas, and farmland—is expensive. This project tests whether a model can learn useful visual features from **unlabeled satellite images** and then achieve good classification accuracy using only a small labeled subset.

## How it works

1. **Dataset:** Use EuroSAT, which contains satellite images from 10 land-cover classes.

2. **Self-supervised pretraining:** Create two augmented versions of every image using crops, rotations, flips, and color changes. A contrastive method such as **SimCLR** trains a ResNet-18 encoder to:

   - Place two versions of the same image close together in embedding space.
   - Keep different images farther apart.

   No class labels are used during this stage.

3. **Limited-label training:** Add a classification layer and train it using only:

   - 1% of the available labels.
   - 10% of the available labels.

4. **Baseline comparison:** Train another ResNet-18 from scratch using exactly the same labeled subsets. This shows whether self-supervised pretraining actually helps.

## Main experiment

| Model | Labels available | Test accuracy |
|---|---:|---:|
| Supervised ResNet-18 | 1% | Baseline |
| SimCLR-pretrained ResNet-18 | 1% | Expected improvement |
| Supervised ResNet-18 | 10% | Baseline |
| SimCLR-pretrained ResNet-18 | 10% | Expected improvement |

Use test accuracy as the primary metric and macro-F1 as an additional metric.

## Required ablation

Compare:

- Standard image augmentations.
- Satellite-specific augmentations.

This determines whether carefully chosen augmentations improve representation learning.

## Demo

The demo can show:

- A satellite image and its predicted land-cover class.
- Similar images retrieved using learned embeddings.
- A 2-D embedding plot where related land-cover classes form clusters.

## Project objective

> Demonstrate that contrastive self-supervised pretraining can reduce the amount of labeled satellite data required for accurate land-cover classification.

The minimum practical implementation is **EuroSAT + ResNet-18 + SimCLR + one supervised baseline + one augmentation ablation**.
