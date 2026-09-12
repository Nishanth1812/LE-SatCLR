# LE-SatCLR Results (2026-09-12)

Hardware: Modal L40S (SSL) + A10G rerun for 1% downstream. Seed 42.
Protocol: transductive (`ssl_scope=all`). Augmentation policy: `standard`.
All numbers below are measured test/val metrics from Modal runs, not estimates.
Checkpoints and logs persist in the `le-satclr-outputs` volume.

## Stage 1 — SimCLR pretraining (unlabeled, 100 epochs, batch 128)

Run: `simclr_standard_unlabeled_seed42/8b46990e25c54c2fb3617e1141dde428`
Checkpoint: `/outputs/models/simclr_standard_unlabeled_seed42/8b46990e25c54c2fb3617e1141dde428/simclr_standard_unlabeled_seed42_best.pt`

| Metric | Epoch 1 | Final (ep 100) | Best |
|---|---|---:|---:|
| NT-Xent train loss | 4.519 | 3.731 | 3.731 (ep 100) |
| Val pair-retrieval accuracy (label-free) | 0.110 | 0.887 | 0.893 (ep 97) |

~51 min total, ~1008 samples/s, 2452 MB GPU peak. A few AMP-overflow steps
were skipped via `grad_nonfinite_skipped` instead of crashing.

## Primary result — test accuracy at fixed label budgets

| Method | Accuracy @ 1% (270 labels) | Accuracy @ 10% (2700 labels) |
|---|---:|---:|
| Supervised ResNet-18 from scratch | 0.6422 | 0.8174 |
| SimCLR + Linear probe (frozen) | 0.7663 | 0.8244 |
| SimCLR + Fine-tuning | **0.7859** | **0.8733** |

SSL pretraining beats scratch by **+14.4pp @ 1%** and **+5.6pp @ 10%**.

## Full test metrics

| Budget | Method | Acc | Precision | Recall | Macro-F1 | Best val (ep) |
|---|---|---:|---:|---:|---:|---:|
| 1% | Probe (50ep) | 0.7663 | 0.7628 | 0.7594 | 0.7580 | 0.7815 |
| 1% | Fine-tune (75ep, enc LR 1e-4) | 0.7859 | 0.7867 | 0.7790 | 0.7807 | 0.8048 |
| 1% | Baseline (75ep) | 0.6422 | 0.6395 | 0.6307 | 0.6260 | 0.6485 |
| 10% | Probe (50ep) | 0.8244 | 0.8203 | 0.8205 | 0.8195 | 0.8378 (ep 36) |
| 10% | Fine-tune (75ep, enc LR 1e-4) | 0.8733 | 0.8692 | 0.8706 | 0.8689 | 0.8878 (ep 66) |
| 10% | Baseline (75ep) | 0.8174 | 0.8213 | 0.8148 | 0.8138 | 0.8393 (ep 64) |

Checkpoints (`/outputs/models/...`):
- 1% probe: `probe_standard_1pct_seed42/250eaa871e8849c9be64e95a5a579c36/probe_standard_1pct_seed42_best.pt`
- 1% fine-tune: `finetune_standard_1pct_seed42/0b89f34627224ba78e989fd9a363719e/finetune_standard_1pct_seed42_best.pt`
- 1% baseline: `baseline_standard_1pct_seed42/96d796bfccc64c3895d63304f1b22674/baseline_standard_1pct_seed42_best.pt`
- 10% probe: `probe_standard_10pct_seed42/49dfba9955ec4107aa9213597a363c20/probe_standard_10pct_seed42_best.pt`
- 10% fine-tune: `finetune_standard_10pct_seed42/2e40498d44cd46cca6eb1bf7dfdf040a/finetune_standard_10pct_seed42_best.pt`
- 10% baseline: `baseline_standard_10pct_seed42/a1bf8016908645bcbf6b03057339b2cc/baseline_standard_10pct_seed42_best.pt`

## Per-class test recall

### @ 1%

| Class (n) | Probe | Fine-tune | Baseline |
|---|---:|---:|---:|
| AnnualCrop (300) | 0.810 | 0.837 | 0.670 |
| Forest (300) | 0.913 | 0.897 | 0.747 |
| HerbaceousVegetation (300) | 0.687 | 0.673 | 0.387 |
| Highway (250) | 0.432 | 0.556 | 0.372 |
| Industrial (250) | 0.740 | 0.764 | 0.792 |
| Pasture (200) | 0.850 | 0.815 | 0.515 |
| PermanentCrop (250) | 0.568 | 0.616 | 0.552 |
| Residential (300) | 0.947 | 0.967 | 0.967 |
| River (250) | 0.744 | 0.736 | 0.416 |
| SeaLake (300) | 0.903 | 0.930 | 0.890 |

### @ 10%

| Class (n) | Probe | Fine-tune | Baseline |
|---|---:|---:|---:|
| AnnualCrop (300) | 0.780 | 0.803 | 0.727 |
| Forest (300) | 0.927 | 0.967 | 0.960 |
| HerbaceousVegetation (300) | 0.743 | 0.807 | 0.687 |
| Highway (250) | 0.600 | 0.716 | 0.712 |
| Industrial (250) | 0.908 | 0.912 | 0.800 |
| Pasture (200) | 0.865 | 0.895 | 0.835 |
| PermanentCrop (250) | 0.684 | 0.772 | 0.716 |
| Residential (300) | 0.943 | 0.973 | 0.987 |
| River (250) | 0.788 | 0.884 | 0.848 |
| SeaLake (300) | 0.967 | 0.977 | 0.877 |

Hardest classes: Highway and PermanentCrop at both budgets. Fine-tuning gains
most over scratch on River (+32pp @ 1%), Pasture (+30pp @ 1%),
HerbaceousVegetation (+29pp @ 1%), Industrial (+11pp @ 10%) and SeaLake (+10pp @ 10%).

## Still pending

- `uv run python -m src.report` figures (UMAP, accuracy-vs-budget curve,
  confusion figures, nearest-neighbor panels, `comparison.csv`) once outputs
  are pulled locally.
