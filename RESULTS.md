# LE-SatCLR Results — SimCLR + 10% Downstream (2026-09-12)

Hardware: Modal L40S. Seed 42. Protocol: transductive (`ssl_scope=all`).
Augmentation policy: `standard`. Checkpoints and logs persist in the
`le-satclr-outputs` volume; metrics below are measured test/val numbers.

## Stage 1 — SimCLR pretraining (unlabeled, 100 epochs, batch 128)

Run: `simclr_standard_unlabeled_seed42/8b46990e25c54c2fb3617e1141dde428`
Checkpoint: `/outputs/models/simclr_standard_unlabeled_seed42/8b46990e25c54c2fb3617e1141dde428/simclr_standard_unlabeled_seed42_best.pt`

| Metric | Epoch 1 | Final (ep 100) | Best |
|---|---|---:|---:|
| NT-Xent train loss | 4.519 | 3.731 | 3.731 (ep 100) |
| Val pair-retrieval accuracy (label-free) | 0.110 | 0.887 | 0.893 (ep 97) |

~51 min total, ~1008 samples/s, 2452 MB GPU peak. A few AMP-overflow steps
were skipped via `grad_nonfinite_skipped` instead of crashing.

## Stage 2/3 + Baseline — 10% labels (2700 images)

| Method | Val acc (best ep) | Test acc | Test precision | Test recall | Test macro-F1 |
|---|---:|---:|---:|---:|---:|
| SimCLR + Linear probe (50ep, frozen) | 0.8378 (ep 36) | 0.8244 | 0.8203 | 0.8205 | 0.8195 |
| SimCLR + Fine-tune (75ep, enc LR 1e-4) | 0.8878 (ep 66) | **0.8733** | 0.8692 | 0.8706 | 0.8689 |
| Supervised ResNet-18 from scratch (75ep) | 0.8393 (ep 64) | 0.8174 | 0.8213 | 0.8148 | 0.8138 |

Checkpoints:
- Probe: `/outputs/models/probe_standard_10pct_seed42/49dfba9955ec4107aa9213597a363c20/probe_standard_10pct_seed42_best.pt`
- Fine-tune: `/outputs/models/finetune_standard_10pct_seed42/2e40498d44cd46cca6eb1bf7dfdf040a/finetune_standard_10pct_seed42_best.pt`
- Baseline: `/outputs/models/baseline_standard_10pct_seed42/a1bf8016908645bcbf6b03057339b2cc/baseline_standard_10pct_seed42_best.pt`

Fine-tuning beats scratch by **+5.6pp** test accuracy; probe beats scratch by +0.7pp.

## Per-class test recall @ 10%

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

Hardest classes: Highway and PermanentCrop. Fine-tuning gains most on
Industrial (+11pp over baseline), SeaLake (+10pp) and HerbaceousVegetation (+12pp).

## Still pending

- 1% budget app (`le-satclr-1pct`): probe started, fine-tune/baseline not yet done.
- After both budgets finish, run `uv run python -m src.report` for UMAP,
  accuracy-vs-budget curve, confusion figures and `comparison.csv`.
