# Sources

Compact source ledger for LE-SatCLR. Add one line here for each new external
source used to make a project decision; keep the purpose, not copied content.

## Project references

- [Architecture specification](<SSL EuroSAT Architecture Plan (1).md>) — primary model/protocol authority; supersedes earlier simplified plans.

- [Project specification](Self-Supervised-Satellite-Land-Cover-Project.md) — required dataset, SimCLR experiment, label budgets, metrics, ablation, and demo.
- [README](README.md) — project name and scope.
- `pyproject.toml` — current Python requirement and MLflow dependency.

## Core method and data

- [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html) — seed and deterministic execution limits.
- [UMAP reproducibility](https://umap-learn.readthedocs.io/en/latest/reproducibility.html) — fixed random state for feature plots.

- [Chen et al., SimCLR (ICML 2020)](https://proceedings.mlr.press/v119/chen20j.html) — two augmented views, projection head, contrastive objective, and representation evaluation.
- [TorchVision EuroSAT](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.EuroSAT.html) — RGB EuroSAT dataset loader and download behavior.
- [TorchVision ResNet-18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html) — ResNet-18 API and input preprocessing reference.

## Modal execution and storage

- [Modal setup guide](https://modal.com/docs/guide) — installation, authentication, and cloud execution model.
- [Modal GPU guide](https://modal.com/docs/guide/gpu) — GPU function configuration; project smoke-tested on a T4.
- [Modal Images](https://modal.com/docs/guide/images) — Python version and dependency installation in the remote image.
- [Modal Volumes](https://modal.com/docs/guide/volumes) — separate persistent data and outputs Volumes, checkpoint/results storage, and commit/reload behavior.
- [Modal `run` CLI](https://modal.com/docs/cli/latest/run) — local command used to invoke remote functions.

## MLflow tracking

- [MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking) — runs, parameters, metrics, and artifacts.
- [MLflow Tracking API](https://mlflow.org/docs/latest/ml/tracking/tracking-api) — `start_run`, `log_param`, `log_metric`, and `log_artifact` usage.
- [MLflow PyTorch integration](https://mlflow.org/docs/latest/ml/deep-learning/pytorch/index.html) — manual logging guidance for custom vanilla-PyTorch loops.

## Runtime service

- User-provided MLflow endpoint: `https://coaches-succeed-fresh-joy.trycloudflare.com/` — configured only as `MLFLOW_TRACKING_URI` through the Modal Secret `le-satclr-mlflow`; this is not a research citation and may be temporary.

Last updated: 2026-09-12
