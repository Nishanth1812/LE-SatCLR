# LE-SatCLR
Label-Efficient Satellite Classification using Contrastive Learning

Architecture: [docs/architecture.md](docs/architecture.md).
References: [docs/sources.md](docs/sources.md).

## Setup and verification

```powershell
uv sync --frozen
uv run python -m unittest discover -s tests -v
uv run modal run modal_app.py --stage simclr --epochs 1 --batch-size 8 --max-batches 2
```

No experiment server is required. Training logs to the console and
`results/<experiment>/<run_id>/training.log`; metrics, history, summaries and
checkpoints persist in the outputs Volume. The data Volume is
`le-satclr-data` at `/vol`; the outputs Volume is `le-satclr-outputs` at `/outputs`.
Pass `--tracking-uri` (HTTP server or SQLite URI) only if you want optional
MLflow logging alongside the file logs.

## Training

All Modal functions use A10G GPUs. Omit `--epochs`/`--batch-size` to use
per-stage defaults: SimCLR 100/128, probe 50/64, fine-tune 75/64,
baseline 75/64. Add `--detach` to close your laptop mid-run.

```powershell
uv run modal run --detach modal_app.py --stage simclr
uv run python launch_downstream.py --ssl-checkpoint /outputs/models/EXPERIMENT/RUN_ID/CHECKPOINT.pt
```

`launch_downstream.py` runs 1% and 10% budgets in parallel as detached apps
`le-satclr-1pct` and `le-satclr-10pct`; each runs probe, fine-tune and baseline
(3 results) from the same SSL checkpoint via `downstream_remote`. Single-stage
runs are also available:

```powershell
uv run modal run modal_app.py --stage probe --label-percent 1 --encoder-checkpoint /outputs/models/EXPERIMENT/RUN_ID/CHECKPOINT.pt
uv run modal run modal_app.py --stage finetune --label-percent 1 --encoder-checkpoint /outputs/models/EXPERIMENT/RUN_ID/CHECKPOINT.pt
uv run modal run modal_app.py --stage baseline --label-percent 1
```

Repeat downstream commands with `--label-percent 10`. Each probe/fine-tune starts
from the same SSL checkpoint, with a fresh classifier. `--policy satellite`
enables rotation/vertical flip and milder color jitter for the augmentation
ablation; compare to `standard` with otherwise identical settings.
`--max-batches` limits training and evaluation batches (a dev shortcut for quick
checks); omit it for scientific results. GPU throughput uses mixed precision on
CUDA, pinned memory, non-blocking transfers, 4 data workers on Modal, and
cudnn benchmarking with seed-42 initialization.

## Architecture and protocol

Two independent RGB views `[B,3,64,64]` pass through the same random-initialized
standard ResNet-18 (original stem and residual blocks, `fc=Identity`), producing
`[B,512]` features. The projector is Linear(512,512), ReLU, Linear(512,128),
followed by L2 normalization. NT-Xent uses both directions, excludes only self
similarities, and defaults to temperature 0.5. Its similarity/loss computation
is FP32 even when CUDA mixed precision is enabled.

Probing discards the projector, freezes encoder parameters AND BatchNorm
buffers, and trains Linear(512,10). Fine-tuning trains both encoder (LR 1e-4)
and classifier (LR 1e-3). Baseline initializes both randomly. All classifiers
use cross entropy; optimization uses AdamW and an epoch cosine schedule.
Input normalization is fixed `(x-0.5)/0.5`; no ImageNet weights are used.

Seed is always 42. Saved stratified splits contain 21,600 training, 2,700
validation, and 2,700 test images. Nested labeled subsets contain 270 and 2,700
training images (percentages of all 27,000 images, per specification). Validation
labels are additional model-selection supervision and are not included in those
training-label budgets.

The specification explicitly requests all images for SSL: default `--ssl-scope all`
is therefore **transductive** (validation/test images seen without labels).
Use `--ssl-scope train` consistently across runs for an inductive comparison.
Never mix these protocols in a reported result. Seed 42 does not guarantee
bitwise equivalence across hardware, library versions, or platforms.

## Outputs and logging

`results/<experiment>/<run_id>/` holds timestamped elapsed-time JSON console/file
logs, history, summary, and confusion matrix CSV. If `--tracking-uri` is given,
the same parameters, per-epoch metrics, results, and best checkpoint also go to
MLflow. Exceptions propagate and are recorded in the persisted log.
Volumes commit after each epoch and on exit.
`models/<experiment>/<run_id>/<experiment>_best.pt` prevents rerun collisions;
it includes model/encoder weights, optimizer, scheduler, scaler, epoch and config.
Classifier best is selected by validation accuracy; SSL best by training loss.
SSL validation reports label-free pair retrieval accuracy, not class accuracy.
Checkpoint loading supports downstream initialization and inference; exact
interrupted-training resume is not yet exposed as a command.

## Dashboard (backend + frontend)

The backend serves the trained classifier and the built React frontend from one
process. It needs `Dataset/EuroSAT_RGB`, `outputs/results/splits_seed42.json`
and a classifier checkpoint under `outputs/models/` (newest `*.pt` wins, or set
`LE_SATCLR_CHECKPOINT` to override, e.g.
`$env:LE_SATCLR_CHECKPOINT = "outputs/models/finetune_1pct_best.pt"`).

```powershell
uv run python -m src.dashboard
```

Open http://127.0.0.1:8000 — status, evaluation jobs (accuracy/precision/recall/F1,
confusion matrix, per-class recall, sample predictions with confidence) and
test-split image serving live there. Only images in the saved test split are
served; everything else 404s. To rebuild the frontend: `npm --prefix web install`
then `npm --prefix web run build`. For frontend development with hot reload,
run `uv run uvicorn src.dashboard:app --reload` and `npm --prefix web run dev`
in separate terminals, then open `http://localhost:5173`.

## One-command inference

```powershell
uv run python -m src.infer --image Dataset/EuroSAT_RGB/SeaLake/SeaLake_1.jpg --checkpoint outputs/models/finetune_10pct_best.pt
```

Omit `--checkpoint` to use the newest `*.pt` under `outputs/models/`.

## Reporting

After retrieving outputs, run:

```powershell
uv run python -m src.report --ssl-checkpoint SSL.pt --finetuned-checkpoint FINETUNED.pt
```

This extracts validation/test 512-D features for random, SSL, and fine-tuned
encoders; writes UMAP (seed 42), nearest-neighbor panels, training curves,
confusion matrices, and a comparison CSV/accuracy curve for completed real runs.
Only measured 1% and 10% budgets are plotted; optional t-SNE and extra budgets
are not run. Runs limited with `--max-batches` are excluded from the comparison
table. Test performance is evidence to
measure, not a guaranteed ordering of methods.
