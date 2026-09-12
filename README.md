# LE-SatCLR
Label-Efficient Satellite Classification using Contrastive Learning

Architecture: [SSL EuroSAT Architecture Plan (1).md](<SSL EuroSAT Architecture Plan (1).md>).
References: [SOURCES.md](SOURCES.md).

## Setup and verification

```powershell
uv sync --frozen
uv run python -m unittest discover -s tests -v
uv run python -m modal run modal_app.py --stage simclr --epochs 1 --batch-size 8 --max-batches 2
```

## Final-model dashboard

The dashboard evaluates a trained `probe`, `finetune`, or `baseline` checkpoint
against the saved EuroSAT test split. It never substitutes mock predictions when
a checkpoint is unavailable. The newest classifier checkpoint under
`outputs/models` or `mlartifacts` is selected automatically; set an explicit one
when needed:

```powershell
$env:LE_SATCLR_CHECKPOINT = "C:\path\to\finetune_best.pt"
```

For frontend development, run the API and Vite in separate terminals:

```powershell
uv run uvicorn src.dashboard:app --reload
cd web
npm install
npm run dev
```

Open `http://localhost:5173`. For the single-process demo build:

```powershell
cd web
npm install
npm run build
cd ..
uv run python -m src.dashboard
```

Then open `http://127.0.0.1:8000`. The interface reports exactly which of the
dataset, saved split, or final checkpoint is missing if it cannot start a run.

The existing Modal Secret `le-satclr-mlflow` supplies `MLFLOW_TRACKING_URI`.
Keep the MLflow server and Cloudflare tunnel running. The data Volume is
`le-satclr-data` at `/vol`; the outputs Volume is `le-satclr-outputs` at `/outputs`.
Local runs accept `--tracking-uri` (HTTP server or SQLite URI).

## Training

All Modal functions use L40S GPUs. Omit `--epochs`/`--batch-size` to use
per-stage defaults: SimCLR 200/128, probe 50/64, fine-tune 75/64,
baseline 100/64. Add `--detach` to close your laptop mid-run.

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
logs, history, summary, and confusion matrix CSV. MLflow receives parameters,
per-epoch metrics, results, and the best checkpoint. Exceptions propagate and
are recorded in the persisted log. Volumes commit after each epoch and on exit.
`models/<experiment>/<run_id>/<experiment>_best.pt` prevents rerun collisions;
it includes model/encoder weights, optimizer, scheduler, scaler, epoch and config.
Classifier best is selected by validation accuracy; SSL best by training loss.
SSL validation reports label-free pair retrieval accuracy, not class accuracy.
Checkpoint loading supports downstream initialization and inference; exact
interrupted-training resume is not yet exposed as a command.

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
