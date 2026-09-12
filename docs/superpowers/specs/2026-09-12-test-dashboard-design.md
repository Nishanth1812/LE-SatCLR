# LE-SatCLR Test Dashboard Design

## Goal

Provide a polished local web app that evaluates the final trained classifier on the saved EuroSAT test split and makes the results easy to inspect during a demo.

## Scope

- Use the existing `Classifier`, dataset catalog, transforms, saved split, and metric definitions.
- Discover the final classifier checkpoint automatically, with an environment-variable override.
- Run evaluation on the complete saved test split by default, with an optional smaller sample count for quick demos.
- Show accuracy, macro precision, macro recall, macro F1, a class confusion matrix, checkpoint metadata, progress, and representative predictions.
- Serve dataset images only by validated test-split index.
- Do not support uploads, training controls, accounts, persistence, or fabricated fallback results.

## Architecture

The backend is Python because the model and evaluation pipeline already use PyTorch. A small FastAPI app owns checkpoint discovery, lazy model loading, one in-process evaluation job, and safe image delivery. It imports existing project modules instead of duplicating model or data logic.

The frontend is a Vite React TypeScript app. In development, Vite proxies `/api` to FastAPI. In production, FastAPI serves the built frontend, so the entire demo starts as one process after `npm run build`.

## API and Data Flow

- `GET /api/status` reports dataset, split, device, and checkpoint readiness without loading the model.
- `POST /api/evaluations` accepts `{ "sample_limit": 0 }`, where `0` means the full saved test split. Only one evaluation may run at once.
- `GET /api/evaluations/current` returns `idle`, `running`, `complete`, or `failed`, plus progress and results when available.
- `GET /api/test-images/{dataset_index}` returns an RGB image only when the index belongs to the saved test split.
- The browser polls the current evaluation while it is running and renders returned aggregate metrics, per-class confusion counts, and a bounded set of sample predictions.

## Checkpoint Selection

`LE_SATCLR_CHECKPOINT` has priority. Otherwise the backend selects the newest `*_best.pt` beneath `outputs/models` or `mlartifacts` whose saved configuration stage is `probe`, `finetune`, or `baseline`. It rejects missing, unreadable, SimCLR-only, or incompatible checkpoints with an actionable status message.

## Error Handling and Safety

API payloads are schema-validated. Sample limits are bounded to the test split. Image access uses membership in the saved split rather than a filesystem path from the caller. Checkpoints load with `weights_only=True`. Evaluation errors are recorded and shown in the UI without crashing the server.

## Interface

The dashboard uses an earth-observation visual language: dark navy surfaces, warm satellite-green accents, restrained motion, high-contrast typography, and responsive cards. The primary action runs the full test split; a quick-run control can cap the sample count. Empty, loading, running, success, and error states are all explicit and keyboard accessible.

## Verification

- Backend unit tests cover checkpoint selection, safe index validation, evaluation result shape, and job-state transitions using small injected fakes.
- The existing Python test suite must continue to pass.
- The frontend must type-check and build successfully.
- A browser smoke test must verify the dashboard at desktop and mobile widths and confirm the missing-checkpoint and successful API states render correctly.

