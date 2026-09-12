# LE-SatCLR Test Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a polished Vite dashboard that runs and presents final-model evaluation on the saved EuroSAT test split.

**Architecture:** A FastAPI backend reuses the existing PyTorch model and data pipeline, manages one background evaluation job, and serves the built Vite app. A React TypeScript frontend polls that API and renders readiness, progress, metrics, confusion counts, and sample predictions.

**Tech Stack:** Python 3.12, PyTorch, FastAPI, Uvicorn, React, TypeScript, Vite, CSS

**Spec:** `docs/superpowers/specs/2026-09-12-test-dashboard-design.md`

## Global Constraints

- Evaluate only indices from `outputs/results/splits_seed42.json` and the configured EuroSAT dataset.
- Never emit fake model results when no compatible classifier checkpoint exists.
- Keep one evaluation in process; reject concurrent starts.
- Add no UI component or chart dependency; native HTML, CSS, and SVG are sufficient.
- Reuse `src.models.Classifier`, `src.data.catalog`, `src.data.loader`, and `src.training.evaluate` where their contracts fit.

---

### Task 1: Evaluation service and API

**Files:**
- Create: `src/dashboard.py`
- Create: `tests/test_dashboard.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Produces: `discover_checkpoint(root: Path, override: str | None) -> Path | None`
- Produces: `DashboardService.status() -> dict`, `start(sample_limit: int) -> dict`, `current() -> dict`, and `test_image(index: int) -> Path`
- Produces: FastAPI `app` with `/api/status`, `/api/evaluations`, `/api/evaluations/current`, and `/api/test-images/{dataset_index}`

- [ ] Write focused `unittest` cases using temporary split/checkpoint paths and injected evaluation behavior; assert classifier-only discovery, test-index protection, bounded samples, and complete/failed job states.
- [ ] Run `uv run python -m unittest tests.test_dashboard -v` and confirm it fails because `src.dashboard` does not exist.
- [ ] Implement the smallest service and FastAPI routes that satisfy the design, loading datasets/models lazily and using one daemon thread for inference.
- [ ] Add direct `fastapi` and `uvicorn` runtime dependencies to `pyproject.toml`, then refresh `uv.lock`.
- [ ] Run `uv run python -m unittest tests.test_dashboard -v` and confirm all dashboard tests pass.

### Task 2: Vite React dashboard

**Files:**
- Create: `web/package.json`
- Create: `web/package-lock.json`
- Create: `web/index.html`
- Create: `web/tsconfig.json`
- Create: `web/vite.config.ts`
- Create: `web/src/main.tsx`
- Create: `web/src/App.tsx`
- Create: `web/src/styles.css`

**Interfaces:**
- Consumes: the four `/api` endpoints from Task 1.
- Produces: `web/dist` static files served by the backend in production.

- [ ] Scaffold only the Vite/React/TypeScript files needed to run and build the interface.
- [ ] Implement typed API calls and polling for readiness, running, success, and failure states.
- [ ] Implement responsive semantic markup for the hero, run controls, metrics, confusion matrix, and prediction gallery.
- [ ] Add a focused visual system and reduced-motion, focus, loading, empty, and mobile states in one stylesheet.
- [ ] Run `npm run build` in `web` and confirm TypeScript and Vite complete successfully.

### Task 3: Production serving and operator workflow

**Files:**
- Modify: `src/dashboard.py`
- Modify: `README.md`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: `web/dist` from Task 2.
- Produces: `uv run uvicorn src.dashboard:app --reload` development command and `uv run python -m src.dashboard` production command.

- [ ] Add static production serving without shadowing `/api` routes, plus a module entry point with host and port arguments.
- [ ] Document checkpoint override, development commands, production build/start, and the truthful missing-checkpoint state.
- [ ] Ignore generated frontend artifacts while retaining the lockfile.
- [ ] Run the full Python suite and frontend production build.
- [ ] Start the API, exercise status and invalid-image routes, and verify the rendered dashboard in the collaborative browser at desktop and mobile widths.

