"""Local API for evaluating a trained classifier on the saved test split."""
import argparse
import json
import logging
import os
import random
import threading
import time
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from .data import CLASSES, catalog, loader
from .models import Classifier

log = logging.getLogger(__name__)
CLASSIFIER_STAGES = {"probe", "finetune", "baseline"}


def select_evaluation_indices(indices, sample_limit):
    return indices if not sample_limit else random.sample(indices, sample_limit)


def _load_checkpoint(path):
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
        stage = state.get("config", {}).get("stage")
        return state if stage in CLASSIFIER_STAGES and "model" in state else None
    except (OSError, RuntimeError, ValueError, TypeError):
        return None


@lru_cache(maxsize=128)
def _cached_checkpoint_info(path, mtime_ns, ctime_ns, size):
    state = _load_checkpoint(path)
    if state is None:
        return None
    return {
        "config": {
            "stage": state["config"]["stage"],
            "label_percent": state["config"].get("label_percent"),
        },
        "epoch": state.get("epoch"),
        "score": state.get("score"),
    }


def _checkpoint_info(path):
    if path is None:
        return None
    try:
        path = Path(path).resolve()
        stat = path.stat()
        info = _cached_checkpoint_info(str(path), stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size)
        return {**info, "config": dict(info["config"])} if info else None
    except OSError:
        return None


def discover_checkpoint(root, override=None, model_dirs=None):
    root = Path(root)
    if override:
        path = Path(override).expanduser()
        return path.resolve() if path.is_file() and _checkpoint_info(path) else None
    if model_dirs is None:
        model_dirs = [root / "outputs/models", root / "mlartifacts"]
    candidates = []
    for directory in model_dirs:
        directory = Path(directory)
        if directory.is_dir():
            candidates.extend(path for path in directory.rglob("*.pt") if _checkpoint_info(path))
    return max(candidates, key=lambda path: path.stat().st_mtime).resolve() if candidates else None


class DashboardService:
    def __init__(self, root=None, checkpoint_override=None, data_root=None,
                 split_path=None, model_dirs=None):
        self.root = Path(root or Path.cwd()).resolve()
        self.data_root = Path(data_root or os.getenv("LE_SATCLR_DATA_ROOT")
                              or self.root / "Dataset/EuroSAT_RGB")
        self.split_path = Path(split_path or os.getenv("LE_SATCLR_SPLIT_PATH")
                               or self.root / "outputs/results/splits_seed42.json")
        raw_dirs = model_dirs or os.getenv("LE_SATCLR_MODEL_DIRS")
        if isinstance(raw_dirs, str):
            raw_dirs = [d for d in raw_dirs.split(os.pathsep) if d]
        self.model_dirs = [Path(d) for d in raw_dirs] if raw_dirs else None
        self.checkpoint_override = checkpoint_override or os.getenv("LE_SATCLR_CHECKPOINT")
        self._lock = threading.Lock()
        self._job = {"state": "idle", "progress": {"done": 0, "total": 0}}
        self._cache_lock = threading.RLock()
        self._split_cache = None
        self._dataset_cache = None
        self._model_cache = None

    @staticmethod
    def _file_version(path):
        path = Path(path).resolve()
        stat = path.stat()
        return str(path), stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size

    def _test_indices(self):
        with self._cache_lock:
            version = self._file_version(self.split_path)
            if self._split_cache is None or self._split_cache[0] != version:
                data = json.loads(self.split_path.read_text(encoding="utf-8"))
                indices = data.get("test")
                if not isinstance(indices, list) or not indices or not all(type(i) is int and i >= 0 for i in indices):
                    raise ValueError("Saved split has no valid test indices")
                self._split_cache = version, tuple(indices), frozenset(indices)
            return list(self._split_cache[1])

    def _dataset(self):
        with self._cache_lock:
            root = self.data_root
            if not (root / CLASSES[0]).is_dir():
                root = root / "EuroSAT_RGB"
            version = tuple(self._file_version(root / name) for name in CLASSES)
            now = time.monotonic()
            if (self._dataset_cache is None or self._dataset_cache[0] != version
                    or now - self._dataset_cache[1] >= 5):
                self._dataset_cache = version, now, catalog(self.data_root)
            return self._dataset_cache[2]

    def _classifier(self, checkpoint):
        with self._cache_lock:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            version = self._file_version(checkpoint), str(device)
            if self._model_cache is None or self._model_cache[0] != version:
                self._model_cache = None
                state = _load_checkpoint(checkpoint)
                if not state:
                    raise RuntimeError("No compatible classifier checkpoint")
                model = Classifier(state["config"]["stage"] == "probe")
                model.load_state_dict(state["model"], strict=True)
                model.to(device).eval()
                self._model_cache = version, model, device
            return self._model_cache[1:]

    def status(self):
        checkpoint = discover_checkpoint(self.root, self.checkpoint_override, self.model_dirs)
        dataset_ready = (self.data_root / CLASSES[0]).is_dir()
        try:
            test_samples = len(self._test_indices())
            split_ready = True
        except (OSError, ValueError, json.JSONDecodeError):
            test_samples, split_ready = 0, False
        info = _checkpoint_info(checkpoint) if checkpoint else None
        ready = bool(checkpoint and dataset_ready and split_ready)
        missing = []
        if not checkpoint:
            missing.append("classifier checkpoint")
        if not dataset_ready:
            missing.append("EuroSAT dataset")
        if not split_ready:
            missing.append("saved test split")
        return {
            "ready": ready,
            "message": "Ready to evaluate." if ready else f"Missing {', '.join(missing)}.",
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "datasetReady": dataset_ready,
            "splitReady": split_ready,
            "testSamples": test_samples,
            "checkpoint": None if not info else {
                "name": checkpoint.name,
                "stage": info["config"]["stage"],
                "epoch": info.get("epoch"),
                "score": info.get("score"),
            },
        }

    def checkpoints(self):
        infos = []
        for directory in (self.model_dirs or [self.root / "outputs/models",
                                              self.root / "mlartifacts"]):
            directory = Path(directory)
            if not directory.is_dir():
                continue
            for path in directory.rglob("*.pt"):
                info = _checkpoint_info(path)
                if info:
                    infos.append({
                        "name": path.name,
                        "path": str(path.resolve()),
                        "stage": info["config"]["stage"],
                        "labelPercent": info["config"].get("label_percent"),
                        "epoch": info.get("epoch"),
                        "score": info.get("score"),
                    })
        infos.sort(key=lambda item: (item["labelPercent"] or 0, item["stage"], item["name"]))
        return infos

    def _resolve_checkpoint(self, name):
        if not name:
            return discover_checkpoint(self.root, self.checkpoint_override, self.model_dirs)
        for info in self.checkpoints():
            if info["name"] == name:
                return Path(info["path"])
        override = self.checkpoint_override and Path(self.checkpoint_override)
        if override and override.name == name and _checkpoint_info(override):
            return override.resolve()
        return None

    def test_image(self, index):
        with self._cache_lock:
            self._test_indices()
            if index not in self._split_cache[2]:
                raise PermissionError("Image is not in the saved test split")
        dataset = self._dataset()
        if index < 0 or index >= len(dataset.samples):
            raise IndexError("Dataset index is out of range")
        return Path(dataset.samples[index][0])

    def start(self, sample_limit=0, checkpoint=None):
        status = self.status()
        if not status["ready"]:
            raise RuntimeError(status["message"])
        total = status["testSamples"]
        if sample_limit < 0 or sample_limit > total:
            raise ValueError(f"sampleLimit must be between 0 and {total}")
        resolved = self._resolve_checkpoint(checkpoint)
        if not resolved or not _checkpoint_info(resolved):
            raise ValueError("Unknown checkpoint. Pick one from the model list.")
        total = sample_limit or total
        with self._lock:
            if self._job["state"] == "running":
                raise RuntimeError("An evaluation is already running")
            self._job = {
                "state": "running",
                "startedAt": datetime.now(UTC).isoformat(),
                "progress": {"done": 0, "total": total},
                "checkpoint": Path(resolved).name,
            }
        threading.Thread(target=self._run, args=(sample_limit, str(resolved)), daemon=True).start()
        return self.current()

    def current(self):
        with self._lock:
            return json.loads(json.dumps(self._job))

    def _run(self, sample_limit, checkpoint):
        try:
            result = self._evaluate(sample_limit, checkpoint)
            with self._lock:
                self._job.update(state="complete", result=result,
                                 completedAt=datetime.now(UTC).isoformat())
        except Exception:
            log.exception("Evaluation failed")
            with self._lock:
                self._job.update(state="failed", error="Evaluation failed. Check the server log.",
                                 completedAt=datetime.now(UTC).isoformat())

    def _evaluate(self, sample_limit, checkpoint=None):
        started = time.monotonic()
        dataset = self._dataset()
        indices = self._test_indices()
        indices = select_evaluation_indices(indices, sample_limit)
        checkpoint = Path(checkpoint) if checkpoint else discover_checkpoint(
            self.root, self.checkpoint_override, self.model_dirs)
        if checkpoint is None:
            raise RuntimeError("No compatible classifier checkpoint")
        model, device = self._classifier(checkpoint)
        actual, predicted, confidence = [], [], []
        with torch.inference_mode():
            for images, labels in loader(dataset, indices, 64):
                logits = model(images.to(device, non_blocking=True))
                if not torch.isfinite(logits).all():
                    raise FloatingPointError("Nonfinite test logits")
                probabilities = logits.softmax(1)
                scores, guesses = probabilities.max(1)
                actual.extend(labels.tolist())
                predicted.extend(guesses.cpu().tolist())
                confidence.extend(scores.cpu().tolist())
                with self._lock:
                    self._job["progress"]["done"] = len(actual)
        precision, recall, f1, _ = precision_recall_fscore_support(
            actual, predicted, labels=list(range(10)), average="macro", zero_division=0
        )
        matrix = confusion_matrix(actual, predicted, labels=list(range(10)))
        positions = np.linspace(0, len(indices) - 1, min(24, len(indices)), dtype=int)
        samples = [{
            "datasetIndex": indices[position],
            "actual": CLASSES[actual[position]],
            "predicted": CLASSES[predicted[position]],
            "confidence": confidence[position],
            "isCorrect": actual[position] == predicted[position],
        } for position in positions]
        supports = matrix.sum(axis=1)
        return {
            "sampleCount": len(indices),
            "checkpoint": Path(checkpoint).name,
            "durationSeconds": round(time.monotonic() - started, 2),
            "metrics": {
                "accuracy": accuracy_score(actual, predicted),
                "precision": precision,
                "recall": recall,
                "macroF1": f1,
            },
            "classes": CLASSES,
            "confusionMatrix": matrix.tolist(),
            "perClass": [{
                "name": name,
                "support": int(supports[i]),
                "accuracy": float(matrix[i, i] / supports[i]) if supports[i] else 0,
            } for i, name in enumerate(CLASSES)],
            "samples": samples,
        }


class EvaluationRequest(BaseModel):
    sampleLimit: int = Field(default=0, ge=0)
    checkpoint: str | None = Field(default=None)


def create_app(service, web_dist=None):
    dashboard = FastAPI(title="LE-SatCLR Dashboard", docs_url=None, redoc_url=None)
    origins = [origin.strip() for origin in os.getenv("LE_SATCLR_CORS_ORIGINS", "*").split(",") if origin.strip()]
    dashboard.add_middleware(CORSMiddleware, allow_origins=origins,
                             allow_methods=["*"], allow_headers=["*"])

    @dashboard.middleware("http")
    async def security_headers(request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @dashboard.get("/api/status")
    def get_status():
        return service.status()

    @dashboard.get("/api/checkpoints")
    def list_checkpoints():
        return service.checkpoints()

    @dashboard.post("/api/evaluations", status_code=202)
    def start_evaluation(request: EvaluationRequest):
        try:
            return service.start(request.sampleLimit, request.checkpoint)
        except ValueError as error:
            raise HTTPException(422, str(error)) from None
        except RuntimeError as error:
            raise HTTPException(409, str(error)) from None

    @dashboard.get("/api/evaluations/current")
    def current_evaluation():
        return service.current()

    @dashboard.get("/api/test-images/{dataset_index}")
    def test_image(dataset_index: int):
        try:
            return FileResponse(service.test_image(dataset_index))
        except (PermissionError, IndexError, OSError, ValueError, json.JSONDecodeError):
            raise HTTPException(404, "Test image not found") from None

    web_dist = Path(web_dist) if web_dist else None
    if web_dist and (web_dist / "index.html").is_file():
        dashboard.mount("/", StaticFiles(directory=web_dist, html=True), name="dashboard")
    return dashboard


service = DashboardService()
app = create_app(service, Path.cwd() / "web/dist")


def main():
    parser = argparse.ArgumentParser(description="Serve the LE-SatCLR evaluation dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    import uvicorn
    uvicorn.run("src.dashboard:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
