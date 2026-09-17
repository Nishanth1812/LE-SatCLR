import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import os
import json
from unittest.mock import patch

import pytest

import torch

from src.dashboard import (
    DashboardService, _cached_checkpoint_info, _checkpoint_info,
    _load_checkpoint, select_evaluation_indices,
)


def test_select_evaluation_indices_draws_a_unique_subset():
    indices = [10, 20, 30, 40]

    selected = select_evaluation_indices(indices, 3)

    assert len(selected) == 3
    assert len(set(selected)) == 3
    assert set(selected).issubset(indices)


def test_select_evaluation_indices_keeps_full_split_for_zero_limit():
    indices = [10, 20, 30]

    assert select_evaluation_indices(indices, 0) == indices


def test_checkpoint_metadata_is_cached_without_weights(tmp_path):
    checkpoint = tmp_path / "model.pt"
    torch.save({"config": {"stage": "finetune", "label_percent": 10},
                "model": {"weight": torch.ones(2)}, "epoch": 4, "score": 0.8}, checkpoint)
    _cached_checkpoint_info.cache_clear()
    with patch("src.dashboard.torch.load", wraps=torch.load) as load:
        first = _checkpoint_info(checkpoint)
        second = _checkpoint_info(checkpoint)
        assert load.call_count == 1
    assert first == second
    assert "model" not in first
    first["config"]["stage"] = "changed"
    assert _checkpoint_info(checkpoint)["config"]["stage"] == "finetune"
    assert "weight" in _load_checkpoint(checkpoint)["model"]


def test_checkpoint_cache_invalidates_on_replacement_and_deletion(tmp_path):
    checkpoint = tmp_path / "model.pt"
    torch.save({"config": {"stage": "probe"}, "model": {}, "epoch": 1}, checkpoint)
    _cached_checkpoint_info.cache_clear()
    assert _checkpoint_info(checkpoint)["epoch"] == 1
    previous = checkpoint.stat()
    torch.save({"config": {"stage": "probe"}, "model": {}, "epoch": 2}, checkpoint)
    os.utime(checkpoint, ns=(previous.st_atime_ns, previous.st_mtime_ns + 1_000_000_000))
    assert _checkpoint_info(checkpoint)["epoch"] == 2
    checkpoint.unlink()
    assert _checkpoint_info(checkpoint) is None


def test_dashboard_discovery_reuses_cached_metadata(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    checkpoint = model_dir / "model.pt"
    torch.save({"config": {"stage": "baseline"}, "model": {}}, checkpoint)
    service = DashboardService(root=tmp_path, model_dirs=[model_dir])
    _cached_checkpoint_info.cache_clear()
    with patch("src.dashboard.torch.load", wraps=torch.load) as load:
        assert service.status()["checkpoint"]["stage"] == "baseline"
        assert service.checkpoints()[0]["stage"] == "baseline"
        assert service._resolve_checkpoint("model.pt") == checkpoint.resolve()
        assert load.call_count == 1


def test_model_cache_reuses_weights_and_invalidates(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    service = DashboardService(root=tmp_path)
    state = {"config": {"stage": "baseline"}, "model": {}}
    with patch("src.dashboard._load_checkpoint", return_value=state) as load, patch("src.dashboard.Classifier") as classifier:
        model, device = service._classifier(checkpoint)
        assert service._classifier(checkpoint) == (model, device)
        assert load.call_count == 1
        assert classifier.call_count == 1
        previous = checkpoint.stat()
        os.utime(checkpoint, ns=(previous.st_atime_ns, previous.st_mtime_ns + 1_000_000_000))
        service._classifier(checkpoint)
        assert load.call_count == 2
        other = tmp_path / "other.pt"
        other.touch()
        service._classifier(other)
        service._classifier(checkpoint)
        assert load.call_count == 4
        checkpoint.unlink()
        with pytest.raises(FileNotFoundError):
            service._classifier(checkpoint)


def test_split_cache_returns_copies_and_invalidates(tmp_path):
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"test": [1, 2]}))
    service = DashboardService(root=tmp_path, split_path=split)
    with patch.object(Path, "read_text", autospec=True, wraps=Path.read_text) as read:
        read.side_effect = lambda path, **kwargs: '{"test": [1, 2]}'
        assert service._test_indices() == [1, 2]
        service._test_indices().append(3)
        assert service._test_indices() == [1, 2]
        assert read.call_count == 1
    split.write_text(json.dumps({"test": [3, 4, 5]}))
    assert service._test_indices() == [3, 4, 5]
    with pytest.raises(PermissionError):
        service.test_image(1)
    split.unlink()
    with pytest.raises(FileNotFoundError):
        service._test_indices()


def test_dataset_cache_reuses_catalog_and_refreshes(tmp_path):
    from src.data import CLASSES

    root = tmp_path / "dataset"
    root.mkdir()
    for name in CLASSES:
        (root / name).mkdir()
    service = DashboardService(root=tmp_path, data_root=root)
    with patch("src.dashboard.catalog") as catalog, patch("src.dashboard.time.monotonic", return_value=10) as clock:
        first = service._dataset()
        assert service._dataset() is first
        assert catalog.call_count == 1
        clock.return_value = 16
        service._dataset()
        assert catalog.call_count == 2
        directory = root / CLASSES[0]
        previous = directory.stat()
        os.utime(directory, ns=(previous.st_atime_ns, previous.st_mtime_ns + 1_000_000_000))
        service._dataset()
        assert catalog.call_count == 3


def test_warm_evaluation_runs_fresh_inference_with_identical_results(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    service = DashboardService(root=tmp_path)
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12, 10)).eval()
    batch = [(torch.ones(2, 3, 2, 2), torch.tensor([0, 1]))]
    state = {"config": {"stage": "baseline"}, "model": model.state_dict()}
    with patch("src.dashboard.torch.cuda.is_available", return_value=False), patch("src.dashboard.Classifier", return_value=model), patch("src.dashboard._load_checkpoint", return_value=state) as load, patch.object(service, "_dataset"), patch.object(service, "_test_indices", return_value=[0, 1]), patch("src.dashboard.loader", return_value=batch), patch.object(model, "forward", wraps=model.forward) as forward:
        first = service._evaluate(0, checkpoint)
        second = service._evaluate(0, checkpoint)
        assert load.call_count == 1
        assert forward.call_count == 2
        assert first["metrics"] == second["metrics"]
        assert first["samples"] == second["samples"]
        assert first["confusionMatrix"] == second["confusionMatrix"]
