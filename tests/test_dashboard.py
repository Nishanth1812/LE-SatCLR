import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from src.dashboard import DashboardService, discover_checkpoint


def checkpoint_state(stage):
    return {"config": {"stage": stage}, "model": {}}


class DashboardTests(unittest.TestCase):
    def test_discovers_newest_classifier_checkpoint_and_honors_override(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ssl = root / "outputs/models/ssl.pt"
            final = root / "outputs/models/final.pt"
            ssl.parent.mkdir(parents=True)
            torch.save(checkpoint_state("simclr"), ssl)
            torch.save(checkpoint_state("finetune"), final)

            self.assertEqual(discover_checkpoint(root), final)
            self.assertEqual(discover_checkpoint(root, str(final)), final)
            self.assertIsNone(discover_checkpoint(root, str(root / "missing.pt")))

    def test_serves_only_images_in_saved_test_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "Dataset/EuroSAT_RGB"
            for class_name in ("AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
                               "Industrial", "Pasture", "PermanentCrop", "Residential",
                               "River", "SeaLake"):
                folder = data / class_name
                folder.mkdir(parents=True)
                Image.new("RGB", (4, 4)).save(folder / "sample.jpg")
            split = root / "outputs/results/splits_seed42.json"
            split.parent.mkdir(parents=True)
            split.write_text(json.dumps({"test": [3]}), encoding="utf-8")
            service = DashboardService(root=root)

            self.assertEqual(service.test_image(3).name, "sample.jpg")
            with self.assertRaises(PermissionError):
                service.test_image(2)
            with self.assertRaises(PermissionError):
                service.test_image(99)

    def test_evaluation_job_completes_and_rejects_invalid_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "outputs/models/final.pt"
            checkpoint.parent.mkdir(parents=True)
            torch.save(checkpoint_state("finetune"), checkpoint)
            split = root / "outputs/results/splits_seed42.json"
            split.parent.mkdir(parents=True)
            split.write_text(json.dumps({"test": [1, 2, 3]}), encoding="utf-8")
            (root / "Dataset/EuroSAT_RGB/AnnualCrop").mkdir(parents=True)
            service = DashboardService(root=root)
            result = {"metrics": {"accuracy": 1.0}, "sampleCount": 3}

            with self.assertRaises(ValueError):
                service.start(4)
            with patch.object(service, "_evaluate", return_value=result):
                service.start(0)
                for _ in range(100):
                    if service.current()["state"] != "running":
                        break
                    time.sleep(0.01)

            self.assertEqual(service.current()["state"], "complete")
            self.assertEqual(service.current()["result"], result)

    def test_evaluation_job_reports_safe_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "outputs/models/final.pt"
            checkpoint.parent.mkdir(parents=True)
            torch.save(checkpoint_state("baseline"), checkpoint)
            split = root / "outputs/results/splits_seed42.json"
            split.parent.mkdir(parents=True)
            split.write_text(json.dumps({"test": [1]}), encoding="utf-8")
            (root / "Dataset/EuroSAT_RGB/AnnualCrop").mkdir(parents=True)
            service = DashboardService(root=root)

            with patch.object(service, "_evaluate", side_effect=RuntimeError("private path")):
                service.start(0)
                for _ in range(100):
                    if service.current()["state"] != "running":
                        break
                    time.sleep(0.01)

            self.assertEqual(service.current()["state"], "failed")
            self.assertEqual(service.current()["error"], "Evaluation failed. Check the server log.")


if __name__ == "__main__":
    unittest.main()
