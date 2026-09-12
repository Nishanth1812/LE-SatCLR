import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from src.data import CLASSES
from src.infer import load_classifier, predict_image
from src.models import Classifier


class InferTests(unittest.TestCase):
    def test_predict_returns_valid_topk(self):
        torch.manual_seed(42)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            torch.save({"config": {"stage": "baseline"},
                        "model": Classifier(False).state_dict()}, checkpoint)
            image = Path(directory) / "query.jpg"
            Image.new("RGB", (64, 64), (10, 120, 40)).save(image)
            model, stage, device = load_classifier(checkpoint, torch.device("cpu"))
            self.assertEqual(stage, "baseline")
            result = predict_image(model, image, device, topk=3)
            self.assertIn(result["predicted"], CLASSES)
            self.assertEqual(len(result["topk"]), 3)
            self.assertGreaterEqual(result["confidence"], 0)
            self.assertLessEqual(result["confidence"], 1)
            json.dumps(result)


if __name__ == "__main__":
    unittest.main()
