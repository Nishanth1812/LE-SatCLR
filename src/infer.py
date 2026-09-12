"""One-command inference: predict the land-cover class of a satellite image."""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms as T

from .dashboard import discover_checkpoint
from .data import CLASSES
from .models import Classifier


def load_classifier(checkpoint, device=None):
    checkpoint = Path(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    stage = state["config"]["stage"]
    if stage not in ("probe", "finetune", "baseline"):
        raise ValueError(f"Not a classifier checkpoint: {checkpoint}")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Classifier(stage == "probe")
    model.load_state_dict(state["model"], strict=True)
    return model.to(device).eval(), stage, device


def predict_image(model, image_path, device=None, topk=3):
    device = device or next(model.parameters()).device
    preprocess = T.Compose([
        T.Resize((64, 64)), T.ToTensor(), T.Normalize((0.5,) * 3, (0.5,) * 3),
    ])
    with Image.open(image_path) as image:
        tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = model(tensor).softmax(1)[0]
    if not torch.isfinite(probabilities).all():
        raise FloatingPointError("Nonfinite prediction")
    scores, indices = probabilities.topk(min(topk, len(CLASSES)))
    return {
        "predicted": CLASSES[indices[0].item()],
        "confidence": scores[0].item(),
        "topk": [
            {"class": CLASSES[i.item()], "confidence": s.item()}
            for s, i in zip(scores, indices)
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Predict EuroSAT land-cover class")
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--topk", type=int, default=3)
    args = parser.parse_args()
    checkpoint = args.checkpoint or discover_checkpoint(Path.cwd())
    if not checkpoint:
        raise SystemExit("No classifier checkpoint found under outputs/models/")
    model, stage, device = load_classifier(checkpoint)
    print(json.dumps({"checkpoint": str(checkpoint), "stage": stage,
                      **predict_image(model, args.image, device, args.topk)}, indent=2))


if __name__ == "__main__":
    main()
