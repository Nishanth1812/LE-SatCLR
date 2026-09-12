import math

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18


def encoder():
    model = resnet18(weights=None)
    model.fc = nn.Identity()
    return model


class SimCLR(nn.Module):
    def __init__(self, hidden_dim=512, projection_dim=128):
        super().__init__()
        self.encoder = encoder()
        self.projector = nn.Sequential(
            nn.Linear(512, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, projection_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        return h, F.normalize(self.projector(h), dim=1)


class Classifier(nn.Module):
    def __init__(self, frozen=False):
        super().__init__()
        self.encoder = encoder()
        self.head = nn.Linear(512, 10)
        self.frozen = frozen
        self.encoder.requires_grad_(not frozen)
        self.train()

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.encoder.eval()  # Freeze BatchNorm buffers as well as parameters.
        return self

    def forward(self, x):
        return self.head(self.encoder(x))


def nt_xent(z1, z2, temperature=0.5):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    if z1.ndim != 2 or z1.shape != z2.shape or z1.shape[0] < 2:
        raise ValueError("Expected matching [B,D] projections with B >= 2")
    # Keep similarity and log-sum-exp in FP32 under mixed precision.
    with torch.autocast(device_type=z1.device.type, enabled=False):
        z = F.normalize(torch.cat((z1, z2)).float(), dim=1)
        logits = z @ z.T / temperature
        n = z1.shape[0]
        logits = logits.masked_fill(torch.eye(2*n, device=z.device, dtype=torch.bool), -torch.inf)
        positives = (torch.arange(2*n, device=z.device) + n) % (2*n)
        return F.cross_entropy(logits, positives)


def parameter_counts(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total-trainable}
