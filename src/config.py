from dataclasses import dataclass


STAGE_EPOCHS = {'simclr': 100, 'probe': 50, 'finetune': 75, 'baseline': 100}
STAGE_BATCH_SIZE = {'simclr': 128, 'probe': 64, 'finetune': 64, 'baseline': 64}


def resolve_epochs(stage, epochs):
    if epochs in (None, 0):
        return STAGE_EPOCHS[stage]
    return epochs


def resolve_batch_size(stage, batch_size):
    if batch_size in (None, 0):
        return STAGE_BATCH_SIZE[stage]
    return batch_size


@dataclass
class Config:
    stage: str = 'simclr'
    epochs: int = 100
    batch_size: int = 128
    label_percent: int = 1
    policy: str = 'standard'
    ssl_scope: str = 'all'
    temperature: float = 0.5
    hidden_dim: int = 512
    projection_dim: int = 128
    lr: float = 1e-3
    encoder_lr: float = 1e-4
    weight_decay: float = 1e-4
    workers: int = 0
    max_batches: int = 0
    amp: bool = True
    seed: int = 42

    def __post_init__(self):
        if self.stage not in ('simclr','probe','finetune','baseline'): raise ValueError('Invalid stage')
        if self.policy not in ('standard','satellite'): raise ValueError('Invalid policy')
        if self.ssl_scope not in ('all','train'): raise ValueError('Invalid SSL scope')
        if self.label_percent not in (1,10): raise ValueError('Label budget must be 1 or 10')
        if self.seed != 42: raise ValueError('Seed must be 42')
        if self.epochs < 1 or self.batch_size < 2 or self.max_batches < 0: raise ValueError('Invalid training limits')
