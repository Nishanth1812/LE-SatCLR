import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms as T
from sklearn.model_selection import train_test_split

CLASSES = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
           'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']


def seed_everything():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def transforms(training=False, policy='standard'):
    ops = []
    if training:
        ops = [T.RandomResizedCrop(64, scale=(0.5, 1.0)), T.RandomHorizontalFlip()]
        if policy == 'satellite':
            ops += [T.RandomVerticalFlip(), T.RandomRotation((0, 360))]
        strength = 0.2 if policy == 'satellite' else 0.5
        ops += [T.RandomApply([T.ColorJitter(strength,strength,strength,0.05)], p=0.8),
                T.RandomApply([T.GaussianBlur(3)], p=0.5)]
    else:
        ops = [T.Resize((64,64))]
    return T.Compose(ops + [T.ToTensor(), T.Normalize((0.5,)*3,(0.5,)*3)])


class EuroSATFolder(ImageFolder):
    def find_classes(self, directory):
        missing = [name for name in CLASSES if not (Path(directory)/name).is_dir()]
        if missing: raise ValueError(f'Missing EuroSAT classes: {missing}')
        return CLASSES, {name:i for i,name in enumerate(CLASSES)}


def catalog(root):
    root = Path(root)
    if not (root/'AnnualCrop').is_dir(): root = root/'EuroSAT_RGB'
    dataset = EuroSATFolder(root)
    if dataset.classes != CLASSES:
        raise ValueError(f'Expected ten EuroSAT RGB classes, got {dataset.classes}')
    return dataset


def splits(dataset, path):
    labels = np.asarray(dataset.targets)
    indices = np.arange(len(labels))
    train, held = train_test_split(indices, test_size=0.2, stratify=labels, random_state=42)
    val, test = train_test_split(held, test_size=0.5, stratify=labels[held], random_state=42)
    # Nested budgets use 1% and 10% of the complete dataset, as specified.
    ten, _ = train_test_split(train, train_size=round(len(labels)*0.1), stratify=labels[train], random_state=42)
    one, _ = train_test_split(ten, train_size=round(len(labels)*0.01), stratify=labels[ten], random_state=42)
    result = {k:v.tolist() for k,v in dict(train=train,val=val,test=test,labels1=one,labels10=ten).items()}
    result['classes'] = dataset.classes
    result['seed'] = 42
    result['catalog_sha256'] = hashlib.sha256('\n'.join(
        str(Path(p).relative_to(dataset.root)) for p,_ in dataset.samples).encode()).hexdigest()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and json.loads(path.read_text()) != result:
        raise ValueError('Saved split differs from dataset; use a new output directory')
    path.write_text(json.dumps(result), encoding='utf-8')
    return result


class Views(Dataset):
    def __init__(self, dataset, indices, training=False, paired=False, policy='standard'):
        self.dataset, self.indices, self.paired = dataset, indices, paired
        self.transform = transforms(training, policy)

    def __len__(self): return len(self.indices)

    def __getitem__(self, i):
        image, label = self.dataset[self.indices[i]]
        first = self.transform(image)
        return (first, self.transform(image)) if self.paired else (first, label)


def loader(dataset, indices, batch_size, training=False, paired=False, policy='standard', workers=0):
    return DataLoader(Views(dataset, indices, training, paired, policy),
                      batch_size=batch_size, shuffle=training, drop_last=paired,
                      num_workers=workers, worker_init_fn=seed_worker,
                      generator=torch.Generator().manual_seed(42))
