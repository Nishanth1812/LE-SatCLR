import json
import math
import tempfile
import unittest
from contextlib import ExitStack
from itertools import count
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn

from src import training
from src.config import Config


class TinyModel(nn.Module):
    def __init__(self, paired):
        super().__init__()
        self.encoder = nn.Linear(3, 4)
        self.head = nn.Linear(4, 10)
        self.unused = nn.Parameter(torch.ones(1))
        self.paired = paired

    def forward(self, x):
        h = self.encoder(x)
        return (h, nn.functional.normalize(h, dim=1)) if self.paired else self.head(h)


class Batches:
    def __init__(self, batches):
        self.batches = batches
        self.iterations = 0

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        self.iterations += 1
        return iter(self.batches)


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.stack.enter_context(patch.object(training, 'time', SimpleNamespace(monotonic=count().__next__)))
        self.stack.enter_context(patch.object(training, 'mlflow', None))
        self.stack.enter_context(patch.object(training, 'seed_everything', lambda: torch.manual_seed(42)))
        self.stack.enter_context(patch.object(torch.cuda, 'is_available', return_value=False))
        self.stack.enter_context(patch.object(training, 'catalog', return_value=list(range(8))))
        self.stack.enter_context(patch.object(training, 'splits', return_value={
            'train': [0, 1, 2, 3], 'labels1': [0, 1, 2, 3],
            'val': [4, 5], 'test': [6, 7],
        }))
        self.loads = []
        self.empty_validation = False
        self.stack.enter_context(patch.object(training, 'loader', side_effect=self.loader))
        self.clip = self.stack.enter_context(patch.object(
            torch.nn.utils, 'clip_grad_norm_', wraps=torch.nn.utils.clip_grad_norm_))

    def loader(self, dataset, indices, batch_size, training=False, paired=False,
               policy='standard', workers=0):
        x = torch.tensor([[1., 0., 0.], [0., 1., 0.]])
        y = x.flip(1) if paired else torch.tensor([0, 1])
        batches = [(x, y)] * (2 if indices[0] == 0 else 1)
        if indices[0] == 4 and self.empty_validation:
            batches = []
        result = Batches(batches)
        self.loads.append((indices, batch_size, training, paired, policy, workers, result))
        return result

    def prepare(self, stage='baseline', epochs=1, workers=0, policy='standard', max_batches=0,
                scaler_enabled=True):
        self.config = Config(stage=stage, epochs=epochs, batch_size=2, workers=workers,
                             policy=policy, max_batches=max_batches, amp=False)
        self.model = TinyModel(stage == 'simclr')
        self.stack.enter_context(patch.object(training, 'SimCLR', return_value=self.model))
        self.stack.enter_context(patch.object(training, 'Classifier', return_value=self.model))
        self.real_scaler = torch.amp.GradScaler('cpu', enabled=scaler_enabled, init_scale=8.)
        self.scaler = Mock(wraps=self.real_scaler)
        self.stack.enter_context(patch.object(torch.amp, 'GradScaler', return_value=self.scaler))

    def run_training(self):
        return training.train(self.config, self.root, self.root)

    def history(self):
        return json.loads(next((self.root/'results').rglob('history.json')).read_text())

    def test_validation_reused_with_paired_transforms_and_workers(self):
        self.prepare('simclr', epochs=2, workers=3, policy='satellite', max_batches=1)
        summary = self.run_training()
        self.assertEqual(len(self.loads), 2)
        indices, size, augmented, paired, policy, workers, batches = self.loads[1]
        self.assertEqual((indices, size, augmented, paired, policy, workers),
                         ([4, 5], 2, True, True, 'satellite', 3))
        self.assertEqual(batches.iterations, 2)
        self.assertEqual(self.clip.call_count, 2)
        self.assertTrue(Path(summary['checkpoint']).is_file())
        self.assertTrue(all(math.isfinite(row['val_pair_accuracy']) for row in self.history()))

    def test_supervised_validation_reused_without_training_transforms(self):
        self.prepare(epochs=2, workers=3)
        self.run_training()
        self.assertEqual(len(self.loads), 3)
        self.assertEqual(self.loads[1][2:6], (False, False, 'standard', 3))
        self.assertEqual(self.loads[1][-1].iterations, 2)
        self.assertEqual(self.scaler.step.call_count, 4)
        self.assertEqual(self.scaler.update.call_count, 4)
        self.assertEqual(self.clip.call_count, 4)
        self.assertIsNone(self.model.unused.grad)
        self.assertTrue(all(math.isfinite(row['train_loss']) for row in self.history()))

    def test_nonfinite_gradient_skips_step_updates_scaler_and_recovers(self):
        self.prepare()
        gradients = iter([float('nan'), 1.])
        self.model.encoder.weight.register_hook(lambda grad: grad * next(gradients))
        self.run_training()
        self.assertEqual(self.scaler.unscale_.call_count, 2)
        self.assertEqual(self.clip.call_count, 2)
        self.assertEqual(self.scaler.step.call_count, 1)
        self.assertEqual(self.scaler.update.call_count, 2)
        self.assertEqual(self.real_scaler.get_scale(), 4.)
        row = self.history()[0]
        self.assertAlmostEqual(row['samples_per_second'] * row['epoch_seconds'], 2.)
        self.assertTrue(all(torch.isfinite(p).all() for p in self.model.parameters()))

    def test_all_nonfinite_gradients_raise_before_metrics_or_checkpoint(self):
        self.prepare()
        before = {key: value.clone() for key, value in self.model.state_dict().items()}
        self.model.encoder.weight.register_hook(lambda grad: torch.full_like(grad, float('inf')))
        with self.assertRaisesRegex(FloatingPointError, 'No successful training samples'):
            self.run_training()
        self.assertEqual(self.scaler.step.call_count, 0)
        self.assertEqual(self.scaler.update.call_count, 2)
        self.assertEqual(self.real_scaler.get_scale(), 2.)
        self.assertFalse(list(self.root.rglob('history.json')))
        self.assertFalse(list(self.root.rglob('*.pt')))
        for key, value in self.model.state_dict().items():
            torch.testing.assert_close(value, before[key])

    def test_nonfinite_total_norm_skips_even_when_gradients_are_finite(self):
        self.prepare()
        self.model.encoder.weight.register_hook(lambda grad: torch.full_like(grad, 1e30))
        with self.assertRaisesRegex(FloatingPointError, 'No successful training samples'):
            self.run_training()
        self.assertEqual(self.scaler.step.call_count, 0)
        self.assertEqual(self.scaler.update.call_count, 2)
        self.assertEqual(self.real_scaler.get_scale(), 8.)

    def test_disabled_scaler_still_skips_nonfinite_gradients(self):
        self.prepare(scaler_enabled=False)
        gradients = iter([float('inf'), 1.])
        self.model.encoder.weight.register_hook(lambda grad: grad * next(gradients))
        self.run_training()
        self.assertEqual(self.scaler.step.call_count, 1)
        self.assertEqual(self.scaler.update.call_count, 2)

    def test_empty_paired_validation_raises_clear_error(self):
        self.prepare('simclr')
        self.empty_validation = True
        with self.assertRaisesRegex(ValueError, 'No validation samples available'):
            self.run_training()


if __name__ == '__main__':
    unittest.main()
