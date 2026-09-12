import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path

import torch
from src.config import Config
from src.training import train
from src.models import Classifier


class TrainingTests(unittest.TestCase):
    def test_all_stages_real_batches(self):
        root = Path('Dataset/EuroSAT_RGB')
        if not root.exists(): self.skipTest('Local dataset not available')
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as directory, ExitStack() as cleanup:
            out = Path(directory)
            uri = 'sqlite:///' + (out/'tracking.db').as_posix()
            from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
            # MLflow caches pooled SQLite connections; close before Windows cleanup.
            cleanup.callback(lambda: SqlAlchemyStore._engine_map[uri].dispose() if uri in SqlAlchemyStore._engine_map else None)
            ssl = train(Config(epochs=1,batch_size=4,max_batches=2),root,out,tracking_uri=uri)
            checkpoint = torch.load(ssl['checkpoint'],weights_only=True)
            self.assertEqual(checkpoint['scheduler']['last_epoch'],1)
            for percent in (1,10):
                for stage in ('probe','finetune','baseline'):
                    summary = train(Config(stage=stage,label_percent=percent,epochs=1,batch_size=4,max_batches=2),root,out,
                                    encoder_checkpoint=ssl['checkpoint'] if stage!='baseline' else None,tracking_uri=uri)
                    self.assertTrue(Path(summary['checkpoint']).exists())
                    restored = Classifier(stage=='probe')
                    state = torch.load(summary['checkpoint'],weights_only=True)
                    restored.load_state_dict(state['model'])
                    self.assertTrue(torch.isfinite(restored.eval()(torch.randn(2,3,64,64))).all())
                    same = all(torch.equal(v,checkpoint['encoder'][k]) for k,v in state['encoder'].items())
                    self.assertEqual(same,stage=='probe')
                    self.assertIn('macro_f1',summary['test'])
            logs = list((out/'results').glob('*/*/training.log'))
            self.assertEqual(len(logs),7)
            self.assertTrue(all('run_finished' in p.read_text() for p in logs))
