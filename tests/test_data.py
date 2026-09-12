import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from src.data import splits, catalog, loader, seed_everything


class DataTests(unittest.TestCase):
    def test_real_dataset_views_and_splits(self):
        root = Path('Dataset/EuroSAT_RGB')
        if not root.exists(): self.skipTest('Local EuroSAT not available')
        dataset = catalog(root)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'split.json'
            split = splits(dataset,path)
            modified = path.stat().st_mtime_ns
            self.assertEqual(split,splits(dataset,path))
            self.assertEqual(path.stat().st_mtime_ns, modified, 'Reusing splits must not rewrite shared files')
            self.assertEqual(len(split['labels1']),270)
            self.assertEqual(len(split['labels10']),2700)
            self.assertTrue(set(split['labels1']) <= set(split['labels10']) <= set(split['train']))
            self.assertFalse(set(split['train']) & set(split['val']))
            self.assertFalse(set(split['train']) & set(split['test']))
            self.assertFalse(set(split['val']) & set(split['test']))
            for key in ['labels1','labels10']:
                self.assertEqual(len(set(np.asarray(dataset.targets)[split[key]])),10)
            seed_everything()
            a,b = next(iter(loader(dataset,split['train'],4,True,True)))
            seed_everything()
            c,e = next(iter(loader(dataset,split['train'],4,True,True)))
            self.assertEqual(a.shape,(4,3,64,64))
            self.assertTrue(torch.isfinite(a).all())
            torch.testing.assert_close(a,c); torch.testing.assert_close(b,e)
            self.assertFalse(torch.equal(a,b))
