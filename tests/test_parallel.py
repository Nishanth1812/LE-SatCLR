import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import modal_app


class ParallelTests(unittest.TestCase):
    def test_two_launches_overlap_and_failures_are_reported(self):
        import launch_downstream
        barrier = threading.Barrier(2)
        calls = []

        def run(command, **kwargs):
            calls.append((command, kwargs['env']['LE_SATCLR_APP_NAME']))
            barrier.wait(timeout=5)  # Sequential submission would fail here.
            from subprocess import CompletedProcess
            return CompletedProcess(command, 1 if command[-1] == '10' else 0)

        with patch('launch_downstream.subprocess.run', side_effect=run):
            with self.assertRaisesRegex(RuntimeError, '10%'):
                launch_downstream.launch('/outputs/models/ssl.pt', epochs=2, max_batches=1)
        self.assertEqual({name for _, name in calls}, {'le-satclr-1pct', 'le-satclr-10pct'})
        for command, name in calls:
            self.assertIn('--detach', command)
            self.assertEqual(command[command.index('--encoder-checkpoint') + 1], '/outputs/models/ssl.pt')
            self.assertEqual(name, f'le-satclr-{command[-1]}pct')

    def test_each_budget_runs_three_methods_from_original_ssl(self):
        self.assertTrue(hasattr(modal_app, 'downstream_remote'), 'Missing downstream suite')
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root/'results').mkdir()
            (root/'results/splits_seed42.json').write_text('{}')
            (root/'eurosat/EuroSAT_RGB').mkdir(parents=True)
            checkpoint = root/'ssl.pt'
            checkpoint.touch()
            with patch.object(modal_app, 'OUTPUT_PATH', d), patch.object(modal_app, 'VOLUME_PATH', d):
                for budget in (1, 10):
                    with patch.object(modal_app.train_remote, 'local', return_value={}) as train:
                        modal_app.downstream_remote.local(2, 8, budget, str(checkpoint), 1, 'standard', 'all', True)
                        self.assertEqual([c.args[0] for c in train.call_args_list], ['probe', 'finetune', 'baseline'])
                        self.assertEqual([c.args[3] for c in train.call_args_list], [budget]*3)
                        self.assertEqual([c.args[4] for c in train.call_args_list], [str(checkpoint), str(checkpoint), ''])
                    with patch.object(modal_app.train_remote, 'local', side_effect=RuntimeError('training failed')):
                        with self.assertRaisesRegex(RuntimeError, 'training failed'):
                            modal_app.downstream_remote.local(2, 8, budget, str(checkpoint), 1, 'standard', 'all', True)
