import math
import unittest
import torch
from src.models import SimCLR, Classifier, nt_xent, parameter_counts


class ModelTests(unittest.TestCase):
    def test_shapes_gradients_update(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        model = SimCLR()
        x = torch.randn(4, 3, 64, 64)
        h, z = model(x)
        self.assertEqual(h.shape, (4, 512))
        self.assertEqual(z.shape, (4, 128))
        torch.testing.assert_close(z.norm(dim=1), torch.ones(4))
        before = model.projector[0].weight.detach().clone()
        opt = torch.optim.Adam(model.parameters())
        loss = nt_xent(z[:2], z[2:])
        loss.backward()
        for p in model.parameters():
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())
        opt.step()
        self.assertFalse(torch.equal(before, model.projector[0].weight))
        self.assertEqual(parameter_counts(model)['total'], 11504832)

    def test_loss_reference(self):
        torch.manual_seed(42)
        a, b = torch.randn(3, 8), torch.randn(3, 8)
        z = torch.nn.functional.normalize(torch.cat((a, b)), dim=1)
        reference = torch.stack([
            -(z[i] @ z[(i+3)%6])/0.5 + torch.logsumexp(
                torch.stack([z[i] @ z[j]/0.5 for j in range(6) if j != i]), 0)
            for i in range(6)
        ]).mean()
        torch.testing.assert_close(nt_xent(a, b), reference)
        self.assertAlmostEqual(nt_xent(torch.ones(3,8), torch.ones(3,8)).item(), math.log(5), places=6)
        self.assertLess(nt_xent(torch.eye(3), torch.eye(3)), nt_xent(torch.eye(3), torch.eye(3).roll(1,0)))
        for t in (0, -1, float('nan')):
            with self.assertRaises(ValueError): nt_xent(a,b,t)

    def test_probe_freezes_buffers_and_checkpoint(self):
        model = Classifier(frozen=True).train()
        state = {k:v.clone() for k,v in model.encoder.state_dict().items()}
        opt = torch.optim.Adam(model.head.parameters())
        x = torch.randn(2,3,64,64)
        torch.nn.functional.cross_entropy(model(x), torch.tensor([0,1])).backward()
        opt.step()
        for k,v in model.encoder.state_dict().items(): torch.testing.assert_close(v,state[k])
        self.assertTrue(all(p.grad is None for p in model.encoder.parameters()))
        self.assertEqual(parameter_counts(model)['trainable'], 5130)
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'model.pt'
            torch.save(model.state_dict(), path)
            restored = Classifier(True)
            restored.load_state_dict(torch.load(path, weights_only=True))
            torch.testing.assert_close(model.eval()(x), restored.eval()(x))

if __name__ == '__main__': unittest.main()
