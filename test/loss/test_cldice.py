import unittest
import torch

class TestclDiceLoss(unittest.TestCase):
    def test_cldice_random(self):
        from torch_em.loss.cldice import CombinedclDiceLoss
        loss = CombinedclDiceLoss()

        shape = (1, 1, 32, 32)
        x = torch.rand(*shape)
        x.requires_grad = True
        x.retain_grad = True
        y = torch.rand(*shape)

        lval = loss(x, y)

        # loss should be be in range (0, 1)
        self.assertGreaterEqual(lval.item(), 0.0)
        self.assertLessEqual(lval.item(), 1.0)
        lval.backward()
        grads = x.grad
        self.assertEqual(grads.shape, x.shape)

    def test_cldice_perfect_overlap(self):
        from torch_em.loss.cldice import CombinedclDiceLoss
        loss = CombinedclDiceLoss()

        shape = (1, 1, 32, 32)

        # 4-pixel wide overlapping lines
        x = torch.zeros(*shape)
        x[0, 0, 14:18, :] = 1.0

        y = torch.zeros(*shape)
        y[0, 0, 14:18, :] = 1.0

        lval = loss(x, y)
        self.assertAlmostEqual(lval.item(), 0.0, places=1)

    def test_cldice_no_overlap(self):
        from torch_em.loss.cldice import CombinedclDiceLoss
        loss = CombinedclDiceLoss()

        shape = (1, 1, 32, 32)

        # 4-pixel wide non-overlapping lines
        x = torch.zeros(*shape)
        x[0, 0, 4:8, :] = 1.0
        
        y = torch.zeros(*shape)
        y[0, 0, 24:28, :] = 1.0

        lval = loss(x, y)
        self.assertAlmostEqual(lval.item(), 1.0, places=1)

    def test_cldice_invalid(self):
        from torch_em.loss.cldice import CombinedclDiceLoss
        loss = CombinedclDiceLoss()

        shape1 = (1, 1, 32, 32)
        shape2 = (1, 2, 32, 32)
        x = torch.rand(*shape1)
        y = torch.rand(*shape2)
        with self.assertRaises(ValueError):
            loss(x, y)

if __name__ == '__main__':
    unittest.main()