import unittest
import torch


class TestLossWrapper(unittest.TestCase):
    def test_masking(self):
        from torch_em.loss import (ApplyAndRemoveMask,
                                   DiceLoss,
                                   LossWrapper)
        loss = LossWrapper(DiceLoss(),
                           transform=ApplyAndRemoveMask())

        shape = (1, 1, 128, 128)
        x = torch.rand(*shape)
        x.requires_grad = True
        x.retain_grad = True

        y = torch.rand(*shape)
        mask = torch.rand(*shape) > .5
        y = torch.cat([
            y, mask.to(dtype=y.dtype)
        ], dim=1)

        lval = loss(x, y)
        self.assertTrue(0. < lval.item() < 1.)
        lval.backward()

        grad = x.grad.numpy()
        mask = mask.numpy()
        # print((grad[mask] == 0).sum())
        self.assertFalse((grad[mask] == 0).all())
        # print((grad[~mask] == 0).sum())
        self.assertTrue((grad[~mask] == 0).all())


if __name__ == '__main__':
    unittest.main()
