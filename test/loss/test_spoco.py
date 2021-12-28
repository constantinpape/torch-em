import unittest
import numpy as np
import torch
from torch_em.util.test import make_gt


class TestSpoco(unittest.TestCase):
    def _test_spoco(self, aux_loss):
        from torch_em.loss import SPOCOLoss
        loss = SPOCOLoss(delta_var=0.75, delta_dist=2.0, aux_loss=aux_loss)
        input1 = torch.from_numpy(np.random.rand(2, 8, 64, 64))
        input1.requires_grad = True
        input1.retain_grad = True
        input2 = torch.from_numpy(np.random.rand(2, 8, 64, 64))
        target = make_gt((64, 64), n_batches=2, with_channels=True, dtype="int64")
        lval = loss((input1, input2), target)
        self.assertNotEqual(lval.item(), 0.0)

        lval.backward()
        grads = input1.grad
        self.assertEqual(grads.shape, input1.shape)
        self.assertFalse(np.allclose(grads.numpy(), 0))

    def test_spoco_loss_dice(self):
        self._test_spoco("dice")

    # def test_extended_contrastive_loss(self):
    #     from torch_em.loss.spoco_loss import ExtendedContrastiveLoss


if __name__ == "__main__":
    unittest.main()
