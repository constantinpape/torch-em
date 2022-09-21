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

    def test_ApplyMask_output_shape_crop(self):
        from torch_em.loss import ApplyMask

        # _crop batch_size=1
        shape = (1, 1, 10, 128, 128)
        p = torch.rand(*shape)
        t = torch.rand(*shape)
        m = torch.rand(*shape) > .5
        p_masked, t_masked = ApplyMask()(p, t, m)
        out_shape = (m.sum(), shape[1])
        self.assertTrue(p_masked.shape == out_shape)
        self.assertTrue(t_masked.shape == out_shape)

        # _crop batch_size>1
        shape = (5, 1, 10, 128, 128)
        p = torch.rand(*shape)
        t = torch.rand(*shape)
        m = torch.rand(*shape) > .5
        p_masked, t_masked = ApplyMask()(p, t, m)
        out_shape = (m.sum(), shape[1])
        self.assertTrue(p_masked.shape == out_shape)
        self.assertTrue(t_masked.shape == out_shape)

        # _crop n_channels>1
        shape = (1, 2, 10, 128, 128)
        p = torch.rand(*shape)
        t = torch.rand(*shape)
        m = torch.rand(*shape) > .5
        with self.assertRaises(ValueError):
            p_masked, t_masked = ApplyMask()(p, t, m)

        # _crop different shapes
        shape_pt = (5, 2, 10, 128, 128)
        p = torch.rand(*shape_pt)
        t = torch.rand(*shape_pt)
        shape_m = (5, 1, 10, 128, 128)
        m = torch.rand(*shape_m) > .5
        p_masked, t_masked = ApplyMask()(p, t, m)
        out_shape = (m.sum(), shape_pt[1])
        self.assertTrue(p_masked.shape == out_shape)
        self.assertTrue(t_masked.shape == out_shape)

    def test_ApplyMask_output_shape_multiply(self):
        from torch_em.loss import ApplyMask

        # _multiply
        shape = (2, 5, 10, 128, 128)
        p = torch.rand(*shape)
        t = torch.rand(*shape)
        m = torch.rand(*shape) > .5

        p_masked, t_masked = ApplyMask(masking_method="multiply")(p, t, m)
        self.assertTrue(p_masked.shape == shape)
        self.assertTrue(t_masked.shape == shape)


if __name__ == '__main__':
    unittest.main()
