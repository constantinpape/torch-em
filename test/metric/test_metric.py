import unittest
import torch
import numpy as np

from scipy.ndimage import distance_transform_edt
from skimage.measure import label
from skimage.segmentation import watershed


class TestMetric(unittest.TestCase):
    batch_size = 2
    offsets = [[-1, 0], [0, -1], [-6, 0], [0, -6]]
    min_size = 20

    # TODO refactor and use in all tests
    def _make_gt(self, shape, with_mask=False):
        segs = []
        for _ in range(self.batch_size):
            seeds = np.random.rand(*shape)
            seeds = label(seeds > 0.99)
            hmap = distance_transform_edt(seeds == 0)
            if with_mask:
                mask = np.random.rand(*shape) > 0.5
                assert mask.shape == hmap.shape
            else:
                mask = None
            seg = watershed(hmap, markers=seeds, mask=mask)
            segs.append(seg[None, None])
        seg = np.concatenate(segs, axis=0)
        return torch.from_numpy(seg)

    def _test_metric(self, pred, gt, metric, upper_bound=None):
        score = metric(pred, gt)
        self.assertGreaterEqual(score, 0.0)
        if upper_bound is not None:
            self.assertLessEqual(score, upper_bound)

    def test_mws_rand(self):
        from torch_em.metric import MWSRandMetric
        affs = torch.from_numpy(np.random.rand(self.batch_size, 4, 128, 128))
        gt = self._make_gt((128, 128))
        metric = MWSRandMetric(self.offsets, min_seg_size=self.min_size)
        self._test_metric(affs, gt, metric, upper_bound=1.0)

    def test_mws_voi(self):
        from torch_em.metric import MWSVOIMetric
        affs = torch.from_numpy(np.random.rand(self.batch_size, 4, 128, 128))
        gt = self._make_gt((128, 128))
        metric = MWSVOIMetric(self.offsets, min_seg_size=self.min_size)
        self._test_metric(affs, gt, metric)

    def test_mws_iou(self):
        from torch_em.metric import MWSIOUMetric
        affs = torch.from_numpy(np.random.rand(self.batch_size, 5, 128, 128))
        gt = self._make_gt((128, 128), with_mask=True)
        metric = MWSIOUMetric(self.offsets, self.min_size)
        self._test_metric(affs, gt, metric, upper_bound=1.0)

    def test_multicut_rand(self):
        from torch_em.metric import MulticutRandMetric
        bd = torch.from_numpy(np.random.rand(self.batch_size, 1, 128, 128))
        gt = self._make_gt((128, 128))
        metric = MulticutRandMetric(self.min_size)
        self._test_metric(bd, gt, metric, upper_bound=1.0)

    def test_multicut_voi(self):
        from torch_em.metric import MulticutVOIMetric
        bd = torch.from_numpy(np.random.rand(self.batch_size, 1, 128, 128))
        gt = self._make_gt((128, 128))
        metric = MulticutVOIMetric(self.min_size)
        self._test_metric(bd, gt, metric)

    def test_embed_mws_iou(self):
        from torch_em.metric import EmbeddingMWSIOUMetric
        emebd = torch.from_numpy(np.random.rand(self.batch_size, 6, 128, 128))
        gt = self._make_gt((128, 128), with_mask=True)
        metric = EmbeddingMWSIOUMetric(delta=2.0, offsets=self.offsets, min_seg_size=self.min_size)
        self._test_metric(emebd, gt, metric, upper_bound=1.0)

    def test_embed_mws_rand(self):
        from torch_em.metric import EmbeddingMWSRandMetric
        emebd = torch.from_numpy(np.random.rand(self.batch_size, 6, 128, 128))
        gt = self._make_gt((128, 128))
        metric = EmbeddingMWSRandMetric(delta=2.0, offsets=self.offsets, min_seg_size=self.min_size)
        self._test_metric(emebd, gt, metric, upper_bound=1.0)

    def test_embed_mws_voi(self):
        from torch_em.metric import EmbeddingMWSVOIMetric
        embed = torch.from_numpy(np.random.rand(self.batch_size, 6, 128, 128))
        gt = self._make_gt((128, 128))
        metric = EmbeddingMWSVOIMetric(delta=2.0, offsets=self.offsets, min_seg_size=self.min_size)
        self._test_metric(embed, gt, metric)

    def test_hdbscan_iou(self):
        from torch_em.metric import HDBScanIOUMetric
        embed = torch.from_numpy(np.random.rand(self.batch_size, 6, 128, 128))
        gt = self._make_gt((128, 128), with_mask=True)
        metric = HDBScanIOUMetric(min_size=50, eps=1.0e-4)
        self._test_metric(embed, gt, metric, upper_bound=1.0)

    def test_hdbscan_rand(self):
        from torch_em.metric import HDBScanRandMetric
        embed = torch.from_numpy(np.random.rand(self.batch_size, 6, 128, 128))
        gt = self._make_gt((128, 128))
        metric = HDBScanRandMetric(min_size=50, eps=1.0e-4)
        self._test_metric(embed, gt, metric, upper_bound=1.0)

    def test_hdbscan_voi(self):
        from torch_em.metric import HDBScanVOIMetric
        embed = torch.from_numpy(np.random.rand(self.batch_size, 6, 128, 128))
        gt = self._make_gt((128, 128))
        metric = HDBScanVOIMetric(min_size=50, eps=1.0e-4)
        self._test_metric(embed, gt, metric)


if __name__ == "__main__":
    unittest.main()
