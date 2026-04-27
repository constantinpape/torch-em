import unittest
import numpy as np

class TestclDiceMetric(unittest.TestCase):

    def _test_perfect_overlap(self, skeletonize_method):
        from torch_em.metric.cldice import clDice

        shape = (32, 32)

         # 4-pixel wide overlapping lines
        x = np.zeros(shape)
        x[14:18, :] = 1.0
        y = np.zeros(shape)
        y[14:18, :] = 1.0

        score = clDice(x, y, skeletonize_method=skeletonize_method)
        self.assertAlmostEqual(score, 1.0, places=1)

    def _test_no_overlap(self, skeletonize_method):
        from torch_em.metric.cldice import clDice

        shape = (32, 32)

        # 4-pixel wide non-overlapping lines
        x = np.zeros(shape)
        x[4:8, :] = 1.0
        y = np.zeros(shape)
        y[24:28, :] = 1.0

        score = clDice(x, y, skeletonize_method=skeletonize_method)
        self.assertAlmostEqual(score, 0.0, places=1)
    
    def test_perfect_overlap_skimage(self):
        self._test_perfect_overlap("skimage")

    def test_no_overlap_skimage(self):
        self._test_no_overlap("skimage")

    def test_perfect_overlap_soft(self):
        self._test_perfect_overlap("soft")

    def test_no_overlap_soft(self):
        self._test_no_overlap("soft")

if __name__ == '__main__':
    unittest.main()