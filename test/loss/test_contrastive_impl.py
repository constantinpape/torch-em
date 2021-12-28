import unittest

import numpy as np
import torch
import vigra

from torch_em.loss.contrastive_impl import (_compute_cluster_means,
                                            _compute_cluster_means_scatter,
                                            expand_as_one_hot)

try:
    import torch_scatter
except ImportError:
    torch_scatter = None


class TestContrastiveImpl(unittest.TestCase):
    def generate_data(self, device):
        target_shape = (1, 1, 32, 32)
        pred_shape = (1, 8, 32, 32)
        x = torch.rand(*pred_shape).to(device)
        y = torch.randint(low=0, high=5, size=target_shape).to(device)
        return x, y

    def _test_mean_embedding(self, device, impl):
        x, y = self.generate_data(device)
        n_instances = int(torch.unique(y).shape[0])

        exp_shape = (n_instances, 8)
        mean_emb = impl(x, y, n_instances).detach().cpu().numpy()
        self.assertEqual(mean_emb.shape, exp_shape)

        x = x.detach().cpu().numpy().astype("float32")[0]
        y = y.detach().cpu().numpy().astype("uint32")[0, 0]
        exp = np.concatenate([
            vigra.analysis.extractRegionFeatures(chan, y, features=["mean"])["mean"][None] for chan in x
        ], axis=0).T
        self.assertEqual(exp.shape, exp_shape)

        self.assertTrue(np.allclose(mean_emb, exp))

    def _compute_mean_expand(self, x, y, n_instances,
                             return_embed=False, squeeze=True):
        y1 = expand_as_one_hot(y[0], n_instances)
        mean_emb, embed_per_instance = _compute_cluster_means(x, y1, ndim=2)
        if squeeze:
            mean_emb = mean_emb.squeeze()
        if return_embed:
            return mean_emb, embed_per_instance
        else:
            return mean_emb

    def test_mean_embedding_expand(self):
        _impl = self._compute_mean_expand
        self._test_mean_embedding(torch.device("cpu"), _impl)
        if torch.cuda.is_available():
            self._test_mean_embedding(torch.device("cuda"), _impl)

    def _compute_mean_scatter(self, x, y, n_instances):
        return _compute_cluster_means_scatter(x, y, ndim=2)

    @unittest.skipIf(torch_scatter is None, "need torch_scatter")
    def test_mean_embedding_scatter(self):
        _impl = self._compute_mean_scatter
        self._test_mean_embedding(torch.device("cpu"), _impl)
        if torch.cuda.is_available():
            self._test_mean_embedding(torch.device("cuda"), _impl)

    @unittest.skipIf(torch_scatter is None, "need torch_scatter")
    def test_distance_term(self):
        from torch_em.loss.contrastive_impl import (_compute_distance_term,
                                                    _compute_distance_term_scatter)

        def _test_distance(device):
            ndim = 2
            norm = "fro"
            delta = 2.0

            x, y = self.generate_data(device)
            n_instances = int(torch.unique(y).shape[0])

            # distance term expand
            mean_expand = self._compute_mean_expand(x, y, n_instances, squeeze=False)
            dist_expand = _compute_distance_term(mean_expand, n_instances,
                                                 ndim, norm, delta)

            # distance term scatter
            mean_scatter = self._compute_mean_scatter(x, y, n_instances)
            dist_scatter = _compute_distance_term_scatter(mean_scatter, norm, delta)

            self.assertAlmostEqual(dist_expand.item(), dist_scatter.item(), places=5)

        _test_distance(torch.device("cpu"))
        if torch.cuda.is_available():
            _test_distance(torch.device("cuda"))

    @unittest.skipIf(torch_scatter is None, "need torch_scatter")
    def test_variance_term(self):
        from torch_em.loss.contrastive_impl import (_compute_variance_term,
                                                    _compute_variance_term_scatter)

        def _test_variance(device):
            ndim = 2
            norm = "fro"
            delta = 1.0

            x, y = self.generate_data(device)
            instances, sizes = torch.unique(y, return_counts=True)
            n_instances = int(instances.shape[0])

            # variance term expand
            mean_expand, embed_per_instance = self._compute_mean_expand(x, y, n_instances,
                                                                        return_embed=True, squeeze=False)
            target = expand_as_one_hot(y[0], n_instances)
            var_expand = _compute_variance_term(mean_expand, embed_per_instance,
                                                target, ndim, norm, delta)

            # variance term scatter
            mean_scatter = self._compute_mean_scatter(x, y, n_instances)
            var_scatter = _compute_variance_term_scatter(mean_scatter, x, y[0],
                                                         norm, delta, sizes)

            self.assertAlmostEqual(var_expand.item(), var_scatter.item(), places=6)

        _test_variance(torch.device("cpu"))
        if torch.cuda.is_available():
            _test_variance(torch.device("cuda"))


if __name__ == "__main__":
    unittest.main()
