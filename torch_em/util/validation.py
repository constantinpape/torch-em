import imageio
import numpy as np

from elf.io import open_file
from elf.util import normalize_index

from ..data import ConcatDataset, ImageCollectionDataset, SegmentationDataset
from .modelzoo import _get_trainer, _get_normalizer
from .prediction import predict_with_halo

try:
    import napari
except ImportError:
    napari = None

# TODO implement prefab metrics


class SampleGenerator:
    def __init__(self, trainer, max_samples, need_gt, n_threads):
        self.need_gt = need_gt
        self.n_threads = n_threads

        dataset = trainer.val_loader.dataset
        self.ndim = dataset.ndim

        (n_samples, load_2d_from_3d, rois,
         raw_paths, raw_key,
         label_paths, label_key) = self.paths_from_ds(dataset)

        if max_samples is None:
            self.n_samples = n_samples
        else:
            self.n_samples = min(max_samples, n_samples)
        self.load_2d_from_3d = load_2d_from_3d
        self.rois = rois
        self.raw_paths, self.raw_key = raw_paths, raw_key
        self.label_paths, self.label_key = label_paths, label_key

        if self.load_2d_from_3d:
            shapes = [
                open_file(rp, 'r')[self.raw_key].shape if roi is None else tuple(r.stop - r.start for r in roi)
                for rp, roi in zip(self.raw_paths, self.rois)
            ]
            lens = [shape[0] for shape in shapes]
            self.offsets = np.cumsum(lens)

    def paths_from_ds(self, dataset):
        if isinstance(dataset, ConcatDataset):
            (n_samples, load_2d_from_3d, rois,
             raw_paths, raw_key,
             label_paths, label_key) = self.paths_from_ds(dataset[0])

            for ds in dataset[1:]:
                ns, l2d3d, bb, rp, rk, lp, lk = self.paths_from_ds(ds)
                assert rk == raw_key
                assert lk == label_key
                assert l2d3d == load_2d_from_3d
                raw_paths.extend(rp)
                label_paths.extend(lp)
                rois.append(bb)
                n_samples += ns

        elif isinstance(dataset, ImageCollectionDataset):
            raw_paths, label_paths = dataset.raw_images, dataset.label_images
            raw_key, label_key = None, None
            n_samples = len(raw_paths)
            load_2d_from_3d = False
            rois = [None] * n_samples

        elif isinstance(dataset, SegmentationDataset):
            raw_paths, label_paths = [dataset.raw_path], [dataset.label_path]
            raw_key, label_key = dataset.raw_key, dataset.label_key
            shape = open_file(raw_paths[0], 'r')[raw_key].shape

            roi = dataset.roi
            if roi is not None:
                roi = normalize_index(roi, shape)
                shape = tuple(r.stop - r.start for r in roi)

            if self.ndim == len(shape):
                n_samples = len(raw_paths)
                load_2d_from_3d = False
            elif self.ndim == 2 and len(shape) == 3:
                n_samples = shape[0]
                load_2d_from_3d = True
            else:
                raise RuntimeError

        else:
            raise RuntimeError

        return (n_samples, load_2d_from_3d, rois,
                raw_paths, raw_key, label_paths, label_key)

    def load_data(self, path, key, roi, z):
        if key is None:
            assert roi is None and z is None
            return imageio.imread(path)

        bb = np.s_[:, :, :] if roi is None else roi
        if z is not None:
            bb[0] = z if roi is None else roi[0].start + z

        with open_file(path, 'r') as f:
            ds = f[key]
            ds.n_threads = self.n_threads
            data = ds[bb]
        return data

    def load_sample(self, sample_id):
        if self.load_2d_from_3d:
            ds_id = 0
            while True:
                if sample_id < self.offsets[ds_id]:
                    break
                ds_id += 1
            offset = self.offsets[ds_id - 1] if ds_id > 0 else 0
            z = sample_id - offset
        else:
            ds_id = sample_id
            z = None

        roi = self.rois[ds_id]
        raw = self.load_data(self.raw_paths[ds_id], self.raw_key, roi, z)
        if not self.need_gt:
            return raw
        gt = self.load_data(self.label_paths[ds_id], self.label_key, roi, z)
        return raw, gt

    def __iter__(self):
        for sample_id in range(self.n_samples):
            sample = self.load_sample(sample_id)
            yield sample


def _predict(model, raw, trainer, gpu_ids):
    # TODO refactor the normalizer attribute to make this easier
    normalizer = _get_normalizer(trainer)
    dataset = trainer.val_loader.dataset
    ndim = dataset.ndim
    if isinstance(dataset, ConcatDataset):
        block_shape = dataset[0].patch_shape
    else:
        block_shape = dataset.patch_shape

    if ndim == 2 and len(block_shape) == 3:
        block_shape = block_shape[1:]
    assert len(block_shape) == ndim
    # choose a small halo
    halo = [32, 32] if ndim == 2 else [8, 16, 16]

    gpu_ids = [int(gpu) if gpu != 'cpu' else gpu for gpu in gpu_ids]
    return predict_with_halo(
        raw, model, gpu_ids, block_shape, halo,
        preprocess=normalizer
    )


def _visualize(raw, prediction, ground_truth):
    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(raw)
        viewer.add_image(prediction)
        if ground_truth is not None:
            viewer.add_labels(ground_truth)


# TODO implement caching the predictions
def validate_checkpoint(
    checkpoint,
    gpu_ids,
    samples=None,
    max_samples=None,
    visualize=True,
    metrics=None,
    n_threads=None
):
    """Validate model for the given checkpoint visually and/or via metrics.
    """
    if visualize and napari is None:
        raise RuntimeError

    trainer = _get_trainer(checkpoint, device='cpu')
    n_threads = trainer.train_loader.num_workers if n_threads is None else n_threads
    model = trainer.model
    model.eval()

    need_gt = metrics is not None
    if samples is None:
        samples = SampleGenerator(trainer, max_samples, need_gt, n_threads)
    else:
        assert isinstance(samples, (list, tuple))
        if need_gt:
            assert all(len(sample, 2) for sample in samples)
        else:
            assert all(isinstance(sample, np.ndarray) for sample in samples)

    results = []
    for sample in samples:
        raw, gt = sample if need_gt else sample, None
        pred = _predict(model, raw, trainer, gpu_ids)
        if visualize:
            _visualize(raw, pred, gt)
        if metrics is not None:
            res = metrics(gt, pred)
            results.append(res)
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', required=True,
                        help="Path to the checkpoint")
    parser.add_argument('-g', '--gpus', type=str, nargs='+', required=True)
    parser.add_argument('-n', '--max_samples', type=int, default=None)
    parser.add_argument('-d', '--data', default=None)
    parser.add_argument('-k', '--key', default=None)
    parser.add_argument('-t', '--n_threads', type=int, default=None)

    args = parser.parse_args()
    # TODO implement loading data
    assert args.data is None
    validate_checkpoint(args.path, args.gpus,
                        max_samples=args.max_samples,
                        n_threads=args.n_threads)
