import os
import warnings
from typing import Union, Optional

import torch
import torch.utils.data

from .util import ensure_array


def _check_plt(loader, n_samples, instance_labels, model=None, device=None, save_path=None):
    import matplotlib.pyplot as plt
    img_size = 5

    fig = None
    n_rows = None

    def to_index(ns, rid, sid):
        index = 1 + rid * ns + sid
        return index

    for ii, (x, y) in enumerate(loader):
        if ii >= n_samples:
            break

        if model is None:
            pred = None
        else:
            pred = model(x if device is None else x.to(device))
            pred = ensure_array(pred)[0]

        # cast the data to array and remove the batch axis / choose first sample in batch
        x = ensure_array(x)[0]
        y = ensure_array(y)[0]
        assert x.ndim == y.ndim
        if x.ndim == 4:  # 3d data (with channel axis)
            z_slice = x.shape[1] // 2
            warnings.warn(f"3d input data is not yet supported, will only show slice {z_slice} / {x.shape[1]}")
            x, y = x[:, z_slice], y[:, z_slice]
            if pred is not None:
                pred = pred[:, z_slice]

        if x.shape[0] > 1:
            warnings.warn(f"Multi-channel input data is not yet supported, will only show channel 0 / {x.shape[0]}")
        x = x[0]

        if pred is None:
            n_target_channels = y.shape[0]
        else:
            n_target_channels = pred.shape[0]
            y = y[:n_target_channels]
            assert y.shape[0] == n_target_channels

        if fig is None:
            n_rows = n_target_channels + 1 if pred is None else 2 * n_target_channels + 1
            fig = plt.figure(figsize=(n_samples*img_size, n_rows*img_size))

        ax = fig.add_subplot(n_rows, n_samples, to_index(n_samples, 0, ii))
        ax.imshow(x, interpolation="nearest", cmap="Greys_r", aspect="auto")

        for chan in range(n_target_channels):
            ax = fig.add_subplot(n_rows, n_samples, to_index(n_samples, 1 + chan, ii))
            if instance_labels:
                ax.imshow(y[chan].astype("uint32"), interpolation="nearest", aspect="auto")
            else:
                ax.imshow(y[chan], interpolation="nearest", cmap="Greys_r", aspect="auto")

        if pred is not None:
            for chan in range(n_target_channels):
                ax = fig.add_subplot(n_rows, n_samples, to_index(n_samples, 1 + n_target_channels + chan, ii))
                ax.imshow(pred[chan], interpolation="nearest", cmap="Greys_r", aspect="auto")

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
        plt.close()


def _check_napari(loader, n_samples, instance_labels, model=None, device=None, rgb=False):
    import napari

    for ii, sample in enumerate(loader):
        if ii >= n_samples:
            break

        try:
            x, y = sample
        except ValueError:
            x = sample
            y = None

        if model is None:
            pred = None
        else:
            pred = model(x if device is None else x.to(device))
            pred = ensure_array(pred)[0]

        x = ensure_array(x)[0]
        if rgb:
            assert x.shape[0] == 3
            x = x.transpose((1, 2, 0))

        v = napari.Viewer()
        v.add_image(x)
        if y is not None:
            y = ensure_array(y)[0]
            if instance_labels:
                v.add_labels(y.astype("uint32"))
            else:
                v.add_image(y)
        if pred is not None:
            v.add_image(pred)

        napari.run()


def check_trainer(
    trainer,
    n_samples: int,
    instance_labels: bool = False,
    split: str = "val",
    loader: Optional[torch.utils.data.DataLoader] = None,
    plt: bool = False,
):
    """Check a trainer visually.

    This function shows images and labels from the training or validation loader
    and predictions from the trainer's model for the images.
    The data will be plotted either with napari or matplotlib.

    Args:
        trainer: The trainer.
        n_samples: The number of samples to plot.
        instance_labels: Whether to visualize the label data as instances.
        split: Which split to use. This will determine which data loader is used for plotting.
            Can be one of "val" or "train".
        loader: An optional loader to use for getting the data. Will be used instead of the loader from the trainer.
        plt: Whether to plot the data with matplotlib instead of napari.
    """
    if loader is None:
        assert split in ("val", "train")
        loader = trainer.val_loader if split == "val" else trainer.train_loader
    with torch.no_grad():
        model = trainer.model
        model.eval()
        if plt:
            _check_plt(loader, n_samples, instance_labels, model=model, device=trainer.device)
        else:
            _check_napari(loader, n_samples, instance_labels, model=model, device=trainer.device)


def check_loader(
    loader: torch.utils.data.DataLoader,
    n_samples: int,
    instance_labels: bool = False,
    plt: bool = False,
    rgb: bool = False,
    save_path: Optional[Union[str, os.PathLike]] = None,
):
    """Check a loader visually.

    This function shows images and labels from the loader with napari or matplotlib.

    Args:
        loader: The data loader.
        n_samples: The number of samples to plot.
        instance_labels: Whether to visualize the label data as instances.
        plt: Whether to plot the data with matplotlib instead of napari.
        rgb: Whether the image data is rgb.
        save_path: Path for saving the images instead of showing them.
            This argument only has an effect if `plt=True`.
    """
    if plt:
        _check_plt(loader, n_samples, instance_labels, save_path=save_path)
    else:
        _check_napari(loader, n_samples, instance_labels, rgb=rgb)
