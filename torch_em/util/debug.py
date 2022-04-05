import warnings
from .util import ensure_array


def _check_plt(loader, n_samples, instance_labels):
    import matplotlib.pyplot as plt
    img_size = 5

    fig = None
    nrows = None

    for ii, (x, y) in enumerate(loader):
        if ii >= n_samples:
            break

        # cast the data to array and remove the batch axis / choose first sample in batch
        x = ensure_array(x)[0]
        y = ensure_array(y)[0]
        assert x.ndim == y.ndim
        if x.ndim == 4:  # 3d data (with channel axis)
            z_slice = x.shape[1] // 2
            warnings.warn(f"3d input data is not yet supported, will only show slice {z_slice} / {x.shape[1]}")
            x, y = x[:, z_slice], y[:, z_slice]

        if x.shape[0] > 1:
            warnings.warn(f"Multi-channel input data is not yet supported, will only show channel 0 / {x.shape[0]}")
        x = x[0]
        n_target_channels = y.shape[0]

        if fig is None:
            nrows = n_target_channels + 1
            fig = plt.figure(figsize=(n_samples*img_size, nrows*img_size))

        ax = fig.add_subplot(nrows, n_samples, ii+1)
        ax.imshow(x, interpolation="nearest", cmap="Greys_r", aspect="auto")

        for chan in range(n_target_channels):
            ax = fig.add_subplot(nrows, n_samples, ii + (chan + 1) * n_samples + 1)
            if instance_labels:
                ax.imshow(y[chan].astype("uint32"), interpolation="nearest", aspect="auto")
            else:
                ax.imshow(y[chan], interpolation="nearest", cmap="Greys_r", aspect="auto")

    plt.show()


def _check_napari(loader, n_samples, instance_labels):
    import napari

    for ii, (x, y) in enumerate(loader):
        if ii >= n_samples:
            break

        x = ensure_array(x).squeeze(0)
        y = ensure_array(y).squeeze(0)

        v = napari.Viewer()
        v.add_image(x)
        if instance_labels:
            v.add_labels(y.astype("uint32"))
        else:
            v.add_image(y)
        napari.run()


def check_loader(loader, n_samples, instance_labels=False, plt=False):
    if plt:
        _check_plt(loader, n_samples, instance_labels)
    else:
        _check_napari(loader, n_samples, instance_labels)
