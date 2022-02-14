from .util import ensure_array


def check_loader(loader, n_samples, instance_labels=False, plt=False):
    if plt:
        import matplotlib.pyplot as plt
        from matplotlib import colors

        nrows = 2
        img_size = 5
        fig = plt.figure(figsize=(n_samples*img_size, nrows*img_size))

        for ii, (x, y) in enumerate(loader):
            if ii >= n_samples:
                break

            x = ensure_array(x).squeeze(0)
            y = ensure_array(y).squeeze(0)

            ax = fig.add_subplot(nrows, n_samples, ii+1)
            ax.imshow(x, interpolation='nearest', cmap='Greys', aspect='auto')

            ax = fig.add_subplot(nrows, n_samples, ii+n_samples+1)
            if instance_labels:
                y = y.astype('uint32')
                ax.imshow(y, interpolation='nearest', aspect='auto')
            else:
                ax.imshow(y, interpolation='nearest', cmap='Greys', aspect='auto')

    else:
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
