import napari
from .util import ensure_array


def check_loader(loader, n_samples):
    for ii, (x, y) in enumerate(loader):
        if ii >= n_samples:
            break

        x = ensure_array(x).squeeze(0)
        y = ensure_array(y).squeeze(0)
        with napari.gui_qt():
            v = napari.Viewer()
            v.add_image(x)
            v.add_image(y)
