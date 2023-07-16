import napari
from elf.io import open_file


# TODO also load the raw data
def check_result(path, with_affinities=False):
    with open_file(path, 'r') as f:
        ds = f['prediction/foreground']
        ds.n_threads = 8
        fg = ds[:]

        if with_affinities:
            ds = f['prediction/affinities']
            ds.n_threads = 8
            affs = ds[:]
        else:
            affs = None

        ds = f['segmentation/mws']
        ds.n_threads = 8
        seg = ds[:]

    with napari.gui_qt():
        v = napari.Viewer()
        v.add_image(fg)
        if affs is not None:
            v.add_image(affs)
        v.add_labels(seg)


if __name__ == '__main__':
    check_result('./prediction.n5')
