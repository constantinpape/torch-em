import os
import napari
import z5py


def view_result(sample, checkpoint_name):

    halo = [25, 512, 512]
    path = f'./data/{sample}.n5'
    with z5py.File(path, 'r') as f:
        ds = f['raw']
        bb = tuple(slice(max(sh // 2 - ha, 0),
                         min(sh // 2 + ha, sh))
                   for sh, ha in zip(ds.shape, halo))
        raw = ds[bb]

        ds = f['labels']
        labels = ds[bb]

        prefix = f'predictions/{checkpoint_name}/'
        fg_key = prefix + 'foreground'
        if fg_key in f:
            ds = f[fg_key]
            fg = ds[bb]
        else:
            fg = None

        bd_key = prefix + 'boundaries'
        aff_key = prefix + 'affinities'
        if bd_key in f:
            ds = f[bd_key]
            boundaries = ds[bb]
        elif aff_key in f:
            ds = f[aff_key]
            bb_affs = (slice(None),) + bb
            boundaries = ds[bb_affs]
        else:
            boundaries = None

        prefix = f'segmentation/{checkpoint_name}'

        # ws_key = prefix + '/watershed'
        # if ws_key in f:
        #     ds = f[ws_key]
        #     ws = ds[bb]
        # else:
        #     ws = None
        ws = None

        mc_key = prefix + '/multicut_postprocessed'
        if mc_key in f:
            ds = f[mc_key]
            mc = ds[bb]
        else:
            mc = None

        mws_key = prefix + '/mutex_watershed_postprocessed'
        if mws_key in f:
            ds = f[mws_key]
            mws = ds[bb]
        else:
            mws = None

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(raw)
        if fg is not None:
            viewer.add_image(fg)
        if boundaries is not None:
            viewer.add_image(boundaries)
        if ws is not None:
            viewer.add_labels(ws)
        if mc is not None:
            viewer.add_labels(mc)
        if mws is not None:
            viewer.add_labels(mws)
        viewer.add_labels(labels)


def view_results(samples, checkpoint):
    checkpoint_name = os.path.split(checkpoint)[1]
    for sample in samples:
        view_result(sample, checkpoint_name)


if __name__ == '__main__':
    view_result('human_small', 'affinity_model_default_human_rat')
