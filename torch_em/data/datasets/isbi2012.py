import torch_em
from .util import download_source, update_kwargs

ISBI_URL = 'https://oc.embl.de/index.php/s/h0TkwqxU0PJDdMd/download'
CHECKSUM = '0e10fe909a1243084d91773470856993b7d40126a12e85f0f1345a7a9e512f29'


def get_isbi_loader(path, patch_shape, download=False,
                    offsets=None, boundaries=False,
                    use_original_labels=False,
                    **kwargs):
    """
    """

    assert len(patch_shape) == 3
    download_source(path, ISBI_URL, download, CHECKSUM)
    ndim = 2 if patch_shape[0] == 1 else 3
    kwargs = update_kwargs(kwargs, 'patch_shape', patch_shape)
    kwargs = update_kwargs(kwargs, 'ndim', ndim)

    raw_key = 'raw'
    label_key = 'labels/membranes' if use_original_labels else 'labels/gt_segmentation'

    assert not ((offsets is not None) and boundaries)
    if offsets is not None:
        assert not use_original_labels
        # we add a binary target channel for foreground background segmentation
        label_transform = torch_em.transform.label.AffinityTransform(offsets=offsets,
                                                                     ignore_label=None,
                                                                     add_binary_target=False,
                                                                     add_mask=True)
        msg = "Offsets are passed, but 'label_transform2' is in the kwargs. It will be over-ridden."
        kwargs = update_kwargs(kwargs, 'label_transform2', label_transform, msg=msg)
    elif boundaries:
        assert not use_original_labels
        label_transform = torch_em.transform.label.BoundaryTransform()
        msg = "Boundaries is set to true, but 'label_transform' is in the kwargs. It will be over-ridden."
        kwargs = update_kwargs(kwargs, 'label_transform', label_transform, msg=msg)

    return torch_em.default_segmentation_loader(
        path, raw_key,
        path, label_key,
        **kwargs
    )
