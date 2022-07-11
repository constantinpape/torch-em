import os
from glob import glob

import torch_em
from .util import download_source, update_kwargs, unzip

URLS = {
    "root": {
        "train": "https://files.de-1.osf.io/v1/resources/9x3g2/providers/osfstorage/?zip=",
        "val": "https://files.de-1.osf.io/v1/resources/vs6gb/providers/osfstorage/?zip=",
        "test": "https://files.de-1.osf.io/v1/resources/tn4xj/providers/osfstorage/?zip=",
    },
    "nuclei": {
        "train": "https://files.de-1.osf.io/v1/resources/thxzn/providers/osfstorage/?zip=",
    },
    "ovules": {
        "train": "https://files.de-1.osf.io/v1/resources/x9yns/providers/osfstorage/?zip=",
        "val": "https://files.de-1.osf.io/v1/resources/xp5uf/providers/osfstorage/?zip=",
        "test": "https://files.de-1.osf.io/v1/resources/8jz7e/providers/osfstorage/?zip=",
    }
}

# FIXME somehow the checksums are not reliably, this is quite worrying...
CHECKSUMS = {
    "root": {
        "train": None, "val": None, "test": None
        # "train": "f72e9525ff716ef14b70ab1318efd4bf303bbf9e0772bf2981a2db6e22a75794",
        # "val": "987280d9a56828c840e508422786431dcc3603e0ba4814aa06e7bf4424efcd9e",
        # "test": "ad71b8b9d20effba85fb5e1b42594ae35939d1a0cf905f3403789fc9e6afbc58",
    },
    "nuclei": {
        "train": None
        # "train": "9d19ddb61373e2a97effb6cf8bd8baae5f8a50f87024273070903ea8b1160396",
    },
    "ovules": {
        "train": None, "val": None, "test": None
        # "train": "70379673f1ab1866df6eb09d5ce11db7d3166d6d15b53a9c8b47376f04bae413",
        # "val": "872f516cb76879c30782d9a76d52df95236770a866f75365902c60c37b14fa36",
        # "test": "a7272f6ad1d765af6d121e20f436ac4f3609f1a90b1cb2346aa938d8c52800b9",
    }
}
# The resolution previous used for the resizing
# I have removed this feature since it was not reliable,
# but leaving this here for reference
# (also implementing resizing would be a good idea,
#  but more general and not for each dataset individually)
# NATIVE_RESOLUTION = (0.235, 0.075, 0.075)


def _require_plantseg_data(path, download, name, split):
    url = URLS[name][split]
    checksum = CHECKSUMS[name][split]
    os.makedirs(path, exist_ok=True)
    out_path = os.path.join(path, f"{name}_{split}")
    if os.path.exists(out_path):
        return out_path
    tmp_path = os.path.join(path, f"{name}_{split}.zip")
    download_source(tmp_path, url, download, checksum)
    unzip(tmp_path, out_path, remove=True)
    return out_path


# TODO add support for ignore label, key: "/label_with_ignore"
def get_plantseg_loader(
    path,
    name,
    split,
    patch_shape,
    download=False,
    offsets=None,
    boundaries=False,
    binary=False,
    ndim=3,
    **kwargs,
):
    assert len(patch_shape) == 3
    data_path = _require_plantseg_data(path, download, name, split)

    file_paths = glob(os.path.join(data_path, "*.h5"))
    file_paths.sort()

    assert not (offsets is not None and boundaries)
    if offsets is not None:
        # we add a binary target channel for foreground background segmentation
        label_transform = torch_em.transform.label.AffinityTransform(offsets=offsets,
                                                                     add_binary_target=binary,
                                                                     add_mask=True)
        msg = "Offsets are passed, but 'label_transform2' is in the kwargs. It will be over-ridden."
        kwargs = update_kwargs(kwargs, "label_transform2", label_transform, msg=msg)
    elif boundaries:
        label_transform = torch_em.transform.label.BoundaryTransform(add_binary_target=binary)
        msg = "Boundaries is set to true, but 'label_transform' is in the kwargs. It will be over-ridden."
        kwargs = update_kwargs(kwargs, "label_transform", label_transform, msg=msg)
    elif binary:
        label_transform = torch_em.transform.label.labels_to_binary
        msg = "Binary is set to true, but 'label_transform' is in the kwargs. It will be over-ridden."
        kwargs = update_kwargs(kwargs, "label_transform", label_transform, msg=msg)

    kwargs = update_kwargs(kwargs, "patch_shape", patch_shape)
    kwargs = update_kwargs(kwargs, "ndim", ndim)

    raw_key, label_key = "raw", "label"
    return torch_em.default_segmentation_loader(
        file_paths, raw_key,
        file_paths, label_key,
        **kwargs
    )
