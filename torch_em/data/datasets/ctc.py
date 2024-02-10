import os
from glob import glob
from shutil import copyfile

import torch_em
from . import util


CTC_URLS = {
    "BF-C2DL-HSC": "http://data.celltrackingchallenge.net/training-datasets/BF-C2DL-HSC.zip",
    "BF-C2DL-MuSC": "http://data.celltrackingchallenge.net/training-datasets/BF-C2DL-MuSC.zip",
    "DIC-C2DH-HeLa": "http://data.celltrackingchallenge.net/training-datasets/DIC-C2DH-HeLa.zip",
    "Fluo-C2DL-Huh7": "http://data.celltrackingchallenge.net/training-datasets/Fluo-C2DL-Huh7.zip",
    "Fluo-C2DL-MSC": "http://data.celltrackingchallenge.net/training-datasets/Fluo-C2DL-MSC.zip",
    "Fluo-N2DH-GOWT1": "http://data.celltrackingchallenge.net/training-datasets/Fluo-N2DH-GOWT1.zip",
    "Fluo-N2DH-SIM+": "http://data.celltrackingchallenge.net/training-datasets/Fluo-N2DH-SIM+.zip",
    "Fluo-N2DL-HeLa": "http://data.celltrackingchallenge.net/training-datasets/Fluo-N2DL-HeLa.zip",
    "PhC-C2DH-U373": "http://data.celltrackingchallenge.net/training-datasets/PhC-C2DH-U373.zip",
    "PhC-C2DL-PSC": "http://data.celltrackingchallenge.net/training-datasets/PhC-C2DL-PSC.zip",
}
CTC_CHECKSUMS = {
    "BF-C2DL-HSC": "0aa68ec37a9b06e72a5dfa07d809f56e1775157fb674bb75ff904936149657b1",
    "BF-C2DL-MuSC": "ca72b59042809120578a198ba236e5ed3504dd6a122ef969428b7c64f0a5e67d",
    "DIC-C2DH-HeLa": "832fed2d05bb7488cf9c51a2994b75f8f3f53b3c3098856211f2d39023c34e1a",
    "Fluo-C2DL-Huh7": "1912658c1b3d8b38b314eb658b559e7b39c256917150e9b3dd8bfdc77347617d",
    "Fluo-C2DL-MSC": "a083521f0cb673ae02d4957c5e6580c2e021943ef88101f6a2f61b944d671af2",
    "Fluo-N2DH-GOWT1": "1a7bd9a7d1d10c4122c7782427b437246fb69cc3322a975485c04e206f64fc2c",
    "Fluo-N2DH-SIM+": "3e809148c87ace80c72f563b56c35e0d9448dcdeb461a09c83f61e93f5e40ec8",
    "Fluo-N2DL-HeLa": "35dd99d58e071aba0b03880128d920bd1c063783cc280f9531fbdc5be614c82e",
    "PhC-C2DH-U373": "b18185c18fce54e8eeb93e4bbb9b201d757add9409bbf2283b8114185a11bc9e",
    "PhC-C2DL-PSC": "9d54bb8febc8798934a21bf92e05d92f5e8557c87e28834b2832591cdda78422",

}


def _require_ctc_dataset(path, dataset_name, download):
    dataset_names = list(CTC_URLS.keys())
    if dataset_name not in dataset_names:
        raise ValueError(f"Inalid dataset: {dataset_name}, choose one of {dataset_names}.")

    data_path = os.path.join(path, dataset_name)

    if os.path.exists(data_path):
        return data_path

    os.makedirs(data_path)
    url, checksum = CTC_URLS[dataset_name], CTC_CHECKSUMS[dataset_name]
    zip_path = os.path.join(path, f"{dataset_name}.zip")
    util.download_source(zip_path, url, download, checksum=checksum)
    util.unzip(zip_path, path, remove=True)

    return data_path


def _require_gt_images(data_path, splits):
    image_paths, label_paths = [], []

    if isinstance(splits, str):
        splits = [splits]

    for split in splits:
        image_folder = os.path.join(data_path, split)
        assert os.path.join(image_folder), f"Cannot find split, {split} in {data_path}."

        label_folder = os.path.join(data_path, f"{split}_GT", "SEG")

        # copy over the images corresponding to the labeled frames
        label_image_folder = os.path.join(data_path, f"{split}_GT", "IM")
        os.makedirs(label_image_folder, exist_ok=True)

        this_label_paths = glob(os.path.join(label_folder, "*.tif"))
        for label_path in this_label_paths:
            fname = os.path.basename(label_path)
            image_label_path = os.path.join(label_image_folder, fname)
            if not os.path.exists(image_label_path):
                im_name = "t" + fname.lstrip("main_seg")
                image_path = os.path.join(image_folder, im_name)
                assert os.path.join(image_path), image_path
                copyfile(image_path, image_label_path)

        image_paths.append(label_image_folder)
        label_paths.append(label_folder)

    return image_paths, label_paths


def get_ctc_segmentation_dataset(
    path,
    dataset_name,
    patch_shape,
    split=None,
    download=False,
    **kwargs,
):
    """Dataset for the cell tracking challenge segmentation data.

    This dataset provides access to the 2d segmentation datsets of the
    cell tracking challenge. If you use this data in your research please cite
    https://doi.org/10.1038/nmeth.4473
    """
    data_path = _require_ctc_dataset(path, dataset_name, download)

    if split is None:
        splits = glob(os.path.join(data_path, "*_GT"))
        splits = [os.path.basename(split) for split in splits]
        splits = [split.rstrip("_GT") for split in splits]
    else:
        splits = split

    image_path, label_path = _require_gt_images(data_path, splits)

    kwargs = util.update_kwargs(kwargs, "ndim", 2)
    return torch_em.default_segmentation_dataset(
        image_path, "*.tif", label_path, "*.tif", patch_shape, is_seg_dataset=True, **kwargs
    )


def get_ctc_segmentation_loader(
    path,
    dataset_name,
    patch_shape,
    batch_size,
    split=None,
    download=False,
    **kwargs,
):
    """Dataloader for cell tracking challenge segmentation data.
    See 'get_ctc_segmentation_dataset' for details.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(
        torch_em.default_segmentation_dataset, **kwargs
    )
    dataset = get_ctc_segmentation_dataset(
        path, dataset_name, patch_shape, split=split, download=download, **ds_kwargs,
    )
    loader = torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
    return loader
