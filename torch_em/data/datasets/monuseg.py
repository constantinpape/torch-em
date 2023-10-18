import os
import shutil
import numpy as np
from tqdm import tqdm
from glob import glob
from xml.dom import minidom

import imageio.v2 as imageio
from skimage.draw import polygon

import torch_em
from torch_em.data.datasets import util


URL = "https://drive.google.com/uc?export=download&id=1ZgqFJomqQGNnsx7w7QBzQQMVA16lbVCA"
# TODO: add labeled test set (monuseg) - https://drive.google.com/file/d/1NKkSQ5T0ZNQ8aUhh0a8Dt2YKYCQXIViw

CHECKSUM = "25d3d3185bb2970b397cafa72eb664c9b4d24294aee382e7e3df9885affce742"


# TODO separate via organ
def _download_monuseg(path, download):
    # check if we have extracted the images and labels already
    im_path = os.path.join(path, "images")
    label_path = os.path.join(path, "labels")
    if os.path.exists(im_path) and os.path.exists(label_path):
        return

    os.makedirs(path, exist_ok=True)
    zip_path = os.path.join(path, "monuseg.zip")
    util.download_source_gdrive(zip_path, URL, download=download, checksum=CHECKSUM)

    _process_monuseg(path)


def generate_labeled_array(shape, xml_file):
    """Function taken from: https://github.com/rshwndsz/hover-net/blob/master/lightning_hovernet.ipynb

    Given image shape and path to annotations (xml file), generatebit mask with the region inside a contour being white
        shape: The image shape on which bit mask will be made
        xml_file: path relative to the current working directory where the xml file is present

    Returns:
        An image of given shape with region inside contour being white..
    """
    # DOM object created by the minidom parser
    xDoc = minidom.parse(xml_file)

    # List of all Region tags
    regions = xDoc.getElementsByTagName('Region')

    # List which will store the vertices for each region
    xy = []
    for region in regions:
        # Loading all the vertices in the region
        vertices = region.getElementsByTagName('Vertex')

        # The vertices of a region will be stored in a array
        vw = np.zeros((len(vertices), 2))

        for index, vertex in enumerate(vertices):
            # Storing the values of x and y coordinate after conversion
            vw[index][0] = float(vertex.getAttribute('X'))
            vw[index][1] = float(vertex.getAttribute('Y'))

        # Append the vertices of a region
        xy.append(np.int32(vw))

    # Creating a completely black image
    mask = np.zeros(shape, np.float32)

    for i, contour in enumerate(xy):
        r, c = polygon(np.array(contour)[:, 1], np.array(contour)[:, 0], shape=shape)
        mask[r, c] = i
    return mask


def _process_monuseg(path):
    util.unzip(os.path.join(path, "monuseg.zip"), path)

    # assorting the images into expected dir;
    # converting the label xml files to numpy arrays (of same dimension as input images) in the expected dir
    root_img_save_dir = os.path.join(path, "images")
    root_label_save_dir = os.path.join(path, "labels")

    os.makedirs(root_img_save_dir, exist_ok=True)
    os.makedirs(root_label_save_dir, exist_ok=True)

    all_img_dir = sorted(glob(os.path.join(path, "*", "Tissue*", "*")))
    all_xml_label_dir = sorted(glob(os.path.join(path, "*", "Annotations", "*")))
    assert len(all_img_dir) == len(all_xml_label_dir)

    for img_path, xml_label_path in tqdm(zip(all_img_dir, all_xml_label_dir),
                                         desc="Converting inputs to the expected format", 
                                         total=len(all_img_dir)):
        desired_label_shape = imageio.imread(img_path).shape[:-1]

        img_id = os.path.split(img_path)[-1]
        dst = os.path.join(root_img_save_dir, img_id)
        shutil.move(src=img_path, dst=dst)

        _label = generate_labeled_array(shape=desired_label_shape, xml_file=xml_label_path)
        _fileid = img_id.split(".")[0]
        imageio.imwrite(os.path.join(root_label_save_dir, f"{_fileid}.tif"), _label)

    shutil.rmtree(glob(os.path.join(path, "MoNuSeg*"))[0])
    shutil.rmtree(glob(os.path.join(path, "__MACOSX"))[0])


def get_monuseg_dataset(
    path, patch_shape, download=False, offsets=None, boundaries=False, binary=False, **kwargs
):
    _download_monuseg(path, download)

    image_path = os.path.join(path, "images")
    label_path = os.path.join(path, "labels")

    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=True, binary=binary, boundaries=boundaries, offsets=offsets
    )
    return torch_em.default_segmentation_dataset(
        image_path, "*.tif", label_path, "*.tif", patch_shape, is_seg_dataset=False, **kwargs
    )


# TODO implement selecting organ
def get_monuseg_loader(
    path, patch_shape, batch_size, download=False, offsets=None, boundaries=False, binary=False, **kwargs
):
    ds_kwargs, loader_kwargs = util.split_kwargs(
        torch_em.default_segmentation_dataset, **kwargs
    )
    dataset = get_monuseg_dataset(
        path, patch_shape, download=download,
        offsets=offsets, boundaries=boundaries, binary=binary, **ds_kwargs
    )
    loader = torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
    return loader


def main():
    path = "/scratch/usr/nimanwai/data/monuseg/"
    patch_shape = (512, 512)

    loader = get_monuseg_loader(
        path=path,
        patch_shape=patch_shape,
        batch_size=2,
        download=True
    )

    print("Length of loader: ", len(loader))

    breakpoint()


if __name__ == "__main__":
    main()
