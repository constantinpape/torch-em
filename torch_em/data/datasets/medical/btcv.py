# source: https://www.synapse.org/#!Synapse:syn3193805/wiki/89480

# how to obtain the dataset?
#   step 1: join "Multi-Atlas Labeling Beyond the Cranial Vault - Workshop and Challenge"
#       - link: https://www.synapse.org/#!Synapse:syn3193805
#   step 2: go to "Files" ->
#       - "Abdomen" -> "RawData.zip" to obtain all the abdominal CT scans
#       - "Cervix" -> "CervixRawData.zip" to obtain all the cervical CT scans
#   step 3: provide the path to the zipped file(s) to the respective datasets that takes care of it.


import os
from pathlib import Path

from .. import util


_PATHS = {
    "RawData.zip": "Abdomen",
    "CervixRawData.zip": "Cervix"
}


def unzip_inputs(zip_path):
    _p = Path(zip_path)
    dir_path = _p.parent / _PATHS[_p.stem]
    os.makedirs(dir_path, exist_ok=True)

    # unzipping the objects to the desired directory
    util.unzip(zip_path, dir_path, remove=False)


def assort_btcv_dataset(path):
    for zipfile in _PATHS.keys():
        # if the directories exists already, we assume that the dataset has been prepared
        if os.path.exists(os.path.join(path, _PATHS[zipfile])):
            return

        # let's check if the zip files have been downloaded
        zip_path = os.path.join(path, zipfile)
        assert os.path.exists(zip_path), f"It seems that the zip file for {_PATHS[zipfile]} CT scans is missing."

        unzip_inputs(zip_path)


def get_btcv_dataset(path, download=False):
    if download:
        raise NotImplementedError("The BTCV dataset cannot be automatically download from `torch_em`."\
                                  "Please download the dataset and provide the directory where zip files are stored.")
    else:
        assort_btcv_dataset(path)

    # implement the dataset from elf
    raise NotImplementedError


def get_btcv_loader(path):
    # get the dataset from elf, pass it to dataloader
    # make a split based on organs - by default uses ct scans from both organs
    # if specified, can use from either abdomen or cervix
    # NOTE: logic to resample the inputs
    # NOTE: logic for normalization for respective modalities
    #       - easiest: follow logic from nnunet (raw-trafo now in torch_em)
    raise NotImplementedError
