import os

import imageio
import h5py
from torch_em.data.datasets.isbi2012 import ISBI_URL, CHECKSUM
from torch_em.data.datasets.util import download_source

os.makedirs("./data", exist_ok=True)
download_source("data/isbi.h5", ISBI_URL, True, CHECKSUM)

with h5py.File("data/isbi.h5", "r") as f:
    image = f["raw"][0]
imageio.imwrite("data/input.tif", image)
