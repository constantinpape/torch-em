import os
import requests
from urllib.parse import urljoin
from typing import Union, Tuple

import pandas as pd

from .. import util

from tcia_utils import nbia


BASE_URL = "https://wiki.cancerimagingarchive.net/download/attachments/68551327/"


URL = {
    "image": urljoin(BASE_URL, "NSCLC-Radiomics-OriginalCTs.tcia"),
    "gt": {
        "thoracic": urljoin(
            BASE_URL, "PleThora%20Thoracic_Cavities%20June%202020.zip?version=1&modificationDate=1593202695428&api=v2"
        ),
        "pleural_effusion": urljoin(
            BASE_URL, "PleThora%20Effusions%20June%202020.zip?version=1&modificationDate=1593202778373&api=v2"
        )
    }
}


CHECKSUMS = {
    "image": None,
    "gt": {
        "thoracic": None,
        "pleural_effusion": None
    }
}


def download_source_tcia(path, url, dst, csv_filename):
    assert url.endswith(".tcia")

    manifest = requests.get(url=url)
    with open(path, "wb") as f:
        f.write(manifest.content)

    if os.path.exists(csv_filename):
        prev_df = pd.read_csv(csv_filename)

    df = nbia.downloadSeries(
        series_data=path, input_type="manifest", path=dst, csv_filename=csv_filename
    )

    neu_df = pd.concat(prev_df, df)
    neu_df.to_csv(csv_filename)


def get_plethora_data(path, download):
    os.makedirs(path, exist_ok=True)

    image_dir = os.path.join(path, "data", "images")
    gt_dir = os.path.join(path, "data", "gt")
    if os.path.exists(image_dir) and os.path.exists(gt_dir):
        return image_dir, gt_dir

    tcia_path = os.path.join(path, "NSCLC-Radiomics-OriginalCTs.tcia")

    download_source_tcia(
        path=tcia_path, url=URL, dst=image_dir, csv_filename=os.path.join(path, "plethora_images")
    )


def _get_plethora_paths(path, download):
    data_dir = get_plethora_data()

    breakpoint()


def get_plethora_dataset(
    path: Union[os.PathLike, str],
    download: bool = False,
    **kwargs
):
    


def get_plethora_loader(
    path: Union[os.PathLike, str],
    download: bool = False,
    **kwargs
):
    ...
