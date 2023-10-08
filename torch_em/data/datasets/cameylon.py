import os
import warnings
from glob import glob


def _download_cameylon(path):
    is_cam16 = os.path.exists(os.path.join(path, "CAMELYON16"))
    is_cam17 = os.path.exists(os.path.join(path, "CAMELYON17"))
    if is_cam16 and is_cam17 is True:
        return

    try:
        import awscli
    except ModuleNotFoundError:
        os.system("pip install awscli")

    warnings.warn("The CAMELYON dataset could take a couple of hours to download the dataset.")

    os.system(f"aws s3 cp --no-sign-request s3://camelyon-dataset/ {path} --recursive")


def get_cameylon_dataset(path):
    """Description of concents in the dataset folders
    - CAMELYON16:
        - checksums.md5
        - annotations/ : 160 (xml) files
        - license.txt
        - README.md
        - pathology-tissue-background-segmentation.json
        - evaluation/ : scripts
        - masks/ : 399 (tif) files
        - background_tissue/ : 400 (tif) files
        - images/ : 399 (tif) files

    - CAMELYON17:
        - checksums.md5
        - annotations/ : 50 (xml) files
        - license.txt
        - README.md
        - stages.csv
        - evaluation/ : scripts
        - masks/ : 100 (tif) files
        - images/ : 998 (tif) files
    """
    all_paths = sorted(glob(os.path.join(path, "CAMELYON16", "images", "*")))
    print(all_paths[-1])

    level = 5  # Image level at which the evaluation is done

    import openslide
    import numpy as np

    slide = openslide.open_slide(all_paths[-1])
    dims = slide.level_dimensions[level]
    pixelarray = np.array(slide.read_region((0, 0), level, dims))
    breakpoint()


def get_cameylon_loader(path):
    _download_cameylon(path)
    get_cameylon_dataset(path)


def main():
    path = "/scratch/usr/nimanwai/data/test/"
    get_cameylon_loader(
        path=path
    )


if __name__ == "__main__":
    main()
