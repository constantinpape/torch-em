import os
import warnings

try:
    import awscli
except ModuleNotFoundError:
    os.system("pip install awscli")


def _download_cameylon(path):
    is_cam16 = os.path.exists(os.path.join(path, "CAMEYLON16"))
    is_cam17 = os.path.exists(os.path.join(path, "CAMEYLON17"))
    if is_cam16 and is_cam17:
        return

    warnings.warn("The CAMEYLON dataset could take a couple of hours to download the dataset.")

    os.system(f"aws s3 sync --no-sign-request s3://camelyon-dataset/ {path} --recursive")


def get_cameylon_loader(path):
    _download_cameylon(path)


def main():
    path = "/scratch/usr/nimanwai/data/test/"
    get_cameylon_loader(
        path=path
    )


if __name__ == "__main__":
    main()
