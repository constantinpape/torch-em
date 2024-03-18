from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_dynamicnuclearnet_loader


DYNAMICNUCLEARNET_ROOT = "/home/anwai/data/deepcell/"


# NOTE: the DynamicNuclearNet data cannot be downloaded automatically.
# you need to download it yourself from https://datasets.deepcell.org/data
def check_dynamicnuclearnet():
    # set this path to where you have downloaded the dynamicnuclearnet data
    loader = get_dynamicnuclearnet_loader(
        DYNAMICNUCLEARNET_ROOT, "train",
        patch_shape=(512, 512), batch_size=2, download=True
    )
    check_loader(loader, 10, instance_labels=True, rgb=False)


if __name__ == "__main__":
    check_dynamicnuclearnet()
