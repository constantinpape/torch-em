import sys

from torch_em.data import MinInstanceSampler
from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_liconn_loader

sys.path.append("..")

DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data/liconn"


def check_liconn():
    for volume, segmentations in [("fig2", ("proofread", "agglomerated")), ("fig1", ("sparse", "dense_auto"))]:
        for seg in segmentations:
            print(f"Checking volume='{volume}', segmentation='{seg}'")
            # The sparse fig1 labels cover only a few neurons, so make sure the patches contain some.
            sampler = MinInstanceSampler(min_num_instances=2) if seg == "sparse" else None
            loader = get_liconn_loader(
                DATA_ROOT, batch_size=1, patch_shape=(32, 256, 256), segmentation=seg,
                volume=volume, download=True, sampler=sampler,
            )
            check_loader(loader, 4, instance_labels=True, plt=True, save_path=f"check_liconn_{volume}_{seg}.png")


def main():
    check_liconn()


if __name__ == "__main__":
    main()
