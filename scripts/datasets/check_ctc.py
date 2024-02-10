from torch_em.data.datasets.ctc import get_ctc_segmentation_loader, CTC_CHECKSUMS
from torch_em.util.debug import check_loader
from torch_em.data.sampler import MinInstanceSampler

ROOT = "/home/anwai/data/ctc/"


# Some of the datasets have partial sparse labels:
# - Fluo-N2DH-GOWT1
# - Fluo-N2DL-HeLa
# Maybe depends on the split?!
def check_ctc_segmentation():
    ctc_dataset_names = list(CTC_CHECKSUMS["train"].keys())[-1]
    for name in [ctc_dataset_names]:
        print("Checking dataset", name)
        loader = get_ctc_segmentation_loader(
            path=ROOT,
            dataset_name=name,
            patch_shape=(1, 512, 512),
            batch_size=1,
            download=True,
            split="train",
            sampler=MinInstanceSampler()
        )
        check_loader(loader, 8, plt=True, save_path="ctc.png")


if __name__ == "__main__":
    check_ctc_segmentation()
