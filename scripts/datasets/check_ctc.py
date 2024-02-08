from torch_em.data.datasets.ctc import get_ctc_segmentation_loader, CTC_URLS
from torch_em.util.debug import check_loader
from torch_em.data.sampler import MinInstanceSampler

ROOT = "/home/pape/Work/data/ctc/ctc-training-data"


# Some of the datasets have partial sparse labels:
# - Fluo-N2DH-GOWT1
# - Fluo-N2DL-HeLa
# Maybe depends on the split?!
def check_ctc_segmentation():
    for name in CTC_URLS.keys():
        if not name.startswith("DIC"):
            continue
        print("Checking dataset", name)
        loader = get_ctc_segmentation_loader(
            ROOT, name, (1, 512, 512), 1, download=True,
            sampler=MinInstanceSampler()
        )
        check_loader(loader, 8, instance_labels=True)


if __name__ == "__main__":
    check_ctc_segmentation()
