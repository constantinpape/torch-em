from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_cholec_instance_seg_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_cholec_instance_seg():
    loader = get_cholec_instance_seg_loader(
        path=f"{DATA_ROOT}/cholec_instance_seg",
        batch_size=2,
        patch_shape=(512, 512),
        split="train",
        resize_inputs=True,
        download=True,
    )
    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_cholec_instance_seg()
