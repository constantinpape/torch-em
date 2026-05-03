import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_orion_crc_loader


sys.path.append("..")


def check_orion_crc(modality, label_type):
    from util import ROOT

    loader = get_orion_crc_loader(
        path=os.path.join(ROOT, "orion_crc"),
        split="train",
        modality=modality,
        label_type=label_type,
        patch_shape=(512, 512),
        batch_size=2,
        download=False,
    )

    is_instance = label_type == "instances"
    print(f"ORION-CRC | modality={modality} | label_type={label_type} | #batches={len(loader)}")
    check_loader(loader, 8, instance_labels=is_instance, rgb=modality == "he")


if __name__ == "__main__":
    check_orion_crc(modality="he", label_type="instances")
    check_orion_crc(modality="he", label_type="semantic")
    check_orion_crc(modality="mif", label_type="instances")
    check_orion_crc(modality="mif", label_type="semantic")
