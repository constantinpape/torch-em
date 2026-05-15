import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_orion_crc_loader


sys.path.append("..")

DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data/orion_crc"


def check_orion_crc(modality, label_type):
    loader = get_orion_crc_loader(
        path=DATA_ROOT,
        split="train",
        modality=modality,
        label_type=label_type,
        patch_shape=(512, 512),
        batch_size=2,
        download=True,
    )

    is_instance = label_type == "instances"
    print(f"ORION-CRC | modality={modality} | label_type={label_type} | #batches={len(loader)}")
    check_loader(
        loader, 8, instance_labels=is_instance, rgb=modality == "he",
        plt=True, save_path=f"check_orion_crc_{modality}_{label_type}.png"
    )


if __name__ == "__main__":
    check_orion_crc(modality="he", label_type="instances")
    check_orion_crc(modality="he", label_type="semantic")
    check_orion_crc(modality="mif", label_type="instances")
    check_orion_crc(modality="mif", label_type="semantic")
