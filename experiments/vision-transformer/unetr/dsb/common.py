import torch_em
from torch_em.data.datasets import get_dsb_loader
from torch_em.transform.raw import get_raw_transform
from micro_sam.training import identity


def get_loaders(normalize_raw, batch_size=4, patch_shape=(1, 256, 256), data_root="./data"):
    if normalize_raw:
        raw_trafo = get_raw_transform()
    else:
        raw_trafo = identity

    label_trafo = torch_em.transform.label.PerObjectDistanceTransform(
        distances=True,
        boundary_distances=True,
        directed_distances=False,
        foreground=True,
        min_size=25,
    )

    train_loader = get_dsb_loader(
        data_root, patch_shape=patch_shape, split="train",
        download=True, batch_size=batch_size, ndim=2,
        label_transform=label_trafo, raw_transform=raw_trafo
    )
    val_loader = get_dsb_loader(
        data_root, patch_shape=patch_shape, split="test", batch_size=batch_size,
        label_transform=label_trafo, raw_transform=raw_trafo, ndim=2,
    )

    return train_loader, val_loader


# TODO visualize the loader
def main():
    pass


if __name__ == "__main__":
    main()
