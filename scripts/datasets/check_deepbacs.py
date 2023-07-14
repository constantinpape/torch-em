from torch_em.data.datasets import get_deepbacs_loader
from torch_em.util.debug import check_loader


def check_deepbacs():
    loader = get_deepbacs_loader("./deepbacs", "test", bac_type="mixed", download=True,
                                 patch_shape=(256, 256), batch_size=1, shuffle=True)
    check_loader(loader, 15, instance_labels=True)


if __name__ == "__main__":
    check_deepbacs()
