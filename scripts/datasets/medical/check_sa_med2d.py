from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_sa_med2d_loader


def check_sa_med2d():
    loader = get_sa_med2d_loader(
        ...
    )

    check_loader(loader, 8)


if __name__ == "__main__":
    check_sa_med2d()
