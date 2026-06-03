import numpy as np
import napari
from torch_em.data.datasets.histopathology.glysac import get_glysac_loader

DATA_PATH = "/tmp/glysac_test"


def check_loader(split, label_choice):
    loader = get_glysac_loader(
        DATA_PATH, batch_size=1, patch_shape=(512, 512), split=split, label_choice=label_choice, download=False
    )
    print(f"Number of batches ({split}, {label_choice}):", len(loader))

    x, y = next(iter(loader))
    print(f"  Input shape: {x.shape}, Labels shape: {y.shape}")

    v = napari.Viewer()
    v.add_image(x.numpy()[0].transpose(1, 2, 0), name="raw")
    if label_choice == "instances":
        v.add_labels(y.numpy()[0, 0].astype(np.int32), name="instances")
    else:
        v.add_labels(y.numpy()[0, 0].astype(np.int32), name="semantic")
    napari.run()


def main():
    check_loader("train", "instances")
    check_loader("train", "semantic")


main()
