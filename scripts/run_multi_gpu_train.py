from torch_em.model import UNet2d
from torch_em.multi_gpu_training import train_multi_gpu

from torch_em.data.datasets.light_microscopy.dsb import get_dsb_loader, get_dsb_data


model_class = UNet2d
model_kwargs = {"in_channels": 1, "out_channels": 1}

data_root = "./data-dsb"
# Download the data
# get_dsb_data(data_root, "reduced", True)
# quit()

train_loader_class = get_dsb_loader
train_loader_kwargs = {
    "path": data_root, "split": "train",
    "patch_shape": (256, 256), "batch_size": 4,
    "binary": True
}

val_loader_class = get_dsb_loader
val_loader_kwargs = {
    "path": data_root, "split": "test",
    "patch_shape": (256, 256), "batch_size": 4,
    "binary": True
}

if __name__ == "__main__":
    train_multi_gpu(
        model_class, model_kwargs,
        train_loader_class, train_loader_kwargs,
        val_loader_class, val_loader_kwargs,
        iterations=1e3, name="multi-gpu-test",
        compile_model=False,
    )
