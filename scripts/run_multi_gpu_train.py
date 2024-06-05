from torch_em.model import UNet2d
from torch_em.multi_gpu_training import train_multi_gpu

from torch_em.data.datasets.light_microscopy.dsb import get_dsb_dataset


model_class = UNet2d
model_kwargs = {"in_channels": 1, "out_channels": 1}

data_root = "./data-dsb"
# Download the data
# get_dsb_data(data_root, "reduced", True)
# quit()

train_dataset_class = get_dsb_dataset
train_dataset_kwargs = {
    "path": data_root, "split": "train",
    "patch_shape": (256, 256), "binary": True
}

val_dataset_class = get_dsb_dataset
val_dataset_kwargs = {
    "path": data_root, "split": "test",
    "patch_shape": (256, 256), "binary": True
}

loader_kwargs = {"batch_size": 4, "shuffle": True, "num_workers": 4}

if __name__ == "__main__":
    train_multi_gpu(
        model_class, model_kwargs,
        train_dataset_class, train_dataset_kwargs,
        val_dataset_class, val_dataset_kwargs,
        loader_kwargs=loader_kwargs,
        iterations=1e3, name="multi-gpu-test",
        compile_model=False,
    )
