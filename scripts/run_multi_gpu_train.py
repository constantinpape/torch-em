from torch_em.model import UNet2d
from torch_em.multi_gpu_training import train_multi_gpu


model_class = UNet2d
model_kwargs = {"in_channels": 1, "out_channels": 1}


# TODO can we multi-process the classes?
train_multi_gpu(
    model_class, model_kwargs,
    # train_loader_class, train_loader_kwargs,
    # val_loader_class, val_loader_kwargs,
    n_iterations=1e3,
)
