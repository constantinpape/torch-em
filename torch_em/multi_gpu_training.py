import os
from functools import partial

import torch
import torch_em
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_grop("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def _train_impl(
    rank, world_size, model_class, model_kwargs, n_iterations, trainer_class, **kwargs
):
    assert "device" not in kwargs
    print("Running DDP on rank {rank}.")
    setup(rank, world_size)

    model = model_class(**model_kwargs).to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    # TODO train and val loader!

    trainer = torch_em.default_segmentation_trainer(model=ddp_model, **kwargs)
    trainer.fit(n_iterations=n_iterations)

    cleanup()


# TODO we need to also accept the train and val loader, but first make sure that works with multi-processing
def train_multi_gpu(model_class, model_kwargs, n_iterations, **kwargs) -> None:
    """

    Args:
        model: The PyTorch model to be trained.
        kwargs: Keyword arguments for `torch_em.segmentation.default_segmentation_trainer`.
    """
    world_size = torch.cuda.device_count()
    train = partial(_train_impl, model_class=model_class, model_kwargs=model_kwargs, **kwargs)
    torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size, join=True)
