import os
from functools import partial

import torch
import torch_em
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def _train_impl(
    rank, world_size,
    model_callable, model_kwargs,
    train_loader_callable, train_loader_kwargs,
    val_loader_callable, val_loader_kwargs,
    iterations, **kwargs
):
    assert "device" not in kwargs
    print(f"Running DDP on rank {rank}.")
    setup(rank, world_size)

    model = model_callable(**model_kwargs).to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    train_loader = train_loader_callable(**train_loader_kwargs)
    val_loader = val_loader_callable(**val_loader_kwargs)

    trainer = torch_em.default_segmentation_trainer(
        model=ddp_model, train_loader=train_loader, val_loader=val_loader,
        device=rank, rank=rank, **kwargs
    )
    trainer.fit(iterations=iterations)

    cleanup()


def train_multi_gpu(
    model_callable, model_kwargs,
    train_loader_callable, train_loader_kwargs,
    val_loader_callable, val_loader_kwargs,
    iterations, **kwargs
) -> None:
    """

    Args:
        model: The PyTorch model to be trained.
        kwargs: Keyword arguments for `torch_em.segmentation.default_segmentation_trainer`.
    """
    world_size = torch.cuda.device_count()
    train = partial(
        _train_impl,
        model_callable=model_callable, model_kwargs=model_kwargs,
        train_loader_callable=train_loader_callable, train_loader_kwargs=train_loader_kwargs,
        val_loader_callable=val_loader_callable, val_loader_kwargs=val_loader_kwargs,
        iterations=iterations, **kwargs
    )
    torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size, join=True)
