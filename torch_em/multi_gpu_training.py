import os
from functools import partial
from typing import Dict, Any, Optional, Callable

import torch
import torch.utils.data
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

import torch_em


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def _create_data_loader(ds_callable, ds_kwargs, loader_kwargs, world_size, rank):
    # Create the dataset.
    ds = ds_callable(**ds_kwargs)

    # Create the sampler
    # Set shuffle on the sampler instead of the loader
    shuffle = loader_kwargs.pop("shuffle", False)
    sampler = torch.utils.data.distributed.DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=shuffle)

    # Create the loader.
    loader = torch.utils.data.DataLoader(ds, sampler=sampler, **loader_kwargs)
    loader.shuffle = shuffle

    return loader


class DDP(DistributedDataParallel):
    """Wrapper for the DistributedDataParallel class that overrides the `__getattr__` method
    to handle access from the "model" object module wrapped by DDP.
    """
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def _train_impl(
    rank: int,
    world_size: int,
    model_callable: Callable[[Any], torch.nn.Module],
    model_kwargs: Dict[str, Any],
    train_dataset_callable: Callable[[Any], torch.utils.data.Dataset],
    train_dataset_kwargs: Dict[str, Any],
    val_dataset_callable: Callable[[Any], torch.utils.data.Dataset],
    val_dataset_kwargs: Dict[str, Any],
    loader_kwargs: Dict[str, Any],
    iterations: int,
    find_unused_parameters: bool = True,
    optimizer_callable: Optional[Callable[[Any], torch.optim.Optimizer]] = None,
    optimizer_kwargs: Optional[Dict[str, Any]] = None,
    lr_scheduler_callable: Optional[Callable[[Any], torch.optim.lr_scheduler._LRScheduler]] = None,
    lr_scheduler_kwargs: Optional[Dict[str, Any]] = None,
    trainer_callable=None,
    **kwargs
):
    assert "device" not in kwargs
    print(f"Running DDP on rank {rank}.")
    setup(rank, world_size)

    model = model_callable(**model_kwargs).to(rank)
    ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=find_unused_parameters)

    if optimizer_callable is not None:
        optimizer = optimizer_callable(model.parameters(), **optimizer_kwargs)
        kwargs["optimizer"] = optimizer
        if lr_scheduler_callable is not None:
            lr_scheduler = lr_scheduler_callable(optimizer, **lr_scheduler_kwargs)
            kwargs["lr_scheduler"] = lr_scheduler

    train_loader = _create_data_loader(train_dataset_callable, train_dataset_kwargs, loader_kwargs, world_size, rank)
    val_loader = _create_data_loader(val_dataset_callable, val_dataset_kwargs, loader_kwargs, world_size, rank)

    if trainer_callable is None:
        trainer_callable = torch_em.default_segmentation_trainer

    trainer = trainer_callable(
        model=ddp_model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=rank,
        rank=rank,
        **kwargs
    )
    trainer.fit(iterations=iterations)

    cleanup()


def train_multi_gpu(
    model_callable: Callable[[Any], torch.nn.Module],
    model_kwargs: Dict[str, Any],
    train_dataset_callable: Callable[[Any], torch.utils.data.Dataset],
    train_dataset_kwargs: Dict[str, Any],
    val_dataset_callable: Callable[[Any], torch.utils.data.Dataset],
    val_dataset_kwargs: Dict[str, Any],
    loader_kwargs: Dict[str, Any],
    iterations: int,
    find_unused_parameters: bool = True,
    optimizer_callable: Optional[Callable[[Any], torch.optim.Optimizer]] = None,
    optimizer_kwargs: Optional[Dict[str, Any]] = None,
    lr_scheduler_callable: Optional[Callable[[Any], torch.optim.lr_scheduler._LRScheduler]] = None,
    lr_scheduler_kwargs: Optional[Dict[str, Any]] = None,
    trainer_callable=None,
    **kwargs
) -> None:
    """

    Args:
        model: The PyTorch model to be trained.
        kwargs: Keyword arguments for `torch_em.segmentation.default_segmentation_trainer`.
    """
    world_size = torch.cuda.device_count()
    train = partial(
        _train_impl,
        model_callable=model_callable,
        model_kwargs=model_kwargs,
        train_dataset_callable=train_dataset_callable,
        train_dataset_kwargs=train_dataset_kwargs,
        val_dataset_callable=val_dataset_callable,
        val_dataset_kwargs=val_dataset_kwargs,
        loader_kwargs=loader_kwargs,
        iterations=iterations,
        find_unused_parameters=find_unused_parameters,
        optimizer_callable=optimizer_callable,
        optimizer_kwargs=optimizer_kwargs,
        lr_scheduler_callable=lr_scheduler_callable,
        lr_scheduler_kwargs=lr_scheduler_kwargs,
        trainer_callable=trainer_callable,
        **kwargs
    )
    torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size, join=True)
