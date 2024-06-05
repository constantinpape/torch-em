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


def _create_data_loader(ds_callable, ds_kwargs, loader_kwargs, world_size, rank):
    # Create the dataset.
    ds = ds_callable(**ds_kwargs)

    # Create the sampler
    # Set shuffle on the sampler instead of the loader
    shuffle = loader_kwargs.pop("shuffle", False)
    sampler = torch.utils.data.distributed.DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=shuffle)

    # Create the loader.
    loader = torch.utils.data.DataLoader(ds, sampler=sampler, **loader_kwargs)
    # loader = torch.utils.data.DataLoader(ds, shuffle=shuffle, **loader_kwargs)
    # loader.shuffle = shuffle

    return loader


def _train_impl(
    rank, world_size,
    model_callable, model_kwargs,
    train_dataset_callable, train_dataset_kwargs,
    val_dataset_callable, val_dataset_kwargs,
    loader_kwargs, iterations, **kwargs
):
    assert "device" not in kwargs
    print(f"Running DDP on rank {rank}.")
    setup(rank, world_size)

    model = model_callable(**model_kwargs).to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    train_loader = _create_data_loader(train_dataset_callable, train_dataset_kwargs, loader_kwargs, world_size, rank)
    val_loader = _create_data_loader(val_dataset_callable, val_dataset_kwargs, loader_kwargs, world_size, rank)

    trainer = torch_em.default_segmentation_trainer(
        model=ddp_model, train_loader=train_loader, val_loader=val_loader,
        device=rank, rank=rank, **kwargs
    )
    trainer.fit(iterations=iterations)

    cleanup()


def train_multi_gpu(
    model_callable, model_kwargs,
    train_dataset_callable, train_dataset_kwargs,
    val_dataset_callable, val_dataset_kwargs,
    loader_kwargs, iterations, **kwargs
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
        train_dataset_callable=train_dataset_callable, train_dataset_kwargs=train_dataset_kwargs,
        val_dataset_callable=val_dataset_callable, val_dataset_kwargs=val_dataset_kwargs,
        loader_kwargs=loader_kwargs, iterations=iterations, **kwargs
    )
    torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size, join=True)
