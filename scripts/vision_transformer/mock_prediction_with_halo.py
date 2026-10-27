"""Estimate the time of block-wise 3D prediction with a halo for a UNETR with a SAM ViT-B encoder.

The mock volume has the shape (260, 625, 625). Each block has the shape (3, 170, 170) and a halo of (2, 43, 43).
So the model input for each block has the training patch shape (7, 256, 256). The volume has 1392 blocks.
The script predicts in bfloat16 (bf16), which is the fastest setup that we found.

Measured times on 2g.20gb multi-instance GPU (MIG) slices of an A100:
- float32 (fp32), 1 slice: 8.3 s per block, 192 min.
- bf16, 1 slice: 1.7 s per block, 39 min.
- bf16, 2 slices: 1.7 s per block on each slice, 696 blocks per slice, 19.5 min.

One process can only use one MIG slice. To use 2 slices, run one process per slice.
Give each process every second block with `iter_list`.

The script times a subset of the blocks after a warmup. Then it extrapolates the time to all blocks.
The model uses random weights, because the weights do not change the runtime.
"""
import time

import numpy as np
import bioimage_cpp as bic

import torch

from torch_em.model.unetr import get_unetr_model
from torch_em.util.prediction import predict_with_halo

SHAPE = (260, 625, 625)
BLOCK_SHAPE = (3, 170, 170)
HALO = (2, 43, 43)
N_WARMUP = 1
N_TIMED = 8


def scale_to_sam_range(x):
    # SAM expects inputs in the range [0, 255].
    x = x.astype("float32")
    return 255 * (x - x.min()) / (x.max() - x.min() + 1e-7)


def predict_bf16(net, inp):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        return net(inp).float()


def run_blocks(input_, model, block_ids):
    output = np.zeros((3,) + input_.shape, dtype="float32")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    predict_with_halo(
        input_, model, gpu_ids=[0], block_shape=BLOCK_SHAPE, halo=HALO, output=output, iter_list=block_ids,
        preprocess=scale_to_sam_range, prediction_function=predict_bf16, disable_tqdm=True,
    )
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def main():
    model = get_unetr_model(ndim=3, backbone="sam", model_type="vit_b", out_channels=3, final_activation="Sigmoid")
    model = model.to("cuda").eval()
    input_ = np.random.default_rng(seed=42).random(SHAPE, dtype="float32")
    n_blocks = bic.utils.Blocking([0, 0, 0], list(SHAPE), list(BLOCK_SHAPE)).number_of_blocks

    run_blocks(input_, model, list(range(N_WARMUP)))
    torch.cuda.reset_peak_memory_stats()
    duration = run_blocks(input_, model, list(range(N_WARMUP, N_WARMUP + N_TIMED)))

    s_per_block = duration / N_TIMED
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Time per block: {s_per_block:.2f} s (timed {N_TIMED} blocks)")
    print(f"Estimated time for {n_blocks} blocks: {s_per_block * n_blocks / 60:.1f} min")
    print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
