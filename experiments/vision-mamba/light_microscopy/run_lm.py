import os
import argparse
from glob import glob

import numpy as np
import pandas as pd
from tqdm import tqdm
import imageio.v3 as imageio

import torch

import torch_em
from torch_em.util import segmentation
from torch_em.transform.raw import standardize
from torch_em.model import get_vimunet_model
from torch_em.loss import DiceBasedDistanceLoss

import elf.segmentation.multicut as mc
import elf.segmentation.watershed as ws
import elf.segmentation.features as feats
from elf.evaluation import mean_segmentation_accuracy

from obtain_lm_datasets import get_lm_loaders


ROOT = "/scratch/projects/nim00007/sam/data"


def run_lm_training(args):
    # the dataloaders for lm datasets
    train_loader, val_loader = get_lm_loaders(ROOT, (512, 512))

    if args.pretrained:
        assert args.model_type == "vim_t"
        checkpoint = "/scratch/usr/nimanwai/models/Vim-tiny/vim_tiny_73p1.pth"
    else:
        checkpoint = None

    # the vision-mamba + decoder (UNet-based) model
    model = get_vimunet_model(
        out_channels=3,
        model_type=args.model_type,
        checkpoint=checkpoint,
        with_cls_token=True
    )

    # saving the model checkpoints
    save_root = os.path.join(
        args.save_root, "pretrained" if args.pretrained else "scratch", "distances", args.model_type
    )

    # loss function
    loss = DiceBasedDistanceLoss(mask_distances_in_bg=True)

    # trainer for the segmentation task
    trainer = torch_em.default_segmentation_trainer(
        name="lm-vimunet",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.lr,
        loss=loss,
        metric=loss,
        log_image_interval=50,
        save_root=save_root,
        compile_model=False,
        scheduler_kwargs={"mode": "min", "factor": 0.9, "patience": 10}
    )
    trainer.fit(iterations=args.iterations)


def _do_bd_multicut_watershed(bd):
    ws_seg, max_id = ws.distance_transform_watershed(bd, threshold=0.5, sigma_seeds=2.0)

    # compute the region adjacency graph
    rag = feats.compute_rag(ws_seg)

    # compute the edge costs
    costs = feats.compute_boundary_features(rag, bd)[:, 0]

    # transform the edge costs from [0, 1] to  [-inf, inf], which is
    # necessary for the multicut. This is done by intepreting the values
    # as probabilities for an edge being 'true' and then taking the negative log-likelihood.

    # in addition, we weight the costs by the size of the corresponding edge
    # for z and xy edges
    z_edges = feats.compute_z_edge_mask(rag, ws_seg)
    xy_edges = np.logical_not(z_edges)
    edge_populations = [z_edges, xy_edges]
    edge_sizes = feats.compute_boundary_mean_and_length(rag, bd)[:, 1]
    costs = mc.transform_probabilities_to_costs(costs, edge_sizes=edge_sizes, edge_populations=edge_populations)

    # run the multicut partitioning, here, we use the kernighan lin
    # heuristics to solve the problem, introduced in
    # http://xilinx.asia/_hdl/4/eda.ee.ucla.edu/EE201A-04Spring/kl.pdf
    node_labels = mc.multicut_kernighan_lin(rag, costs)

    # map the results back to pixels to obtain the final segmentation
    seg = feats.project_node_labels_to_pixels(rag, node_labels)

    return seg


def _do_affs_multicut_watershed(affs, offsets):
    # first, we have to make a single channel input map for the watershed,
    # which we obtain by averaging the affinities
    boundary_input = np.mean(affs, axis=0)

    ws_seg, max_id = ws.distance_transform_watershed(boundary_input, threshold=0.25, sigma_seeds=2.0)

    # compute the region adjacency graph
    rag = feats.compute_rag(ws_seg)

    # compute the edge costs
    # the offsets encode the pixel transition encoded by the
    # individual affinity channels. Here, we only have nearest neighbor transitions
    costs = feats.compute_affinity_features(rag, affs, offsets)[:, 0]

    # transform the edge costs from [0, 1] to  [-inf, inf], which is
    # necessary for the multicut. This is done by intepreting the values
    # as probabilities for an edge being 'true' and then taking the negative log-likelihood.
    # in addition, we weight the costs by the size of the corresponding edge
    edge_sizes = feats.compute_boundary_mean_and_length(rag, boundary_input)[:, 1]
    costs = mc.transform_probabilities_to_costs(costs, edge_sizes=edge_sizes)

    # run the multicut partitioning, here, we use the kernighan lin
    # heuristics to solve the problem, introduced in
    # http://xilinx.asia/_hdl/4/eda.ee.ucla.edu/EE201A-04Spring/kl.pdf
    node_labels = mc.multicut_kernighan_lin(rag, costs)

    # map the results back to pixels to obtain the final segmentation
    seg = feats.project_node_labels_to_pixels(rag, node_labels)

    return seg


def run_lm_inference(args, device):
    # saving the model checkpoints
    save_root = os.path.join(
        args.save_root, "pretrained" if args.pretrained else "scratch", "distances", args.model_type
    )

    checkpoint = os.path.join(save_root, "checkpoints", "lm-vimunet", "best.pt")

    # the vision-mamba + decoder (UNet-based) model
    model = get_vimunet_model(
        out_channels=3,
        model_type=args.model_type,
        with_cls_token=True,
        checkpoint=checkpoint
    )

    test_image_dir = os.path.join(ROOT, "livecell", "images", "livecell_test_images")
    all_test_labels = glob(os.path.join(ROOT, "livecell", "annotations", "livecell_test_images", "*", "*"))

    res_path = os.path.join(save_root, "results.csv")
    if os.path.exists(res_path) and not args.force:
        print(pd.read_csv(res_path))
        print(f"The result is saved at {res_path}")
        return

    msa_list, sa50_list, sa75_list = [], [], []
    for label_path in tqdm(all_test_labels):
        labels = imageio.imread(label_path)
        image_id = os.path.split(label_path)[-1]

        image = imageio.imread(os.path.join(test_image_dir, image_id))
        image = standardize(image)

        tensor_image = torch.from_numpy(image)[None, None].to(device)

        predictions = model(tensor_image)
        predictions = predictions.squeeze().detach().cpu().numpy()

        fg, cdist, bdist = predictions
        instances = segmentation.watershed_from_center_and_boundary_distances(
            cdist, bdist, fg, min_size=50,
            center_distance_threshold=0.5,
            boundary_distance_threshold=0.6,
            distance_smoothing=1.0
        )

        msa, sa_acc = mean_segmentation_accuracy(instances, labels, return_accuracies=True)
        msa_list.append(msa)
        sa50_list.append(sa_acc[0])
        sa75_list.append(sa_acc[5])

    res = {
        "LiveCELL": "Metrics",
        "mSA": np.mean(msa_list),
        "SA50": np.mean(sa50_list),
        "SA75": np.mean(sa75_list)
    }
    df = pd.DataFrame.from_dict([res])
    df.to_csv(res_path)
    print(df)
    print(f"The result is saved at {res_path}")


def main(args):
    print(torch.cuda.get_device_name() if torch.cuda.is_available() else "GPU not available, hence running on CPU")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.train:
        run_lm_training(args)

    if args.predict:
        run_lm_inference(args, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=int(1e5))
    parser.add_argument("-s", "--save_root", type=str, default=os.path.join(ROOT, "experiments", "vimunet"))
    parser.add_argument("-m", "--model_type", type=str, default="vim_t")
    parser.add_argument("--lr", type=float, default=1e-5)

    parser.add_argument("--pretrained", action="store_true")

    parser.add_argument("--train", action="store_true")
    parser.add_argument("--predict", action="store_true")

    parser.add_argument("--force", action="store_true")

    args = parser.parse_args()
    main(args)
