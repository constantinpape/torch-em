import os
import argparse
from tqdm import tqdm
from glob import glob

import numpy as np
import pandas as pd
import imageio.v3 as imageio

import torch

import torch_em
from torch_em.util import segmentation
from torch_em.data import MinInstanceSampler
from torch_em.model import get_vimunet_model
from torch_em.transform.raw import standardize
from torch_em.util.prediction import predict_with_halo
from torch_em.data.datasets.neurips_cell_seg import to_rgb
from torch_em.data.datasets import get_neurips_cellseg_supervised_loader
from torch_em.transform.label import AffinityTransform, BoundaryTransform
from torch_em.loss import DiceLoss, LossWrapper, ApplyAndRemoveMask, DiceBasedDistanceLoss

import elf.segmentation.multicut as mc
import elf.segmentation.watershed as ws
import elf.segmentation.features as feats
from elf.evaluation import mean_segmentation_accuracy


ROOT = "/scratch/usr/nimanwai/"

OFFSETS = [
    [-1, 0], [0, -1],
    [-3, 0], [0, -3],
    [-9, 0], [0, -9],
    [-27, 0], [0, -27]
]


def get_loaders(args, patch_shape=(512, 512)):
    if args.boundaries:
        label_trafo = BoundaryTransform(add_binary_target=True)
    elif args.affinities:
        label_trafo = AffinityTransform(offsets=OFFSETS, add_binary_target=True, add_mask=True)
    elif args.distances:
        label_trafo = torch_em.transform.label.PerObjectDistanceTransform(
            distances=True, boundary_distances=True, directed_distances=False, foreground=True
        )

    sampler = MinInstanceSampler(min_num_instances=3)

    train_loader = get_neurips_cellseg_supervised_loader(
        root=args.input,
        split="train",
        patch_shape=patch_shape,
        batch_size=2,
        make_rgb=True,
        download=True,
        label_transform=label_trafo,
        num_workers=16,
        sampler=sampler,
    )
    val_loader = get_neurips_cellseg_supervised_loader(
        root=args.input,
        split="val",
        patch_shape=patch_shape,
        batch_size=1,
        make_rgb=True,
        download=True,
        label_transform=label_trafo,
        num_workers=16,
        sampler=sampler,
    )

    # increasing the sampling attempts for the neurips cellseg dataset
    train_loader.dataset.max_sampling_attempts = 5000
    val_loader.dataset.max_sampling_attempts = 5000

    return train_loader, val_loader


def get_output_channels(args):
    if args.boundaries:
        output_channels = 2
    elif args.distances:
        output_channels = 3
    elif args.affinities:
        output_channels = (len(OFFSETS) + 1)

    return output_channels


def get_loss_function(args):
    if args.affinities:
        loss = LossWrapper(
            loss=DiceLoss(),
            transform=ApplyAndRemoveMask(masking_method="multiply")
        )
    elif args.distances:
        loss = DiceBasedDistanceLoss(mask_distances_in_bg=True)

    else:
        loss = DiceLoss()

    return loss


def get_save_root(args):
    # experiment_type
    if args.boundaries:
        experiment_type = "boundaries"
    elif args.affinities:
        experiment_type = "affinities"
    elif args.distances:
        experiment_type = "distances"
    else:
        raise ValueError

    model_name = args.model_type

    # saving the model checkpoints
    save_root = os.path.join(
        args.save_root, "pretrained" if args.pretrained else "scratch", experiment_type, model_name
    )
    return save_root


def run_neurips_cellseg_training(args, device):
    # the dataloaders for neurips cellseg dataset
    train_loader, val_loader = get_loaders(args)

    if args.pretrained:
        checkpoint = "/scratch/usr/nimanwai/models/Vim-tiny/vim_tiny_73p1.pth"
    else:
        checkpoint = None

    output_channels = get_output_channels(args)

    # the vision-mamba + decoder (UNet-based) model
    model = get_vimunet_model(
        out_channels=output_channels,
        model_type=args.model_type,
        checkpoint=checkpoint,
        with_cls_token=True
    )

    save_root = get_save_root(args)

    # loss function
    loss = get_loss_function(args)

    trainer = torch_em.default_segmentation_trainer(
        name="neurips-cellseg-vimunet",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.lr,
        loss=loss,
        metric=loss,
        log_image_interval=50,
        save_root=save_root,
        compile_model=False,
        scheduler_kwargs={"mode": "min", "factor": 0.9, "patience": 10}
    )

    trainer.fit(args.iterations)


def _do_bd_multicut_watershed(bd):
    ws_seg, max_id = ws.distance_transform_watershed(bd, threshold=0.25, sigma_seeds=2.0)

    # compute the region adjacency graph
    rag = feats.compute_rag(ws_seg)

    # compute the edge costs
    costs = feats.compute_boundary_features(rag, bd)[:, 0]

    # transform the edge costs from [0, 1] to  [-inf, inf], which is
    # necessary for the multicut. This is done by intepreting the values
    # as probabilities for an edge being 'true' and then taking the negative log-likelihood.
    edge_sizes = feats.compute_boundary_mean_and_length(rag, bd)[:, 1]
    costs = mc.transform_probabilities_to_costs(costs, edge_sizes=edge_sizes)

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


def run_neurips_cellseg_inference(args, device):
    output_channels = get_output_channels(args)

    save_root = get_save_root(args)

    checkpoint = os.path.join(save_root, "checkpoints", "neurips-cellseg-vimunet", "best.pt")

    # the vision-mamba + decoder (UNet-based) model
    model = get_vimunet_model(
        out_channels=output_channels,
        model_type=args.model_type,
        with_cls_token=True,
        checkpoint=checkpoint
    )
    model.eval()

    root_dir = "/scratch/projects/nim00007/sam/data/neurips-cell-seg/zenodo/Testing/Public/"
    all_test_images = sorted(glob(os.path.join(root_dir, "images", "*")))
    all_test_labels = sorted(glob(os.path.join(root_dir, "labels", "*")))
    # for wsi:
    # images - sorted(glob(os.path.join(root_dir, "WSI", "*")))
    # labels - sorted(glob(os.path.join(root_dir, "WSI-labels", "*")))

    assert len(all_test_images) == len(all_test_labels)

    res_path = os.path.join(save_root, "results.csv")
    if os.path.exists(res_path) and not args.force:
        print(pd.read_csv(res_path))
        print(f"The result is saved at {res_path}")
        return

    msa_list, sa50_list, sa75_list = [], [], []
    for image_path, label_path in tqdm(zip(all_test_images, all_test_labels), total=len(all_test_images)):
        image = imageio.imread(image_path)
        labels = imageio.imread(label_path)

        image = to_rgb(image)
        image = standardize(image)

        predictions = predict_with_halo(
            image, model, [device], block_shape=[512, 512], halo=[256, 256],
            disable_tqdm=True, with_channels=True
        )

        if args.boundaries:
            fg, bd = predictions.squeeze()

            if args.multicut:
                instances = _do_bd_multicut_watershed(bd)
            else:
                instances = segmentation.watershed_from_components(bd, fg)

        elif args.affinities:
            fg, affs = predictions[0], predictions[1:]

            if args.multicut:
                instances = _do_affs_multicut_watershed(affs[:4], OFFSETS[:4])
            else:
                instances = segmentation.mutex_watershed_segmentation(fg, affs, offsets=OFFSETS)

        elif args.distances:
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
    assert (args.boundaries + args.affinities + args.distances) == 1

    print(torch.cuda.get_device_name() if torch.cuda.is_available() else "GPU not available, hence running on CPU")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.train:
        run_neurips_cellseg_training(args, device)

    if args.predict:
        run_neurips_cellseg_inference(args, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", type=str, default=os.path.join(ROOT, "data", "neurips-cell-seg", "zenodo")
    )
    parser.add_argument("--iterations", type=int, default=int(1e5))
    parser.add_argument("-s", "--save_root", type=str, default=os.path.join(ROOT, "experiments", "vimunet"))
    parser.add_argument("-m", "--model_type", type=str, required=True)
    parser.add_argument("--lr", type=float, default=1e-5)

    parser.add_argument("--pretrained", action="store_true")

    parser.add_argument("--multicut", action="store_true")

    parser.add_argument("--force", action="store_true")

    parser.add_argument("--train", action="store_true")
    parser.add_argument("--predict", action="store_true")

    parser.add_argument("--boundaries", action="store_true")
    parser.add_argument("--affinities", action="store_true")
    parser.add_argument("--distances", action="store_true")

    args = parser.parse_args()
    main(args)
