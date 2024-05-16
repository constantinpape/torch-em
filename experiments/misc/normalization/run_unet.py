import os

import torch

import torch_em
from torch_em.loss import DiceLoss

from common import get_dataloaders, get_model, get_experiment_name


SAVE_DIR = "/scratch/projects/nim00007/test/verify_normalization"


def run_training(name, model, dataset, task, save_root, device):
    train_loader, val_loader = get_dataloaders(dataset=dataset, task=task)

    from torch_em.util.debug import check_loader
    check_loader(train_loader, 8, plt=True, save_path="./test.png")

    for x, y in train_loader:
        breakpoint()

    breakpoint()

    loss = DiceLoss()

    trainer = torch_em.default_segmentation_trainer(
        name=name,
        save_root=save_root,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss=loss,
        metric=loss,
        device=device,
        learning_rate=1e-4,
        mixed_precision=True,
        compile_model=False,
        log_image_interval=100,
    )
    trainer.fit(iterations=int(1e5))


def run_inference():
    pass


def run_evaluation():
    pass


def main(args):
    phase = args.phase
    dataset = args.dataset
    task = args.task
    norm = args.norm

    assert norm in ["OldDefault", "InstanceNorm"]

    save_root = os.path.join(SAVE_DIR, "models")
    model = get_model(dataset=dataset, task=task, norm=norm)
    name = get_experiment_name(dataset=dataset, task=task, norm=norm, model_choice="unet")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if phase == "train":
        run_training(
            name=name, model=model, dataset=dataset, task=task, save_root=save_root, device=device
        )

    elif phase == "predict":
        run_inference(device=device)

    elif phase == "evaluate":
        run_evaluation()

    else:
        print(f"'{phase}' is not a valid mode. Choose from 'train' / 'predict' / 'evaluate'.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dataset", required=True, type=str, help="The choice of dataset."
    )
    parser.add_argument(
        "-p", "--phase", required=True, type=str, help="The mode for running the scripts."
    )
    parser.add_argument(
        "-t", "--task", required=True, type=str, help="The type of task for segmentation."
    )
    parser.add_argument(
        "-n", "--norm", required=True, type=str, help="The choice of layer normalization."
    )
    args = parser.parse_args()
    main(args)
