import os
from datetime import datetime

import numpy as np

try:
    import wandb
except ImportError:
    wandb = None

from .tensorboard_logger import normalize_im, make_grid_image


class WandbLogger:
    def __init__(self, trainer):
        self.log_dir = "./logs"
        os.makedirs(self.log_dir, exist_ok=True)
        if wandb is None:
            raise RuntimeError("WandbLogger is not available")

        project = os.environ.get("WANDB_PROJECT", None)
        self.wand_run = wandb.init(
            project=project,
            name=trainer.name,
            dir=self.log_dir,
            # config={
            # 'learning_rate': trainer.learning_rate, # TODO get learning rate from the optimizer
            # TODO parse more of the config from the trainer
            # },
        )

        if trainer.name is None:
            if os.environ.get("WANDB_MODE") == "offline":
                trainer.name = f"offline-{datetime.now()}"
            else:
                trainer.name = self.wand_run.name

        self.log_image_interval = trainer.log_image_interval

        wandb.watch(trainer.model)

    def _log_images(self, step, x, y, prediction, name, gradients=None):

        selection = np.s_[0] if x.ndim == 4 else np.s_[0, :, x.shape[2] // 2]

        image = normalize_im(x[selection].cpu())
        grid_image, grid_name = make_grid_image(image, y, prediction, selection, gradients)

        # to numpy and channel last
        image = image.numpy().transpose((1, 2, 0))
        wandb.log({f"images_{name}/input": [wandb.Image(image, caption="Input Data")]}, step=step)

        grid_image = grid_image.numpy().transpose((1, 2, 0))

        wandb.log({f"images_{name}/{grid_name}": [wandb.Image(grid_image, caption=grid_name)]}, step=step)

    def log_train(self, step, loss, lr, x, y, prediction, log_gradients=False):
        wandb.log({"train/loss": loss}, step=step)
        if step % self.log_image_interval == 0:
            gradients = prediction.grad if log_gradients else None
            self._log_images(step, x, y, prediction, "train", gradients=gradients)

    def log_validation(self, step, metric, loss, x, y, prediction):
        wandb.log({"validation/loss": loss, "validation/metric": metric}, step=step)
        self._log_images(step, x, y, prediction, "validation")

    def get_wandb(self):
        return wandb
