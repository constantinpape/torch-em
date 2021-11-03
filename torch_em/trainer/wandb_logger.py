import os
from datetime import datetime
from typing import Optional

import numpy as np

from .logger_base import TorchEmLogger
from .tensorboard_logger import make_grid_image, normalize_im

try:
    import wandb
except ImportError:
    wandb = None

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore


class WandbLogger(TorchEmLogger):
    def __init__(
        self,
        trainer,
        *,
        project_name: Optional[str] = None,
        log_model: Optional[Literal["gradients", "parameters", "all"]] = "all",
        log_model_freq: int = 1,
        log_model_graph: bool = True,
        mode: Literal["online", "offline", "disabled"] = "online",
        **unused_kwargs,
    ):
        if wandb is None:
            raise RuntimeError("WandbLogger is not available")

        super().__init__(trainer)

        self.log_dir = "./logs"
        os.makedirs(self.log_dir, exist_ok=True)

        self.wand_run = wandb.init(
            project=project_name, name=trainer.name, dir=self.log_dir, mode=mode, config=trainer.init_data
        )

        if trainer.name is None:
            if mode == "online":
                trainer.name = self.wand_run.name
            elif mode in ("offline", "disabled"):
                trainer.name = f"{mode}_{datetime.now():%Y-%m-%d_%H-%M-%S}"
            else:
                raise ValueError(mode)

        self.log_image_interval = trainer.log_image_interval

        wandb.watch(trainer.model, log=log_model, log_freq=log_model_freq, log_graph=log_model_graph)

    def _log_images(self, step, x, y, prediction, name, gradients=None):

        selection = np.s_[0] if x.ndim == 4 else np.s_[0, :, x.shape[2] // 2]

        image = normalize_im(x[selection].cpu())
        grid_image, grid_name = make_grid_image(image, y, prediction, selection, gradients)

        # to numpy and channel last
        image = image.numpy().transpose((1, 2, 0))
        wandb.log({f"images_{name}/input": [wandb.Image(image, caption="Input Data")]}, step=step)

        grid_image = grid_image.numpy().transpose((1, 2, 0))

        wandb.log({grid_name: [wandb.Image(grid_image, caption=grid_name)]}, step=step)

    def log_train(self, step, loss, lr, x, y, prediction, log_gradients=False):
        wandb.log({"train/loss": loss}, step=step)
        if step % self.log_image_interval == 0:
            gradients = prediction.grad if log_gradients else None
            self._log_images(step, x, y, prediction, "train", gradients=gradients)

    def log_validation(self, step, metric, loss, x, y, prediction):
        wandb.log({"validation/loss": loss, "validation/metric": metric}, step=step)
        if loss < self.wand_run.summary["validation/loss"]:
            self.wand_run.summary["validation/loss"] = loss

        # like for loss, we consider smaller metrics better
        if metric < self.wand_run.summary["validation/metric"]:
            self.wand_run.summary["validation/metric"] = metric

        self._log_images(step, x, y, prediction, "validation")

    def get_wandb(self):
        return wandb
