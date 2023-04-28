import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from matplotlib.backends.backend_agg import FigureCanvasAgg
from sklearn.metrics import ConfusionMatrixDisplay
from torch_em.trainer.logger_base import TorchEmLogger


def confusion_matrix(y_true, y_pred, class_labels=None, title=None, save_path=None, **plot_kwargs):
    fig, ax = plt.subplots(1)

    if save_path is None:
        canvas = FigureCanvasAgg(fig)

    disp = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, normalize="true", display_labels=class_labels
    )
    disp.plot(ax=ax, **plot_kwargs)

    if title is not None:
        ax.set_title(title)
    if save_path is not None:
        plt.savefig(save_path)
        return

    canvas.draw()
    image = np.asarray(canvas.buffer_rgba())[..., :3]
    image = image.transpose((2, 0, 1))
    plt.close()
    return image


# TODO normalization and stuff
# TODO get the class names
def make_grid(images, target=None, prediction=None, images_per_row=8, **kwargs):
    assert images.ndim == 4
    assert images.shape[1] in (1, 3)

    n_images = images.shape[0]
    n_rows = n_images // images_per_row
    if n_images % images_per_row != 0:
        n_rows += 1

    images = images.detach().cpu().numpy()
    if target is not None:
        target = target.detach().cpu().numpy()
    if prediction is not None:
        prediction = prediction.max(1)[1].detach().cpu().numpy()

    fig, axes = plt.subplots(n_rows, images_per_row)
    canvas = FigureCanvasAgg(fig)
    for r in range(n_rows):
        for c in range(images_per_row):
            i = r * images_per_row + c
            ax = axes[r, c]
            ax.set_axis_off()
            im = images[i]
            im = im.transpose((1, 2, 0))
            if im.shape[-1] == 3:  # rgb
                ax.imshow(im)
            else:
                ax.imshow(im[..., 0], cmap="gray")

            if target is None and prediction is None:
                continue

            # TODO get the class name, and if we have both target
            # and prediction check whether they agree or not and do stuff
            title = ""
            if target is not None:
                title += f"t: {target[i]} "
            if prediction is not None:
                title += f"p: {prediction[i]}"
            ax.set_title(title, fontsize=8)

    canvas.draw()
    image = np.asarray(canvas.buffer_rgba())[..., :3]
    image = image.transpose((2, 0, 1))
    plt.close()
    return image


class ClassificationLogger(TorchEmLogger):
    def __init__(self, trainer, save_root, **unused_kwargs):
        super().__init__(trainer, save_root)
        self.log_dir = f"./logs/{trainer.name}" if save_root is None else\
            os.path.join(save_root, "logs", trainer.name)
        os.makedirs(self.log_dir, exist_ok=True)

        self.tb = torch.utils.tensorboard.SummaryWriter(self.log_dir)
        self.log_image_interval = trainer.log_image_interval

    def add_image(self, x, y, pred, name, step):
        scale_each = False
        marker = make_grid(x[:, 0:1], y, pred, padding=4, normalize=True, scale_each=scale_each)
        self.tb.add_image(tag=f"{name}/marker", img_tensor=marker, global_step=step)
        nucleus = make_grid(x[:, 1:2], padding=4, normalize=True, scale_each=scale_each)
        self.tb.add_image(tag=f"{name}/nucleus", img_tensor=nucleus, global_step=step)
        mask = make_grid(x[:, 2:], padding=4)
        self.tb.add_image(tag=f"{name}/mask", img_tensor=mask, global_step=step)

    def log_train(self, step, loss, lr, x, y, prediction, log_gradients=False):
        self.tb.add_scalar(tag="train/loss", scalar_value=loss, global_step=step)
        self.tb.add_scalar(tag="train/learning_rate", scalar_value=lr, global_step=step)
        if step % self.log_image_interval == 0:
            self.add_image(x, y, prediction, "train", step)

    def log_validation(self, step, metric, loss, x, y, prediction, y_true=None, y_pred=None):
        self.tb.add_scalar(tag="validation/loss", scalar_value=loss, global_step=step)
        self.tb.add_scalar(tag="validation/metric", scalar_value=metric, global_step=step)
        self.add_image(x, y, prediction, "validation", step)
        if y_true is not None and y_pred is not None:
            cm = confusion_matrix(y_true, y_pred)
            self.tb.add_image(tag="validation/confusion_matrix", img_tensor=cm, global_step=step)
