import os

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.backends.backend_agg import FigureCanvasAgg
from sklearn.metrics import ConfusionMatrixDisplay
from torch.utils.tensorboard import SummaryWriter
from torch_em.trainer.logger_base import TorchEmLogger
from torch_em.transform.raw import normalize


def confusion_matrix(y_true, y_pred, class_labels=None, title=None, save_path=None, **plot_kwargs):
    """@private
    """
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


def make_grid(images, target=None, prediction=None, images_per_row=8, **kwargs):
    """@private
    """
    assert images.ndim in (4, 5)
    assert images.shape[1] in (1, 3), f"{images.shape}"

    if images.ndim == 5:
        is_3d = True
        z = images.shape[2] // 2
    else:
        is_3d = False

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
            if i == len(images):
                break
            ax = axes[r, c] if n_rows > 1 else axes[r]
            ax.set_axis_off()
            im = images[i, :, z] if is_3d else images[i]
            im = im.transpose((1, 2, 0))
            im = normalize(im, axis=(0, 1))
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
    """Logger for classification trainer.

    Args:
        trainer: The trainer instance.
        save_root: Root folder for saving the checkpoints and logs.
    """
    def __init__(self, trainer, save_root: str, **unused_kwargs):
        super().__init__(trainer, save_root)
        self.log_dir = f"./logs/{trainer.name}" if save_root is None else\
            os.path.join(save_root, "logs", trainer.name)
        os.makedirs(self.log_dir, exist_ok=True)

        self.tb = SummaryWriter(self.log_dir)
        self.log_image_interval = trainer.log_image_interval

    def add_image(self, x, y, pred, name, step):
        """@private
        """
        scale_each = False
        grid = make_grid(x, y, pred, padding=4, normalize=True, scale_each=scale_each)
        self.tb.add_image(tag=f"{name}/images_and_predictions", img_tensor=grid, global_step=step)

    def log_train(self, step, loss, lr, x, y, prediction, log_gradients=False):
        """@private
        """
        self.tb.add_scalar(tag="train/loss", scalar_value=loss, global_step=step)
        self.tb.add_scalar(tag="train/learning_rate", scalar_value=lr, global_step=step)
        if step % self.log_image_interval == 0:
            self.add_image(x, y, prediction, "train", step)

    def log_validation(self, step, metric, loss, x, y, prediction, y_true=None, y_pred=None):
        """@private
        """
        self.tb.add_scalar(tag="validation/loss", scalar_value=loss, global_step=step)
        self.tb.add_scalar(tag="validation/metric", scalar_value=metric, global_step=step)
        self.add_image(x, y, prediction, "validation", step)
        if y_true is not None and y_pred is not None:
            cm = confusion_matrix(y_true, y_pred)
            self.tb.add_image(tag="validation/confusion_matrix", img_tensor=cm, global_step=step)
