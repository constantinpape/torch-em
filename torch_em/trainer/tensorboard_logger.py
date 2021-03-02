import numpy as np
import torch
from torchvision.utils import make_grid

# tensorboard import only works if tensobard package is available, so
# we wrap this in a try except
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from ..util import ensure_tensor


class TensorboardLogger:
    def __init__(self, log_dir, log_image_interval):
        if SummaryWriter is None:
            msg = "Need tensorboard package to use the logger. Install it via conda install -c conda-forge tensorboard"
            raise RuntimeError(msg)
        self.tb = torch.utils.tensorboard.SummaryWriter(log_dir)
        self.log_image_interval = log_image_interval

    def log_images(self, step, x, y, prediction, name, gradients=None):

        def _normalize(im):
            im = ensure_tensor(im, dtype=torch.float32)
            im -= im.min()
            im /= im.max()
            return im

        selection = np.s_[0] if x.ndim == 4 else np.s_[0, :, x.shape[2] // 2]

        image = _normalize(x[selection].cpu())
        self.tb.add_image(tag=f'{name}/input',
                          img_tensor=image,
                          global_step=step)

        target_image = _normalize(y[selection].cpu())
        pred_image = _normalize(prediction[selection].detach().cpu())

        n_channels = target_image.shape[0]
        if n_channels == 1:
            nrow = 8
            images = [image, target_image, pred_image]
        else:
            nrow = n_channels
            images = nrow * [image]
            images += [channel.unsqueeze(0) for channel in target_image]
            images += [channel.unsqueeze(0) for channel in pred_image]

        im_name = f'{name}/raw_targets_predictions'
        if gradients is not None:
            im_name += '_gradients'
            grad_image = _normalize(gradients[selection].cpu())
            if n_channels == 1:
                images.append(grad_image)
            else:
                images += [channel.unsqueeze(0) for channel in grad_image]

        im = make_grid(images, nrow=nrow, padding=4)
        self.tb.add_image(tag=im_name,
                          img_tensor=im,
                          global_step=step)

    def log_train(self, step, loss, lr, x, y, prediction, log_gradients=False):
        self.tb.add_scalar(tag='train/loss', scalar_value=loss, global_step=step)
        self.tb.add_scalar(tag='train/learning_rate', scalar_value=lr, global_step=step)
        if step % self.log_image_interval == 0:
            gradients = prediction.grad if log_gradients else None
            self.log_images(step, x, y, prediction, 'train',
                            gradients=gradients)

    def log_validation(self, step, metric, loss, x, y, prediction):
        self.tb.add_scalar(tag='validation/loss', scalar_value=loss, global_step=step)
        self.tb.add_scalar(tag='validation/metric', scalar_value=metric, global_step=step)
        self.log_images(step, x, y, prediction, 'validation')
