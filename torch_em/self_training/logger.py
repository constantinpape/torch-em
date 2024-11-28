import os

import torch_em

from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter


class SelfTrainingTensorboardLogger(torch_em.trainer.logger_base.TorchEmLogger):
    def __init__(self, trainer, save_root, **unused_kwargs):
        super().__init__(trainer, save_root)
        self.my_root = save_root
        self.log_dir = f"./logs/{trainer.name}" if self.my_root is None else\
            os.path.join(self.my_root, "logs", trainer.name)
        os.makedirs(self.log_dir, exist_ok=True)

        self.tb = SummaryWriter(self.log_dir)
        self.log_image_interval = trainer.log_image_interval

    def _add_supervised_images(self, step, name, x, y, pred):
        if x.ndim == 5:
            assert y.ndim == pred.ndim == 5
            zindex = x.shape[2] // 2
            x, y, pred = x[:, :, zindex], y[:, :, zindex], pred[:, :, zindex]

        grid = make_grid(
            [torch_em.transform.raw.normalize(x[0]), y[0, 0:1], pred[0, 0:1]],
            padding=8
        )
        self.tb.add_image(tag=f"{name}/supervised/input-labels-prediction", img_tensor=grid, global_step=step)

    def _add_unsupervised_images(self, step, name, x1, x2, pred, pseudo_labels, label_filter):
        if x1.ndim == 5:
            assert x2.ndim == pred.ndim == pseudo_labels.ndim == 5
            zindex = x1.shape[2] // 2
            x1, x2, pred = x1[:, :, zindex], x2[:, :, zindex], pred[:, :, zindex]
            pseudo_labels = pseudo_labels[:, :, zindex]
            if label_filter is not None:
                assert label_filter.ndim == 5
                label_filter = label_filter[:, :, zindex]

        images = [
            torch_em.transform.raw.normalize(x1[0]),
            torch_em.transform.raw.normalize(x2[0]),
            pred[0, 0:1], pseudo_labels[0, 0:1],
        ]
        im_name = f"{name}/unsupervised/aug1-aug2-prediction-pseudolabels"
        if label_filter is not None:
            images.append(label_filter[0, 0:1])
            name += "-labelfilter"
        grid = make_grid(images, nrow=2, padding=8)
        self.tb.add_image(tag=im_name, img_tensor=grid, global_step=step)

    def log_combined_loss(self, step, loss):
        self.tb.add_scalar(tag="train/combined_loss", scalar_value=loss, global_step=step)

    def log_lr(self, step, lr):
        self.tb.add_scalar(tag="train/learning_rate", scalar_value=lr, global_step=step)

    def log_train_supervised(self, step, loss, x, y, pred):
        self.tb.add_scalar(tag="train/supervised/loss", scalar_value=loss, global_step=step)
        if step % self.log_image_interval == 0:
            self._add_supervised_images(step, "validation", x, y, pred)

    def log_validation_supervised(self, step, metric, loss, x, y, pred):
        self.tb.add_scalar(tag="validation/supervised/loss", scalar_value=loss, global_step=step)
        self.tb.add_scalar(tag="validation/supervised/metric", scalar_value=metric, global_step=step)
        self._add_supervised_images(step, "validation", x, y, pred)

    def log_train_unsupervised(self, step, loss, x1, x2, pred, pseudo_labels, label_filter=None):
        self.tb.add_scalar(tag="train/unsupervised/loss", scalar_value=loss, global_step=step)
        if step % self.log_image_interval == 0:
            self._add_unsupervised_images(step, "validation", x1, x2, pred, pseudo_labels, label_filter)

    def log_validation_unsupervised(self, step, metric, loss, x1, x2, pred, pseudo_labels, label_filter=None):
        self.tb.add_scalar(tag="validation/unsupervised/loss", scalar_value=loss, global_step=step)
        self.tb.add_scalar(tag="validation/unsupervised/metric", scalar_value=metric, global_step=step)
        self._add_unsupervised_images(step, "validation", x1, x2, pred, pseudo_labels, label_filter)

    def log_validation(self, step, metric, loss, xt, xt1, xt2, y, z, gt, samples, gt_metric=None):
        self.tb.add_scalar(tag="validation/loss", scalar_value=loss, global_step=step)
        self.tb.add_scalar(tag="validation/metric", scalar_value=metric, global_step=step)
        if gt_metric is not None:
            self.tb.add_scalar(tag="validation/gt_metric", scalar_value=gt_metric, global_step=step)


class ProbabilisticUNetTrainerLogger(torch_em.trainer.logger_base.TorchEmLogger):
    def __init__(self, trainer, save_root, **unused_kwargs):
        super().__init__(trainer, save_root)
        self.log_dir = f"./logs/{trainer.name}" if save_root is None else\
            os.path.join(save_root, "logs", trainer.name)
        os.makedirs(self.log_dir, exist_ok=True)

        self.tb = torch.utils.tensorboard.SummaryWriter(self.log_dir)
        self.log_image_interval = trainer.log_image_interval

    def add_image(self, x, y, samples, name, step):
        # NOTE: we only show the first tensor per batch for all images
        self.tb.add_image(tag=f"{name}/input", img_tensor=x[0], global_step=step)
        self.tb.add_image(tag=f"{name}/target", img_tensor=y[0], global_step=step)
        sample_grid = make_grid([sample[0] for sample in samples], nrow=4, padding=4)
        self.tb.add_image(tag=f"{name}/samples", img_tensor=sample_grid, global_step=step)

    def log_train(self, step, loss, lr, x, y, samples):
        self.tb.add_scalar(tag="train/loss", scalar_value=loss, global_step=step)
        self.tb.add_scalar(tag="train/learning_rate", scalar_value=lr, global_step=step)
        if step % self.log_image_interval == 0:
            self.add_image(x, y, samples, "train", step)

    def log_validation(self, step, metric, loss, x, y, samples):
        self.tb.add_scalar(tag="validation/loss", scalar_value=loss, global_step=step)
        self.tb.add_scalar(tag="validation/metric", scalar_value=metric, global_step=step)
        self.add_image(x, y, samples, "validation", step)
