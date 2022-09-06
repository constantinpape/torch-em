try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore


class TorchEmLogger:
    def __init__(self, trainer, save_root, **kwargs):
        self.trainer = trainer
        self.save_root = save_root

    def log_train(self, step, loss, lr, x, y, prediction, log_gradients=False):
        raise NotImplementedError

    def log_validation(self, step, metric, loss, x, y, prediction):
        raise NotImplementedError
