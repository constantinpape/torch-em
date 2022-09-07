import torch.nn as nn


class LossWrapper(nn.Module):
    """ Wrapper around a torch loss function.

    Applies transformations to prediction and/or target before passing it to the loss.
    """
    def __init__(self, loss, transform):
        super().__init__()
        self.loss = loss

        if not callable(transform):
            raise ValueError("transform has to be callable.")
        self.transform = transform
        self.init_kwargs = {'loss': loss, 'transform': transform}

    def apply_transform(self, prediction, target, **kwargs):
        # check if the tensors (prediction and target are lists)
        # if they are, apply the transform to each element inidvidually
        if isinstance(prediction, (list, tuple)):
            assert isinstance(target, (list, tuple))
            transformed_prediction, transformed_target = [], []
            for pred, targ in zip(prediction, target):
                tr_pred, tr_targ = self.transform(pred, targ, **kwargs)
                transformed_prediction.append(tr_pred)
                transformed_target.append(tr_targ)
            return transformed_prediction, transformed_target
        # tensor input
        else:
            prediction, target = self.transform(prediction, target, **kwargs)
            return prediction, target

    def forward(self, prediction, target, **kwargs):
        prediction, target = self.apply_transform(prediction, target, **kwargs)
        loss = self.loss(prediction, target)
        return loss


#
# Loss transformations
#


class ApplyMask:
    def __call__(self, prediction, target, mask):
        mask.requires_grad = False
        prediction = prediction * mask
        target = target * mask
        return prediction, target


class ApplyAndRemoveMask:
    def __call__(self, prediction, target):
        assert target.dim() == prediction.dim(), f"{target.dim()}, {prediction.dim()}"
        assert target.size(1) == 2 * prediction.size(1), f"{target.size(1)}, {prediction.size(1)}"
        assert target.shape[2:] == prediction.shape[2:], f"{str(target.shape)}, {str(prediction.shape)}"
        seperating_channel = target.size(1) // 2
        mask = target[:, seperating_channel:]
        target = target[:, :seperating_channel]
        prediction, target = ApplyMask()(prediction, target, mask)
        return prediction, target


class MaskIgnoreLabel:
    def __init__(self, ignore_label=-1):
        self.ignore_label = ignore_label

    def __call__(self, prediction, target):
        mask = (target != self.ignore_label)
        prediction, target = ApplyMask()(prediction, target, mask)
        return prediction, target
