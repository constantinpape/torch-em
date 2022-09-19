import torch
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
    def _crop(prediction, target, mask, channel_dim):
        mask = mask.type(torch.bool)
        pred = []
        trgt = []
        n_channels = prediction.shape[channel_dim]
        # for now only support same mask for all channels
        # to be able to stack/concat afterwards
        # TODO: make this more efficient, and for different
        # masks per channel
        assert mask.shape[channel_dim] == 1
        for c in range(n_channels):
            p = torch.index_select(prediction, channel_dim, torch.LongTensor([c]))
            t = torch.index_select(target, channel_dim, torch.LongTensor([c]))
            pred.append(p[mask])
            trgt.append(t[mask])
        # transpose to have N x C
        # needed in torch_em.loss.dice.flatten_samples
        pred = torch.stack(pred).transpose(0, 1)
        trgt = torch.stack(trgt).transpose(0, 1)
        return pred, trgt

    def _multiply(prediction, target, mask, channel_dim):
        prediction = prediction * mask
        target = target * mask
        return prediction, target

    MASKING_FUNCS = {
        "crop": _crop,
        "multiply": _multiply,
    }

    def __init__(self, masking_method="crop", channel_dim=1):
        if masking_method not in self.MASKING_FUNCS.keys():
            raise ValueError(f"{masking_method} is not available, please use one of {list(self.MASKING_FUNCS.keys())}.")
        self.masking_func = self.MASKING_FUNCS[masking_method]
        self.channel_dim = channel_dim

        self.init_kwargs = {
            "masking_method": masking_method,
            "channel_dim": channel_dim,
        }

    def __call__(self, prediction, target, mask):
        mask.requires_grad = False
        return self.masking_func(prediction, target, mask, self.channel_dim)


class ApplyAndRemoveMask(ApplyMask):
    def __call__(self, prediction, target):
        assert target.dim() == prediction.dim(), f"{target.dim()}, {prediction.dim()}"
        assert target.size(1) == 2 * prediction.size(1), f"{target.size(1)}, {prediction.size(1)}"
        assert target.shape[2:] == prediction.shape[2:], f"{str(target.shape)}, {str(prediction.shape)}"
        seperating_channel = target.size(1) // 2
        mask = target[:, seperating_channel:]
        target = target[:, :seperating_channel]
        prediction, target = super().__call__(prediction, target, mask)
        return prediction, target


class MaskIgnoreLabel(ApplyMask):
    def __init__(self, ignore_label=-1, masking_method="crop", channel_dim=1):
        super().__init__(masking_method, channel_dim)
        self.ignore_label = ignore_label
        self.init_kwargs["ignore_label"] = ignore_label

    def __call__(self, prediction, target):
        mask = (target != self.ignore_label)
        prediction, target = super().__call__(prediction, target, mask)
        return prediction, target
