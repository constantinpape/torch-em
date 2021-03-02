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

    def apply_transform(self, prediction, target):
        # check if the tensors (prediction and target are lists)
        # if they are, apply the transform to each element inidvidually
        if isinstance(prediction, (list, tuple)):
            assert isinstance(target, (list, tuple))
            transformed_prediction, transformed_target = [], []
            for pred, targ in zip(prediction, target):
                tr_pred, tr_targ = self.transform(pred, targ)
                transformed_prediction.append(tr_pred)
                transformed_target.append(tr_targ)
            return transformed_prediction, transformed_target
        # tensor input
        else:
            prediction, target = self.transform(prediction, target)
            return prediction, target

    def forward(self, prediction, target):
        prediction, target = self.apply_transform(prediction, target)
        loss = self.loss(prediction, target)
        return loss


#
# Loss transformations
#

class ApplyAndRemoveMask:
    def __call__(self, prediction, target):
        assert target.dim() == prediction.dim(), f"{target.dim()}, {prediction.dim()}"
        assert target.size(1) == 2 * prediction.size(1), f"{target.size(1)}, {prediction.size(1)}"
        assert target.shape[2:] == prediction.shape[2:], f"{str(target.shape)}, {str(prediction.shape)}"

        seperating_channel = target.size(1) // 2
        mask = target[:, seperating_channel:]
        target = target[:, :seperating_channel]
        mask.requires_grad = False

        # mask the prediction
        prediction = prediction * mask
        return prediction, target
