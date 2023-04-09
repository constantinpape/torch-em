import torch.nn as nn
import torch_em


class DefaultSelfTrainingLoss(nn.Module):
    """Loss function for self training.

    Parameters:
        loss [nn.Module] - the loss function to be used. (default: torch_em.loss.DiceLoss)
        activation [nn.Module, callable] - the activation function to be applied to the prediction
            before passing it to the loss. (default: None)
    """
    def __init__(self, loss=torch_em.loss.DiceLoss(), activation=None):
        super().__init__()
        self.activation = activation
        self.loss = loss
        # TODO serialize the class names and kwargs instead
        self.init_kwargs = {}

    def __call__(self, model, input_, labels, label_filter=None):
        prediction = model(input_)
        if self.activation is not None:
            prediction = self.activation(prediction)
        if label_filter is None:
            loss = self.loss(prediction, labels)
        else:
            loss = self.loss(prediction * label_filter, labels * label_filter)
        return loss


class DefaultSelfTrainingLossAndMetric(nn.Module):
    """Loss and metric function for self training.

    Parameters:
        loss [nn.Module] - the loss function to be used. (default: torch_em.loss.DiceLoss)
        metric [nn.Module] - the metric function to be used. (default: torch_em.loss.DiceLoss)
        activation [nn.Module, callable] - the activation function to be applied to the prediction
            before passing it to the loss. (default: None)
    """
    def __init__(self, loss=torch_em.loss.DiceLoss(), metric=torch_em.loss.DiceLoss(), activation=None):
        super().__init__()
        self.activation = activation
        self.loss = loss
        self.metric = metric
        # TODO serialize the class names and dicts instead
        self.init_kwargs = {}

    def __call__(self, model, input_, labels, label_filter=None):
        prediction = model(input_)
        if self.activation is not None:
            prediction = self.activation(prediction)
        if label_filter is None:
            loss = self.loss(prediction, labels)
        else:
            loss = self.loss(prediction * label_filter, labels * label_filter)
        metric = self.metric(prediction, labels)
        return loss, metric
